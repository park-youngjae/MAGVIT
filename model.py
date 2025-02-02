import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
import argparse, os, sys, datetime, glob, importlib

class ResBlockX(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockX, self).__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),  
            nn.SiLU(), 
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels), 
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.block(x) + x

class ResBlockXY(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockXY, self).__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels), 
            nn.SiLU(), 
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),  
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.block(x) 
        skip_out = self.skip(x) 
        return out + skip_out  



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.initial_conv = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        self.res_blocks_64 = nn.Sequential(
            ResBlockX(64, 64),
            ResBlockX(64, 64)
        )
        self.down1 = nn.AvgPool3d(2)
        self.res_blocks_128_1 = nn.Sequential(
            ResBlockXY(64, 128),
            ResBlockX(128, 128)
        )
        self.down2 = nn.AvgPool3d(2)
        self.res_blocks_128_2 = nn.Sequential(
            ResBlockXY(128, 128),
            ResBlockX(128, 128)
        )
        self.down3 = nn.AvgPool3d((1, 2, 2))
        self.res_blocks_256 = nn.Sequential(
            ResBlockXY(128, 256),
            ResBlockX(256, 256),
            ResBlockX(256, 256),
            ResBlockX(256, 256)
        )
        self.final_process = nn.Sequential(
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv3d(256, 256, kernel_size=1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.initial_conv(x)
        x = self.res_blocks_64(x)
        x = self.down1(x)
        x = self.res_blocks_128_1(x)
        x = self.down2(x)
        x = self.res_blocks_128_2(x)
        x = self.down3(x)
        x = self.res_blocks_256(x)
        x = self.final_process(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.initial_conv = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.res_blocks_256 = nn.Sequential(
            ResBlockX(256, 256),
            ResBlockX(256, 256),
            ResBlockX(256, 256),
            ResBlockX(256, 256)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.res_blocks_128_1 = nn.Sequential(
            ResBlockXY(256, 128),
            ResBlockX(128, 128)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        )
        self.res_blocks_128_2 = nn.Sequential(
            ResBlockX(128, 128),
            ResBlockX(128, 128)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        )        
        self.final_res_block = nn.Sequential(
            ResBlockXY(128, 64),
            ResBlockX(64, 64)
        )
        self.final_conv = nn.Sequential(
            nn.GroupNorm(16, 64),
            nn.SiLU(),
            nn.Conv3d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks_256(x)
        x = self.up1(x)
        x = self.res_blocks_128_1(x)
        x = self.up2(x)
        x = self.res_blocks_128_2(x)
        x = self.up3(x)
        x = self.final_res_block(x)
        x = self.final_conv(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        print("vq-z",x.shape)
        flat_x = x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])
        distances = (torch.sum(flat_x**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_x, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.embedding.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)
        return quantized

class VectorQuantizer2(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    def __init__(self, n_e=1024, e_dim=256, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random", "extra", or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed += 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        # reshape z -> (batch, time, height, width, channel) and flatten
        z = rearrange(z, 'b t c h w -> b t h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)  # flatten to (b*t*h*w, e_dim)
        
        # Calculate distances between z and embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b t h w c -> b t c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = self.remap_to_used(min_encoding_indices.view(z.shape[0], -1))
            min_encoding_indices = min_encoding_indices.view(-1, 1)  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.view(z_q.shape[0], z_q.shape[1], z_q.shape[3], z_q.shape[4])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer(1024, 256)
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        quantized = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return encoded, quantized, decoded

class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockDown, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.skip = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2), 
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x_skip = self.skip(x)
        x = self.conv1(x)
        x = self.avg_pool(x)
        x = self.conv2(x)

        return x + x_skip

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.down_blocks = nn.Sequential(
            ResBlockDown(64, 128),
            ResBlockDown(128, 256),
            ResBlockDown(256, 256),
            ResBlockDown(256, 256),
            # ResBlockDown(256, 256)
        )

        self.final_conv = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 1)
        )


    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.initial_conv(x)
        x = self.down_blocks(x)
        x = self.final_conv(x)
        x = self.classifier(x)

        return x


class VQModel(pl.LightningModule):
    def __init__(self,
                 n_embed=1024,
                 embed_dim=256,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quantize = VectorQuantizer2(1024, 256, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)


    def forward(self, input):
        enc = self.encoder(input)
        z_q, q_loss, _ = self.quantize(enc)
        dec = self.decoder(z_q)
        return dec, q_loss




class MultiTaskMaskedTokenModeling(nn.Module):
    def __init__(self, num_tokens, hidden_dim, num_heads, num_layers, num_classes, mask_token_id):
        super(MultiTaskMaskedTokenModeling, self).__init__()
        self.hidden_dim = hidden_dim
        self.mask_token_id = mask_token_id

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )

        # Token and class embeddings
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        self.class_embedding = nn.Embedding(num_classes, hidden_dim)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_tokens)

    def generate_masked_input(self, tokens, condition_tokens, s, s_star, pad_token_id):
        """
        Generate masked input following Equation 2 in MAGVIT.
        m(zi | ˜zi) = 
        ˜zi if si ≤ s* ∧ ¬ispad(˜zi)
        [MASK] if si ≤ s* ∧ ispad(˜zi)
        zi if si > s*
        """
        batch_size, seq_len = tokens.size()
        masked_input = tokens.clone()

        for i in range(batch_size):
            for j in range(seq_len):
                if s[i, j] <= s_star:  # si ≤ s*
                    if condition_tokens[i, j] == pad_token_id:  # ispad(˜zi)
                        masked_input[i, j] = self.mask_token_id  # Replace with [MASK]
                    else:  # ¬ispad(˜zi)
                        masked_input[i, j] = condition_tokens[i, j]  # Replace with ˜zi
                else:  # si > s*
                    masked_input[i, j] = tokens[i, j]  # Keep original zi

        return masked_input

    def forward(self, tokens, task_ids, condition_tokens, mask_ratio=0.15, pad_token_id=0):
        """
        Forward pass with COMMIT masking strategy.
        """
        # Generate masked input
        masked_tokens = self.generate_masked_input(tokens, condition_tokens, mask_ratio, pad_token_id)

        # Embed tokens, positions, and task-specific embeddings
        token_embeddings = self.token_embedding(masked_tokens) + self.positional_encoding
        class_embeddings = self.class_embedding(task_ids).unsqueeze(1)
        embeddings = token_embeddings + class_embeddings.expand_as(token_embeddings)

        # Transformer forward pass
        transformer_output = self.transformer(embeddings, embeddings)

        # Project output to token space
        logits = self.output_projection(transformer_output)
        return logits

    def non_autoregressive_decode(self, condition_tokens, task_ids, steps, temperature, gamma, pad_token_id):
        """
        Non-autoregressive decoding following Algorithm 1 in MAGVIT.
        """
        batch_size, seq_len = condition_tokens.size()
        predicted_tokens = torch.full_like(condition_tokens, self.mask_token_id)  # Start with all [MASK]
        s = torch.zeros_like(predicted_tokens, dtype=torch.float)
        s_star = torch.ones(batch_size, seq_len, device=condition_tokens.device)

        for t in range(steps):
            # Generate masked input based on COMMIT strategy
            masked_input = self.generate_masked_input(predicted_tokens, condition_tokens, mask_ratio=s_star, pad_token_id=pad_token_id)

            # Embed tokens and task-specific embeddings
            token_embeddings = self.token_embedding(masked_input) + self.positional_encoding
            class_embeddings = self.class_embedding(task_ids).unsqueeze(1)
            embeddings = token_embeddings + class_embeddings.expand_as(token_embeddings)

            # Transformer forward pass
            transformer_output = self.transformer(embeddings, embeddings)

            # Predict tokens
            logits = self.output_projection(transformer_output)
            predicted_tokens = torch.argmax(logits, dim=-1)

            # Update confidence scores
            probabilities = F.softmax(logits / temperature, dim=-1)
            s = torch.max(probabilities, dim=-1).values
            s_star = torch.quantile(s, gamma, dim=-1, keepdim=True)

        return predicted_tokens


class MAGVITL(nn.Module):
    def __init__(self):
        super(MAGVITL, self).__init__()
        self.encoder = SpatialTemporalEncoder(3, 64, 1024, 512)
        self.task_modeling = MultiTaskMaskedTokenModeling(1024, 512, 8, 6, 10)

    def forward(self, x, task_id):
        vq_loss, quantized = self.encoder(x)
        predicted = self.task_modeling(quantized, task_id)
        return vq_loss, predicted
