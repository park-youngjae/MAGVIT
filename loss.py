import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

# Perceptual Loss using VGG-like network
class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.criterion = nn.MSELoss()

    def forward(self, reconstructed, target):
        # reconstructed, target: [B, T, C, H, W]
        B, T, C, H, W = reconstructed.shape

        # Reshape tensors to merge batch and temporal dimensions
        reconstructed = reconstructed.contiguous().view(B * T, C, H, W)  # [B*T, C, H, W]
        target = target.contiguous().view(B * T, C, H, W)  # [B*T, C, H, W]

        # Extract features using VGG
        with torch.no_grad():  # No gradients needed for feature extraction
            reconstructed_features = self.feature_extractor(reconstructed)  # [B*T, F, H', W']
            target_features = self.feature_extractor(target)  # [B*T, F, H', W']

        # Compute MSE loss between features
        loss = self.criterion(reconstructed_features, target_features)

        return loss

def gan_loss(logits, is_real):
    targets = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
    return nn.BCEWithLogitsLoss()(logits, targets)

# LeCam Regularization (optional for GAN stability)
def lecam_regularization(discriminator, real_data, fake_data):
    real_logits = discriminator(real_data)
    fake_logits = discriminator(fake_data)
    reg = torch.mean(torch.abs(real_logits - fake_logits))
    return reg

def compute_gradient_penalty(discriminator, real_data, fake_data, device):
    real_data.requires_grad = True
    real_outputs = discriminator(real_data)
    real_gradients = torch.autograd.grad(
        outputs=real_outputs,
        inputs=real_data,
        grad_outputs=torch.ones(real_outputs.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    real_gradients_norm = torch.sqrt(torch.sum(real_gradients ** 2, dim=1) + 1e-12)
    return torch.mean((real_gradients_norm - 1) ** 2)
