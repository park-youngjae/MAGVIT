import pdb
import lpips 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Perceptual Loss using VGG-like network
# class PerceptualLoss(nn.Module):
#     def __init__(self, feature_extractor):
#         super(PerceptualLoss, self).__init__()
#         self.feature_extractor = feature_extractor
#         self.criterion = nn.MSELoss()

#     def forward(self, reconstructed, target):
#         # reconstructed, target: [B, T, C, H, W]
#         B, T, C, H, W = reconstructed.shape

#         # Reshape tensors to merge batch and temporal dimensions
#         reconstructed = reconstructed.contiguous().view(B * T, C, H, W)  # [B*T, C, H, W]
#         target = target.contiguous().view(B * T, C, H, W)  # [B*T, C, H, W]

#         # Extract features using VGG
#         with torch.no_grad():  # No gradients needed for feature extraction
#             reconstructed_features = self.feature_extractor(reconstructed)  # [B*T, F, H', W']
#             target_features = self.feature_extractor(target)  # [B*T, F, H', W']

#         # Compute MSE loss between features
#         loss = self.criterion(reconstructed_features, target_features)

#         return loss



def calculate_video_lpips(video1, video2, lpips_model):
    """
    Calculate LPIPS for each frame of two videos and return the mean LPIPS.
    
    Args:
        video1: Tensor of shape [B, T, C, H, W] (batch, time, channel, height, width).
        video2: Tensor of shape [B, T, C, H, W] (batch, time, channel, height, width).
        lpips_model: Pretrained LPIPS model.

    Returns:
        mean_lpips: Average LPIPS over all frames and batches.
    """
    assert video1.shape == video2.shape, "Input videos must have the same shape"
    
    B, T, C, H, W = video1.shape
    total_lpips = 0.0

    # Iterate through each frame
    for t in range(T):
        frame1 = video1[:, t, :, :, :]  # Shape: [B, C, H, W]
        frame2 = video2[:, t, :, :, :]  # Shape: [B, C, H, W]

        # Calculate LPIPS for the current frame
        frame_lpips = lpips_model(frame1, frame2)  # Shape: [B, 1, 1, 1]
        total_lpips += frame_lpips.mean()

    # Compute mean LPIPS over all frames
    mean_lpips = total_lpips / T
    return mean_lpips

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


import torch

def log_laplace_postprocess(inputs, laplace_eps=0.1):
    """
    Inverse operation of log_laplace_preprocess.

    Args:
        inputs: images of range [0, 1).
        laplace_eps: epsilon as used in log-laplace distribution.

    Returns:
        Postprocessed images for log-laplace modeling.
    """
    img = (inputs - laplace_eps) / (1.0 - 2.0 * laplace_eps)
    # Cap images in value ranges of [0, 1].
    return torch.clamp(img, 0.0, 1.0)


def log_laplace_preprocess(inputs, laplace_eps=0.1):
    """
    Preprocesses input images for log-laplace loss.

    Args:
        inputs: images of range [0, 1).
        laplace_eps: epsilon as used in log-laplace distribution.

    Returns:
        Preprocessed images for log-laplace modeling.
    """
    img = torch.clamp(inputs, 0.0, 1.0)
    # Convert images to [laplace_eps, 1.0 - laplace_eps).
    return img * (1.0 - 2 * laplace_eps) + laplace_eps


def log_laplace_loss(x, mu, log_sigma):
    """
    Computes the log laplace loss.

    Args:
        x: tensor of shape [b, ...]. x is expected to be in range [0, 1). Typically,
          x is the ground truth of a sampled img.
        mu: mean of the log-laplace distribution.
        log_sigma: log of sigma value above. mu and log_sigma are typically
          predictions from a model.

    Returns:
        (loss, neg_logl): a pair of tensors of shape [b, ...]. neg_logl is the
        negative log likelihood of x given p(x|mu, sigma).
    """
    # Cap x to be within [epsilon, 1.0 - epsilon].
    epsilon = 1e-6
    x = torch.clamp(x, epsilon, 1.0 - epsilon)

    logit_x = torch.log(x / (1.0 - x))
    sigma = torch.exp(log_sigma)
    loss = torch.abs(logit_x - mu) / sigma + log_sigma
    # negative log likelihood needs to add in the remaining constant.
    neg_logl = loss + torch.log(2.0 * x * (1.0 - x))
    return loss, neg_logl
