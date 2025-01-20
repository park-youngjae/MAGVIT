import os
import pdb
import pytz
import lpips
import random
import logging
import datetime
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Lambda
from torchvision.io import read_video
from tqdm import tqdm

# Import custom modules
from dataload import MovingMNIST, KineticsDataset
# from model import VQVAE, Discriminator
from model import VQModel as VQVAE
from model import Discriminator
from torch.utils.tensorboard import SummaryWriter
from loss import gan_loss, compute_gradient_penalty, lecam_regularization, log_laplace_loss, calculate_video_lpips
from utils import lr_lambda, collate_fn_ignore_none, save_checkpoint, combined_scheduler, EMA
from eval import calculate_fvd

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Suppress warnings
warnings.filterwarnings('ignore')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train_vqvae(rank,world_size,dataset_name='k600',batchsize=2):
    setup(rank, world_size)
    # Generate a common base name with timestamp
    timezone = pytz.timezone('Etc/GMT+9')
    timestamp = datetime.datetime.now(timezone).strftime("%Y%m%d-%H%M%S")
    common_filename = f"training_{timestamp}_{dataset_name}"

    # Setup TensorBoard and logging with the common filename
    writer = SummaryWriter(log_dir=f'./logs/{common_filename}')
    logging.basicConfig(filename=f'./logs/{common_filename}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = torch.device("cuda", rank)  

    # Settings
    num_epochs_stage1 = 45

    # Dataset Preparation
    if dataset_name=='k600':
        transform = transforms.Compose([ # for Kinetics-600 Train Set
            Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
            transforms.Resize((128, 128)),           # Resize to a fixed size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
        ])


        train_dataset = KineticsDataset('../Data/k600/train', transform=transform)  # Custom Dataset
        # train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, num_workers=0, pin_memory=True)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, sampler=train_sampler, num_workers=0, pin_memory=True)
        # train_loader = DataLoader(train_dataset, batch_size=batchsize, collate_fn=collate_fn_ignore_none, shuffle=False, sampler=train_sampler)

    elif dataset_name=='MMNIST':
        transform = transforms.Compose([ # for Moving MNIST Dataset
            transforms.ToTensor(),
            transforms.Resize((128, 128)),  
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = MovingMNIST('../Data/mmnist', transform=transform)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, sampler=train_sampler)

    # Define models
    model_vqvae = VQVAE().to(device)
    model_discriminator = Discriminator().to(device)
    model_vqvae = DDP(model_vqvae, device_ids=[rank],find_unused_parameters=True)
    ema = EMA(model_vqvae, decay=0.999)
    model_discriminator = DDP(model_discriminator, device_ids=[rank])

    # Optimizers
    optimizer_vqvae = optim.Adam(model_vqvae.parameters(), lr=1e-4, betas=(0, 0.99))  # Peak LR = 1e-4, β1=0, β2=0.99
    optimizer_discriminator = optim.Adam(model_discriminator.parameters(), lr=1e-4, betas=(0, 0.99))

    # Learning rate scheduler
    scheduler_vqvae = torch.optim.lr_scheduler.LambdaLR(
        optimizer_vqvae,
        lr_lambda=lambda epoch: combined_scheduler(epoch, warmup_epochs=5, total_epochs=num_epochs_stage1, initial_lr=1e-4)
    )
    # scheduler_vqvae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vqvae, T_max=len(train_loader))
    scheduler_discriminator = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_discriminator, T_max=len(train_loader))

    # Loss functions
    # vgg_feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True).features.to(device).eval()
    # perceptual_loss = PerceptualLoss(vgg_feature_extractor)
    # perceptual_loss = lpips.LPIPS(net='vgg').to(device)
    gan_loss = nn.BCEWithLogitsLoss()  # GAN loss
    reconstruction_loss = nn.MSELoss()  # Reconstruction loss
    lpips_model = lpips.LPIPS(net='vgg').to(device)

    lambda_perceptual = 0.1  # Perceptual loss weight
    lambda_gan = 0.1  # Generator adversarial loss weight
    lambda_lecam = 0.01  # LeCam Regularization weight
    gradient_penalty_cost = 10  # Discriminator gradient penalty



    # Training loop
    for epoch in range(num_epochs_stage1):
        model_vqvae.train()
        model_discriminator.train()
        total_loss_discriminator = 0
        total_loss_generator = 0

        # Wrap train_loader with tqdm for progress display
        train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs_stage1}")

        for i, videos in train_loader_tqdm:
            videos = videos.to(device)

            # Train Generator
            optimizer_vqvae.zero_grad()  
            reconstructed_videos, quant_loss  = model_vqvae(videos)
            reconstructed_videos = reconstructed_videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

            # Perceptual Loss: Calculate loss between features of real and reconstructed videos
            perceptual_loss_value = calculate_video_lpips(reconstructed_videos, videos, lpips_model)

            # Generator's Adversarial Loss: Discriminator should classify the reconstructed videos as real
            fake_logits = model_discriminator(reconstructed_videos)
            target_real_labels = torch.ones_like(fake_logits)
            gan_loss_value = gan_loss(fake_logits, target_real_labels)

            # Log Laplace Loss: Measure distribution alignment
            mu = reconstructed_videos  # Assuming model predicts mu
            log_sigma = torch.zeros_like(reconstructed_videos)  # Assuming constant sigma for simplicity
            log_laplace_loss_value, neg_log_likelihood = log_laplace_loss(videos, mu, log_sigma)

            # Total Generator Loss: Combine losses with respective weights
            # reconstruction_loss + g_adversarial_loss + perceptual_loss + quantizer_loss + logit_laplace_loss
            reconstruction_loss_value = reconstruction_loss(videos, reconstructed_videos)                                   # Calculate reconstruction loss (l2)

            total_generator_loss = reconstruction_loss_value + lambda_gan * gan_loss_value + log_laplace_loss_value.mean()+ lambda_perceptual * perceptual_loss_value 
            total_loss_generator += total_generator_loss.item()  

            # Backpropagation
            total_generator_loss.backward()
            optimizer_vqvae.step()

            if i == 0:
                # score_fvd = calculate_fvd(videos,reconstructed_videos,device=device)
                # print(f"FVD: {score_fvd} at Epoch {epoch+1}")
                first_video_reconstructed = reconstructed_videos[0, 0] 
                first_video_reconstructed = first_video_reconstructed.unsqueeze(0) # [1, C, H, W]
                writer.add_images('Reconstructed_Videos', first_video_reconstructed, epoch * len(train_loader) + i)

            # Train Discriminator
            optimizer_discriminator.zero_grad()
            real_logits = model_discriminator(videos)
            fake_logits = model_discriminator(reconstructed_videos.detach())

            # Discriminator Loss for Real and Fake Samples
            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)
            real_loss = gan_loss(real_logits, real_labels)
            fake_loss = gan_loss(fake_logits, fake_labels)

            # Calculate LeCam Regularization
            lecam_reg = lecam_regularization(model_discriminator, videos, reconstructed_videos.detach())
            lecam_weight = 0.01  # Define the regularization weight

            # Total Discriminator Loss
            total_discriminator_loss = (real_loss + fake_loss) / 2

            # d_adversarial_loss + grad_penalty + lecam_loss
            lecam_loss_value = lecam_regularization(model_discriminator, videos, reconstructed_videos.detach())             # Calculate lecam regularization
            grad_penalty_value = compute_gradient_penalty(                                                                  # Calculate gradient penalty
                model_discriminator, real_data=videos, fake_data=reconstructed_videos.detach(), device=device
            )
            total_loss_discriminator +=  total_discriminator_loss.item() + grad_penalty_value.item() +lecam_loss_value.item()

            # Backpropagation for Discriminator
            total_discriminator_loss.backward()
            optimizer_discriminator.step()

            # Update the Exponential Moving Average (EMA) for Generator parameters
            ema.update()

            # Log losses to TensorBoard
            writer.add_scalar('Loss/Generator_Total', total_generator_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss/Discriminator_Total', total_discriminator_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss/Generator_Perceptual', perceptual_loss_value.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss/Generator_GAN', gan_loss_value.item(), epoch * len(train_loader) + i)

            # Print and log losses for each batch
            # print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Generator Loss: {total_generator_loss.item()}, Discriminator Loss: {total_discriminator_loss.item()}")
            logging.info(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Generator Loss: {total_generator_loss.item()}, Discriminator Loss: {total_discriminator_loss.item()}")
            torch.cuda.empty_cache()

        # Average the losses
        avg_loss_discriminator = total_loss_discriminator / len(train_loader)
        avg_loss_generator = total_loss_generator / len(train_loader)

        # Print and log average losses for the epoch
        print(f"Epoch {epoch+1}: Average Generator Loss: {avg_loss_generator}, Average Discriminator Loss: {avg_loss_discriminator}")
        logging.info(f"Epoch {epoch+1}: Average Generator Loss: {avg_loss_generator}, Average Discriminator Loss: {avg_loss_discriminator}\n")

        # Log average losses to TensorBoard
        writer.add_scalar('Loss_AVG/Generator', avg_loss_generator, epoch)
        writer.add_scalar('Loss_AVG/Discriminator', avg_loss_discriminator, epoch)

        # Save checkpoint at the end of each epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_VQVAE': model_vqvae.state_dict(),
            'state_dict_Discriminator': model_discriminator.state_dict(),
            'optimizer_vqvae': optimizer_vqvae.state_dict(),
            'optimizer_discriminator': optimizer_discriminator.state_dict(),
        }, filename=f"./checkpoints/checkpoint_{dataset_name}_{timestamp}_epoch_{epoch+1}.pth.tar")


if __name__ == "__main__":
    world_size = 8
    torch.multiprocessing.spawn(train_vqvae, args=(world_size,), nprocs=world_size, join=True)
