import os
import pdb
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
from model import VQVAE, Discriminator
from torch.utils.tensorboard import SummaryWriter
from loss import PerceptualLoss, gan_loss, compute_gradient_penalty
from utils import save_checkpoint, EMA
from eval import calculate_fvd

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Suppress warnings
warnings.filterwarnings('ignore')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '65535'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train(rank,world_size):
    setup(rank, world_size)
    # Generate a common base name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    common_filename = f"training_{timestamp}"

    # Setup TensorBoard and logging with the common filename
    writer = SummaryWriter(log_dir=f'./logs/{common_filename}')
    logging.basicConfig(filename=f'./logs/{common_filename}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    device = torch.device("cuda", rank)  

    # Dataset Preparation
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    transform = transforms.Compose([ # for Kinetics-600 Train Set
        Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
        transforms.Resize((128, 128)),           # Resize to a fixed size
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

    # transform = transforms.Compose([ # for Moving MNIST Dataset
    #     # transforms.ToPILImage(),  
    #     transforms.ToTensor(),
    #     transforms.Resize((128, 128)),  
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    train_dataset = KineticsDataset('../k600/train', transform=transform)  # Custom Dataset
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=train_sampler)
    # train_dataset = MovingMNIST('./data', transform=transform)
    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, sampler=train_sampler)

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
    scheduler_vqvae = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vqvae, T_max=len(train_loader))
    scheduler_discriminator = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_discriminator, T_max=len(train_loader))

    # Loss functions
    vgg_feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True).features.to(device).eval()
    perceptual_loss = PerceptualLoss(vgg_feature_extractor)
    gan_loss = nn.BCEWithLogitsLoss()  # GAN loss
    criterion = nn.MSELoss()  # Reconstruction loss
    lambda_perceptual = 0.1  # Perceptual loss weight
    lambda_gan = 0.1  # Generator adversarial loss weight
    lambda_lecam = 0.01  # LeCam Regularization weight
    gradient_penalty_cost = 10  # Discriminator gradient penalty

    # Settings
    num_epochs_stage1 = 45

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
            z, z_q, reconstructed_videos = model_vqvae(videos)
            reconstructed_videos = reconstructed_videos.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

            # Perceptual Loss: Calculate loss between features of real and reconstructed videos
            perceptual_loss_value = perceptual_loss(reconstructed_videos, videos)

            # Generator's Adversarial Loss: Discriminator should classify the reconstructed videos as real
            fake_logits = model_discriminator(reconstructed_videos)
            target_real_labels = torch.ones_like(fake_logits)
            gan_loss_value = gan_loss(fake_logits, target_real_labels)

            # Total Generator Loss: Combine losses with respective weights
            total_generator_loss = lambda_perceptual * perceptual_loss_value + lambda_gan * gan_loss_value

            # Backpropagation
            total_generator_loss.backward()
            optimizer_vqvae.step()

            if i == 0:
                score_fvd = calculate_fvd(videos,reconstructed_videos,device=device)
                print(f"FVD: {score_fvd} at Epoch {epoch+1}")
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

            # Total Discriminator Loss
            total_discriminator_loss = (real_loss + fake_loss) / 2

            # Backpropagation for Discriminator
            total_discriminator_loss.backward()
            optimizer_discriminator.step()

            # Update the Exponential Moving Average (EMA) for Generator parameters
            ema.update()

            # Log losses to TensorBoard
            writer.add_scalar('Loss/Generator', total_generator_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss/Discriminator', total_discriminator_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss/Perceptual', perceptual_loss_value.item(), epoch * len(train_loader) + i)
            writer.add_scalar('Loss/GAN', gan_loss_value.item(), epoch * len(train_loader) + i)

            # Print and log losses for each batch
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Generator Loss: {total_generator_loss.item()}, Discriminator Loss: {total_discriminator_loss.item()}")
            logging.info(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Generator Loss: {total_generator_loss.item()}, Discriminator Loss: {total_discriminator_loss.item()}")


        # Average the losses
        avg_loss_discriminator = total_loss_discriminator / len(train_loader)
        avg_loss_generator = total_loss_generator / len(train_loader)

        # Print and log average losses for the epoch
        print(f"Epoch {epoch+1}: Average Generator Loss: {avg_loss_generator}, Average Discriminator Loss: {avg_loss_discriminator}")
        logging.info(f"Epoch {epoch+1}: Average Generator Loss: {avg_loss_generator}, Average Discriminator Loss: {avg_loss_discriminator}")

        # Save checkpoint at the end of each epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_VQVAE': model_vqvae.state_dict(),
            'state_dict_Discriminator': model_discriminator.state_dict(),
            'optimizer_vqvae': optimizer_vqvae.state_dict(),
            'optimizer_discriminator': optimizer_discriminator.state_dict(),
        }, filename=f"./checkpoints/checkpoint_{common_filename}_epoch_{epoch+1}.pth.tar")


if __name__ == "__main__":
    world_size = 1
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
