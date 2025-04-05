"""
DALWGAN Training Module

This module implements the Diffusion-Assisted Latent Wasserstein GAN (DALWGAN)
with key components:
1. Encoder for adaptive latent embedding
2. Diffusion purification stage in the latent space
3. Generator and Discriminator networks
4. Wasserstein GAN training with integrated diffusion purification
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Encoder(nn.Module):
    """
    Encoder network for mapping data to latent space with adaptive dimensionality
    """
    def __init__(self, input_dim=3, latent_dim=2, hidden_dims=[128, 64]):
        super(Encoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.fc = nn.Sequential(*layers)
        
        self.A = nn.Parameter(torch.ones(latent_dim))
        
    def forward(self, x):
        z = self.fc(x)
        z = z * self.A
        return z

class DiffusionPurification(nn.Module):
    """
    Diffusion-based purification module for latent codes
    """
    def __init__(self, latent_dim, num_steps=10, method='heun'):
        super(DiffusionPurification, self).__init__()
        self.num_steps = num_steps
        self.method = method
        self.latent_dim = latent_dim
        
        self.diffusion_rate = nn.Parameter(torch.tensor(0.1))
        
        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, latent_dim)
        )
        
    def forward(self, z):
        purified = z
        for step in range(self.num_steps):
            drift = self.drift_net(purified)
            
            if self.method == 'euler':
                purified = purified + self.diffusion_rate * drift
            elif self.method == 'heun':
                pred = purified + self.diffusion_rate * drift
                corrected_drift = self.drift_net(pred)
                corrected = 0.5 * (drift + corrected_drift)
                purified = purified + self.diffusion_rate * corrected
            else:
                raise ValueError(f"Unknown integration method: {self.method}")
                
        return purified
        
class Generator(nn.Module):
    """
    Generator network for DALWGAN
    """
    def __init__(self, latent_dim=2, output_dim=3, hidden_dims=[64, 128]):
        super(Generator, self).__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.model(z)
        
class Discriminator(nn.Module):
    """
    Discriminator/Critic network for DALWGAN
    """
    def __init__(self, input_dim=3, hidden_dims=[128, 64]):
        super(Discriminator, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
        
class DALWGAN:
    """
    Diffusion-Assisted Latent Wasserstein GAN implementation
    """
    def __init__(self, 
                 input_dim=3, 
                 latent_dim=2, 
                 diffusion_steps=10,
                 integration_method='heun',
                 lambda_rank=0.01,
                 lambda_diff=0.1,
                 device='cuda'):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.lambda_rank = lambda_rank
        self.lambda_diff = lambda_diff
        
        self.encoder = Encoder(input_dim, latent_dim).to(device)
        self.diffusion = DiffusionPurification(latent_dim, diffusion_steps, integration_method).to(device)
        self.generator = Generator(latent_dim, input_dim).to(device)
        self.discriminator = Discriminator(input_dim).to(device)
        
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=0.0002, betas=(0.5, 0.9))
        self.diffusion_opt = optim.Adam(self.diffusion.parameters(), lr=0.0002, betas=(0.5, 0.9))
        self.generator_opt = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.9))
        self.discriminator_opt = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.9))
        
    def train_step(self, real_data, n_critic=5):
        """
        Execute one training step
        
        Args:
            real_data (torch.Tensor): Batch of real data
            n_critic (int): Number of critic updates per generator update
            
        Returns:
            dict: Dictionary with loss values
        """
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        
        for _ in range(n_critic):
            self.discriminator_opt.zero_grad()
            
            d_real = self.discriminator(real_data)
            
            z = self.encoder(real_data)
            
            z_purified = self.diffusion(z)
            
            fake_data = self.generator(z_purified)
            d_fake = self.discriminator(fake_data.detach())
            
            d_loss = -(torch.mean(d_real) - torch.mean(d_fake))
            
            
            d_loss.backward()
            self.discriminator_opt.step()
        
        self.encoder_opt.zero_grad()
        self.diffusion_opt.zero_grad()
        self.generator_opt.zero_grad()
        
        z = self.encoder(real_data)
        
        z_purified = self.diffusion(z)
        
        fake_data = self.generator(z_purified)
        d_fake = self.discriminator(fake_data)
        
        g_loss = -torch.mean(d_fake)
        
        rank_penalty = torch.sum(torch.abs(self.encoder.A))
        
        diff_loss = torch.mean((z_purified - z) ** 2)
        
        combined_loss = g_loss + self.lambda_rank * rank_penalty + self.lambda_diff * diff_loss
        
        combined_loss.backward()
        
        self.encoder_opt.step()
        self.diffusion_opt.step()
        self.generator_opt.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'rank_penalty': rank_penalty.item(),
            'diff_loss': diff_loss.item()
        }
        
    def save_models(self, path):
        """
        Save all model parameters to a directory
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pt'))
        torch.save(self.diffusion.state_dict(), os.path.join(path, 'diffusion.pt'))
        torch.save(self.generator.state_dict(), os.path.join(path, 'generator.pt'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pt'))
        
    def load_models(self, path):
        """
        Load all model parameters from a directory
        """
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pt')))
        self.diffusion.load_state_dict(torch.load(os.path.join(path, 'diffusion.pt')))
        self.generator.load_state_dict(torch.load(os.path.join(path, 'generator.pt')))
        self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pt')))
