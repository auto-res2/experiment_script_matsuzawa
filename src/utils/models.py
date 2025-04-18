"""
Neural network models for DALWGAN experiments
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder network for mapping input data to latent space
    """
    def __init__(self, input_dim=3, latent_dim=2):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class DiffusionPurification(nn.Module):
    """
    Diffusion-based purification for latent codes
    """
    def __init__(self, latent_dim, num_steps=10, method='heun'):
        super(DiffusionPurification, self).__init__()
        self.num_steps = num_steps
        self.method = method
        self.latent_dim = latent_dim
        self.diffusion_rate = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, z):
        purified = z
        for step in range(self.num_steps):
            drift = purified
            if self.method == 'euler':
                purified = purified + self.diffusion_rate * drift
            elif self.method == 'heun':
                pred = purified + self.diffusion_rate * drift
                corrected = 0.5 * (drift + pred)
                purified = purified + self.diffusion_rate * corrected
            else:
                raise ValueError("Unknown integration method")
        return purified

class Generator(nn.Module):
    """
    Generator network for creating images from latent codes
    """
    def __init__(self, latent_dim=100, img_channels=1, img_size=64):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_channels * img_size * img_size),
            nn.Tanh()
        )
        self.img_channels = img_channels
        self.img_size = img_size
    
    def forward(self, z):
        img = self.fc(z)
        return img.view(z.size(0), self.img_channels, self.img_size, self.img_size)
