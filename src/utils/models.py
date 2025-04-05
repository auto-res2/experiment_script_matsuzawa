"""
Model definitions for D2PTR experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN classifier for CIFAR-10 dataset.
    """
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # [B,16,32,32]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,16,16,16]
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # [B,32,16,16]
            nn.ReLU(),
            nn.MaxPool2d(2)                              # [B,32,8,8]
        )
        self.fc = nn.Linear(32*8*8, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LatentEncoder(nn.Module):
    """
    An encoder that maps images to a latent space.
    """
    def __init__(self, latent_dim: int = 128):
        super(LatentEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # [B, 32, 16,16]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # [B, 64, 8,8]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class DiffusionPurifier(nn.Module):
    """
    A diffusion-based purification module for adversarial examples.
    """
    def __init__(self, num_steps: int = 10, step_size: float = 0.1):
        super(DiffusionPurifier, self).__init__()
        self.num_steps = num_steps
        self.step_size = step_size

    def forward(self, x):
        noisy_x = x.clone()
        for i in range(self.num_steps):
            noise = torch.randn_like(noisy_x) * self.step_size
            noisy_x = noisy_x + noise
        purified = self.heun_reverse(noisy_x)
        return purified

    def heun_reverse(self, x_noisy):
        purified = x_noisy.clone()
        for i in range(self.num_steps):
            f1 = self.diffusion_function(purified)
            provisional = purified - self.step_size * f1
            f2 = self.diffusion_function(provisional)
            purified = purified - self.step_size * 0.5 * (f1 + f2)
        return purified

    def diffusion_function(self, x):
        return torch.zeros_like(x)
