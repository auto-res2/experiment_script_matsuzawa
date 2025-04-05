"""
Preprocessing module for TEDP (Trigger-Eradicating Diffusion Purification).
Implements dataset creation and initial data purification.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class SyntheticDataset(Dataset):
    """
    Generates synthetic 32x32 RGB images.
    With probability poison_ratio, a "poisoned" trigger (a white square patch) is added.
    Labels: 0 for clean, 1 for poisoned.
    """
    def __init__(self, size=1000, poison_ratio=0.05):
        self.size = size
        self.poison_ratio = poison_ratio
        self.data = []
        self.labels = []
        self.generate_data()
    
    def generate_data(self):
        for i in range(self.size):
            image = np.random.rand(3, 32, 32).astype(np.float32)
            if random.random() < self.poison_ratio:
                image[:, 28:32, 28:32] = 1.0
                label = 1
            else:
                label = 0
            self.data.append(image)
            self.labels.append(label)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index]), self.labels[index]
    
    def __len__(self):
        return self.size


class PurificationModule(torch.nn.Module):
    def __init__(self, noise_std_init=0.05, variance_explosion=1.1, step_size=0.1):
        """
        noise_std_init: initial standard deviation for Gaussian noise
        variance_explosion: multiplier applied based on batch index
        step_size: step-size for the Heun integration step
        """
        super(PurificationModule, self).__init__()
        self.noise_std_init = noise_std_init
        self.variance_explosion = variance_explosion
        self.step_size = step_size

    def forward(self, x):
        batch_size = x.size(0)
        noise_factors = torch.arange(batch_size, dtype=x.dtype, device=x.device).view(-1, 1, 1, 1)
        noise_std = self.noise_std_init * (self.variance_explosion ** noise_factors)
        x_noisy = x + torch.randn_like(x) * noise_std

        f_x = self.diffusion_dynamics(x_noisy)
        x_euler = x_noisy - self.step_size * f_x  # Euler prediction
        f_x_euler = self.diffusion_dynamics(x_euler)
        x_heun = x_noisy - (self.step_size/2.0)*(f_x + f_x_euler)
        return x_heun

    def diffusion_dynamics(self, x):
        return F.relu(x)
