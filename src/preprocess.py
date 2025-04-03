"""
Data preprocessing module for ANGAS experiments.

This module contains functions for creating synthetic data and loading data for experiments.
"""

import torch
import torch.utils.data as data
import numpy as np


class SyntheticDataset(data.Dataset):
    """
    Generate synthetic data for testing diffusion models.
    
    For demonstration purposes, this creates random noise tensors.
    In a real application, this would be replaced with real image data.
    """
    def __init__(self, size=32, channels=3, num_samples=1000):
        """
        Initialize the synthetic dataset.
        
        Args:
            size: Image size (square images of size x size)
            channels: Number of image channels
            num_samples: Number of samples in the dataset
        """
        self.size = size
        self.channels = channels
        self.num_samples = num_samples
        
        self.data = torch.randn(num_samples, channels, size, size)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def get_data_loaders(batch_size=16, size=32, channels=3, num_samples=1000):
    """
    Create data loaders for training and testing.
    
    Args:
        batch_size: Batch size for the data loaders
        size: Image size
        channels: Number of image channels
        num_samples: Total number of samples
        
    Returns:
        train_loader, test_loader: DataLoaders for training and testing
    """
    full_dataset = SyntheticDataset(size=size, channels=channels, num_samples=num_samples)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = data.random_split(
        full_dataset, [train_size, test_size]
    )
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader


def prepare_initial_noise(batch_size=16, channels=3, size=32, device='cuda'):
    """
    Prepare initial noise for diffusion sampling.
    
    Args:
        batch_size: Number of samples to generate
        channels: Number of image channels
        size: Image size
        device: Device to create tensors on
        
    Returns:
        Tensor of random noise
    """
    return torch.randn(batch_size, channels, size, size, device=device)
