"""
DALWGAN Data Preprocessing Module

This module includes functions for:
1. Generating synthetic data (Swiss roll, S-curve)
2. Loading and preprocessing real-world datasets (MNIST, etc.)
3. Data utilities for batch processing and normalization
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets import make_swiss_roll

def generate_synthetic_data(dataset='swiss_roll', n_samples=1500, noise=0.1):
    """
    Generate synthetic data for manifold learning experiments
    
    Args:
        dataset (str): Type of dataset ('swiss_roll' or 's_curve')
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add to the dataset
        
    Returns:
        data (numpy.ndarray): Generated data with shape (n_samples, 3)
    """
    if dataset == 'swiss_roll':
        data, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        return data.astype(np.float32)
    else:
        raise ValueError(f"Dataset {dataset} not implemented")

def load_mnist(batch_size=64, img_size=64):
    """
    Load and preprocess MNIST dataset
    
    Args:
        batch_size (int): Batch size for dataloader
        img_size (int): Target image size
        
    Returns:
        dataloader (torch.utils.data.DataLoader): DataLoader for MNIST dataset
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
