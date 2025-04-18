"""
Data preprocessing functions for DALWGAN experiments
"""

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.datasets import make_swiss_roll

def generate_synthetic_data(n_samples=1500, noise=0.1, random_seed=None):
    """
    Generate synthetic Swiss roll dataset for dimensionality experiments
    
    Args:
        n_samples: Number of samples to generate
        noise: Noise level for the dataset
        random_seed: Random seed for reproducibility
        
    Returns:
        data: numpy array of shape (n_samples, 3)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    data, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
    return data.astype(np.float32)

def load_mnist_data(batch_size=64, image_size=64, root='./data'):
    """
    Load MNIST dataset for real-world image evaluation
    
    Args:
        batch_size: Batch size for dataloader
        image_size: Size to resize images to
        root: Data directory
        
    Returns:
        dataloader: PyTorch DataLoader with MNIST dataset
        dataset: MNIST dataset
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, dataset
