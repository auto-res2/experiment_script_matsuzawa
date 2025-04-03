"""
Script for preprocessing data for Joint-Guided Bayesian Flow Networks (JG-BFN) experiments.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_moons

def load_mnist_data(root='./data', batch_size=64, subset_size=None):
    """
    Load MNIST dataset.
    
    Args:
        root: Root directory for dataset storage
        batch_size: Batch size for data loading
        subset_size: Optional size to limit dataset (for quick tests)
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    
    if subset_size is not None:
        indices = list(range(subset_size))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def generate_synthetic_data(n_samples=1024, noise=0.1, batch_size=64):
    """
    Generate synthetic 2D data using make_moons for experiment 2.
    
    Args:
        n_samples: Number of samples to generate
        noise: Noise level for the dataset
        batch_size: Batch size for data loading
        
    Returns:
        dataloader: DataLoader for the synthetic data
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    X_padded = torch.zeros(n_samples, 4, dtype=torch.float32)
    X_padded[:, :2] = X_tensor  # Copy the original 2D data to the first 2 positions
    
    X_img = X_padded.reshape(n_samples, 1, 2, 2)
    
    dataset = TensorDataset(X_img, torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader
