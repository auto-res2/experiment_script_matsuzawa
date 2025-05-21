#!/usr/bin/env python3
"""
Data preprocessing utilities for Iso-LWGAN experiments.
"""
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

def generate_synthetic_data(n_samples=1000):
    """
    Generate a synthetic dataset with a mixture of two 2D Gaussians.
    """
    mean1 = np.array([2, 2])
    mean2 = np.array([-2, -2])
    cov = np.array([[0.5, 0], [0, 0.5]])
    n1 = n_samples // 2
    n2 = n_samples - n1
    data1 = np.random.multivariate_normal(mean1, cov, n1)
    data2 = np.random.multivariate_normal(mean2, cov, n2)
    data = np.vstack([data1, data2])
    return torch.tensor(data, dtype=torch.float32)

def generate_multimodal_data(n_samples=1500):
    """
    Generate multimodal 2D dataset with three clusters.
    """
    centers = [np.array([0, 0]), np.array([3, 3]), np.array([-3, 3])]
    data_list = []
    for center in centers:
        n = n_samples // len(centers)
        data_list.append(np.random.randn(n, 2) * 0.5 + center)
    data = np.vstack(data_list)
    return torch.tensor(data, dtype=torch.float32)

def load_mnist(batch_size=128):
    """
    Load and return the MNIST training loader.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return loader

def prepare_dataset(dataset_type, params):
    """
    Prepare dataset based on experiment type.
    
    Args:
        dataset_type: String indicating which dataset to prepare ('synthetic', 'multimodal', or 'mnist')
        params: Dictionary of parameters
        
    Returns:
        DataLoader or data tensor depending on the experiment type
    """
    if dataset_type == 'synthetic':
        data = generate_synthetic_data(n_samples=params.get('n_samples', 2000))
        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=params.get('batch_size', 128), shuffle=True)
    
    elif dataset_type == 'multimodal':
        data = generate_multimodal_data(n_samples=params.get('n_samples', 1500))
        dataset = TensorDataset(data)
        return DataLoader(dataset, batch_size=params.get('batch_size', 128), shuffle=True)
    
    elif dataset_type == 'mnist':
        return load_mnist(batch_size=params.get('batch_size', 128))
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
