"""
Data preprocessing functions for D2PTR experiments.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys
from typing import Tuple, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_dataset(dataset_name: str, data_dir: str) -> Tuple[DataLoader, DataLoader]:
    """
    Load and preprocess the specified dataset.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'mnist', etc.)
        data_dir: Directory to store/load the dataset
        
    Returns:
        train_loader, test_loader: DataLoaders for training and test data
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    if dataset_name.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # Use 0 for CPU-only environments
    
    try:
        from config.experiment_config import BATCH_SIZE, NUM_WORKERS
    except ImportError:
        print("WARNING: Could not import BATCH_SIZE and NUM_WORKERS from config. Using defaults.")
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True)
    
    return train_loader, test_loader
