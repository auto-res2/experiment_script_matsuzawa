"""
Scripts for data preprocessing.

This module handles loading and preprocessing datasets for D-DAME experiments.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloader(dataset_name='CIFAR10', batch_size=128, num_workers=2, train=True):
    """
    Create a dataloader for the specified dataset.
    
    Args:
        dataset_name: Name of the dataset (CIFAR10, CIFAR100, or AFHQ-DOG)
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        train: Whether to use the training set (True) or test set (False)
        
    Returns:
        DataLoader for the specified dataset
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = datasets.CIFAR100(root='./data', train=train, download=True, transform=transform)
    elif dataset_name == 'AFHQ-DOG':
        dataset = datasets.ImageFolder(root='./data/AFHQ-DOG', transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
