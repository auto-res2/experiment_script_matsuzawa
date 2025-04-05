"""
Script for preprocessing data.

This module handles the loading and preprocessing of the CIFAR-10 dataset
for use in the ASID-M experiments.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloader(batch_size=128, train=True, download=True):
    """
    Creates and returns a DataLoader for the CIFAR-10 dataset.
    
    Args:
        batch_size (int): The batch size to use for the DataLoader.
        train (bool): Whether to load the training set (True) or test set (False).
        download (bool): Whether to download the dataset if not already downloaded.
        
    Returns:
        DataLoader: A PyTorch DataLoader containing the CIFAR-10 dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root='./data', 
        train=train, 
        download=download, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train
    )
    
    return dataloader

if __name__ == "__main__":
    train_loader = get_dataloader(batch_size=128, train=True)
    test_loader = get_dataloader(batch_size=128, train=False)
    
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"Sample batch shape: {sample_batch.shape}")
    print(f"Sample label shape: {sample_labels.shape}")
