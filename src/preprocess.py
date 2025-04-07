"""
Script for preprocessing data.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import os
from config.experiment_config import BATCH_SIZE, DATA_DIR

def get_cifar10_loader(batch_size=BATCH_SIZE, train=False):
    """
    Get CIFAR10 dataloader.
    
    Args:
        batch_size: Batch size for dataloader
        train: Whether to load training or test data
        
    Returns:
        DataLoader: PyTorch dataloader for CIFAR10
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, 
        train=train, 
        download=True, 
        transform=transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train
    )
    
    return dataloader

def add_trigger(images, intensity=1.0):
    """
    Add a synthetic trigger pattern (a small bright square) onto a batch of images.
    
    Args:
        images: A torch.Tensor with shape (N, C, H, W)
        intensity: The intensity parameter scales the brightness of the trigger
        
    Returns:
        torch.Tensor: Images with added trigger
    """
    triggered = images.clone()
    patch_size = 4  # small patch
    for i in range(triggered.size(0)):
        patch = (torch.ones(3, patch_size, patch_size) * intensity).to(triggered.device)
        triggered[i, :, 0:patch_size, 0:patch_size] = torch.clamp(
            triggered[i, :, 0:patch_size, 0:patch_size] + patch, 0.0, 1.0
        )
    return triggered
