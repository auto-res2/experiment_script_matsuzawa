"""
Data preprocessing module for ANCD experiments.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.experiment_config import BATCH_SIZE, TEST_BATCH_SIZE, RANDOM_SEED

def get_dataloaders(batch_size=BATCH_SIZE, test_mode=False):
    """
    Load and preprocess the CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for data loading
        test_mode: If True, use a small subset of data
        
    Returns:
        train_loader: DataLoader for training data
    """
    torch.manual_seed(RANDOM_SEED)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    if test_mode:
        indices = list(range(min(256, len(train_dataset))))
        train_dataset.data = train_dataset.data[indices]
        train_dataset.targets = [train_dataset.targets[i] for i in indices]
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size if not test_mode else TEST_BATCH_SIZE,
        shuffle=True, 
        num_workers=2
    )
    
    return train_loader
