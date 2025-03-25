# Scripts for data preprocessing.
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.dataset import get_fake_dataset, get_dataloader

def preprocess_data(config):
    """
    Preprocess data for training and evaluation.
    
    Args:
        config: Configuration dictionary containing preprocessing parameters.
    
    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """
    print("Preprocessing data...")
    
    # In a real implementation, load actual datasets
    # For demonstration, we use fake datasets
    train_dataset = get_fake_dataset(
        num_samples=config['train_samples'],
        image_size=config['image_size']
    )
    
    val_dataset = get_fake_dataset(
        num_samples=config['val_samples'],
        image_size=config['image_size']
    )
    
    # Create data loaders
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    val_loader = get_dataloader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False
    )
    
    return train_loader, val_loader
