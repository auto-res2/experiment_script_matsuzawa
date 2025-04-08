"""
Preprocessing module for the Optimized Characteristic Resampling (OCR) experiment.

This module handles dataset preparation and transformations for the OCR experiments.
"""

import torch
from torchvision import datasets, transforms

def get_dummy_dataset(batch_size=16, image_size=(3, 32, 32), dataset_size=64):
    """
    Creates a dummy dataset for testing the OCR method.
    
    Args:
        batch_size (int): The batch size for the dataloader
        image_size (tuple): Image dimensions (channels, height, width)
        dataset_size (int): Number of samples in the dataset
    
    Returns:
        dataloader (torch.utils.data.DataLoader): DataLoader with the dummy dataset
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FakeData(size=dataset_size, image_size=image_size, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def prepare_data(config):
    """
    Prepares data for the OCR experiment based on configuration.
    
    Args:
        config (dict): Configuration dictionary with data parameters
    
    Returns:
        dataloader (torch.utils.data.DataLoader): DataLoader with the prepared dataset
    """
    return get_dummy_dataset(
        batch_size=config.get("batch_size", 16),
        image_size=config.get("image_size", (3, 32, 32)),
        dataset_size=config.get("dataset_size", 64)
    )
