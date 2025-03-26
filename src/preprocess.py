"""
Preprocessing module for Graph-GaussianAssembler experiments.

This module handles data loading and preprocessing for the experiments.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TextPromptDataset(Dataset):
    """Dataset class for text prompts used in the experiments."""
    
    def __init__(self, prompts_list):
        """
        Initialize the dataset with a list of text prompts.
        
        Args:
            prompts_list (list): List of text prompts
        """
        self.prompts = prompts_list
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]


def create_datasets():
    """
    Create datasets for different experiments.
    
    Returns:
        dict: Dictionary containing datasets for each experiment
    """
    exp1_prompts = ["a futuristic city", "a medieval castle", "a natural landscape"]
    exp2_prompts = ["complex mechanical part", "intricate sculpture"]
    exp3_prompts = ["simple bowl", "busy marketplace", "dense forest scene"]
    
    exp1_dataset = TextPromptDataset(exp1_prompts)
    exp2_dataset = TextPromptDataset(exp2_prompts)
    exp3_dataset = TextPromptDataset(exp3_prompts)
    
    return {
        "experiment1": exp1_dataset,
        "experiment2": exp2_dataset,
        "experiment3": exp3_dataset
    }


def create_dataloaders(datasets, batch_size=1):
    """
    Create dataloaders for the datasets.
    
    Args:
        datasets (dict): Dictionary of datasets
        batch_size (int): Batch size for dataloaders
        
    Returns:
        dict: Dictionary containing dataloaders for each dataset
    """
    dataloaders = {}
    for name, dataset in datasets.items():
        dataloaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloaders


def get_device():
    """
    Get the device to use for computation.
    
    Returns:
        torch.device: Device to use (cuda or cpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    
    return device


def ensure_directories_exist():
    """Ensure that all necessary directories exist."""
    dirs = ["logs", "models", "data", "config"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
