"""
Data preprocessing for the MEAB-DG experiments.
This module handles loading, transforming, and preparing data for the experiments.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.experiment_config import (
    BATCH_SIZE, MAX_TEXT_LENGTH, IMAGE_SIZE, NUM_WORKERS
)

def get_image_transforms():
    """Get image transformations for the MEAB-DG experiments."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class DummyMultimodalDataset(Dataset):
    """
    Dummy dataset for multimodal inputs (text + image).
    For demo purposes - in real experiments, replace with actual dataset.
    """
    def __init__(self, size=100, tokenizer=None, image_transform=None):
        self.size = size
        self.tokenizer = tokenizer
        self.transform = image_transform or get_image_transforms()
        
        self.texts = [f"Sample text {i} for multimodal experiment." for i in range(size)]
        self.labels = torch.randint(0, 10, (size,))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        image = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)
        label = self.labels[idx]
        
        if self.tokenizer:
            encoded_text = self.tokenizer(
                text, 
                padding="max_length",
                truncation=True,
                max_length=MAX_TEXT_LENGTH,
                return_tensors="pt"
            )
            encoded_text = {k: v.squeeze(0) for k, v in encoded_text.items()}
            return encoded_text, image, label
        
        return text, image, label

class DummyLongTextDataset(Dataset):
    """
    Dummy dataset for long text inputs.
    For demo purposes - in real experiments, replace with actual dataset.
    """
    def __init__(self, size=50):
        self.size = size
        self.texts = []
        for i in range(size):
            paragraphs = [
                f"Paragraph {j} of document {i}. This contains some sample text for testing " 
                f"the dynamic context splitting module. " * (j % 3 + 1)
                for j in range(5)
            ]
            self.texts.append("\n\n".join(paragraphs))
        
        self.targets = torch.randn(size)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.texts[idx], self.targets[idx]

def get_dataloaders(tokenizer):
    """
    Create and return dataloaders for the experiments.
    
    Args:
        tokenizer: Tokenizer for text encoding
        
    Returns:
        Dictionary of dataloaders for different experiments
    """
    multimodal_dataset = DummyMultimodalDataset(size=100, tokenizer=tokenizer)
    multimodal_loader = DataLoader(
        multimodal_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    
    longtext_dataset = DummyLongTextDataset(size=50)
    longtext_loader = DataLoader(
        longtext_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    
    return {
        "multimodal": multimodal_loader,
        "longtext": longtext_loader
    }


def prepare_data():
    """
    Prepare data for all experiments.
    This is the main function to be called from other scripts.
    
    Returns:
        Dictionary with data info and paths
    """
    os.makedirs("data", exist_ok=True)
    
    print("Preparing data for MEAB-DG experiments...")
    print("Using dummy datasets for demonstration. For real experiments, replace with actual datasets.")
    
    return {
        "data_path": "data/",
        "multimodal_size": 100,
        "longtext_size": 50
    }


if __name__ == "__main__":
    data_info = prepare_data()
    print(f"Data preparation completed. Info: {data_info}")
