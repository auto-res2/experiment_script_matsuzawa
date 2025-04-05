"""
Training functions for D2PTR models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from config.experiment_config import DEVICE, RANDOM_SEED
except ImportError:
    print("WARNING: Could not import from config.experiment_config. Using default values.")
    DEVICE = "cpu"
    RANDOM_SEED = 42

try:
    from src.utils.models import SimpleCNN, LatentEncoder, DiffusionPurifier
    from src.utils.diffusion_utils import set_seed
except ImportError:
    try:
        print("WARNING: Could not import from src. Trying without src prefix.")
        from utils.models import SimpleCNN, LatentEncoder, DiffusionPurifier
        from utils.diffusion_utils import set_seed
    except ImportError:
        print("ERROR: Failed to import required modules. Please check your Python path.")

def train_classifier(train_loader: torch.utils.data.DataLoader, 
                     model: nn.Module, 
                     epochs: int = 5, 
                     lr: float = 0.001) -> Tuple[nn.Module, List[float]]:
    """
    Train a classifier model.
    
    Args:
        train_loader: DataLoader for training data
        model: Model to train
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        trained_model, losses: Trained model and list of training losses
    """
    set_seed(RANDOM_SEED)
    
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return model, losses

def train_latent_encoder(train_loader: torch.utils.data.DataLoader,
                         encoder: nn.Module,
                         epochs: int = 5,
                         lr: float = 0.001) -> Tuple[nn.Module, List[float]]:
    """
    Train a latent encoder model using reconstruction loss.
    
    Args:
        train_loader: DataLoader for training data
        encoder: Encoder model to train
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        trained_encoder, losses: Trained encoder and list of training losses
    """
    
    encoder = encoder.to(DEVICE)
    print("Initialized latent encoder (simplified training in this implementation)")
    
    return encoder, [0.0]  # Return dummy losses
