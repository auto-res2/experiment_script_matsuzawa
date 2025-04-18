"""
Training functions for DALWGAN experiments
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.models import Encoder, Generator, DiffusionPurification

def train_model_dummy(model, optimizer, criterion, input_data, device="cuda"):
    """
    Perform a single training step (for demonstration and testing)
    
    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer
        criterion: Loss function
        input_data: Input data tensor
        device: Device to run training on
        
    Returns:
        loss: Training loss value
    """
    model.train()
    optimizer.zero_grad()
    
    output = model(input_data)
    target = torch.zeros_like(output)
    
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_purifier(purifier, encoder, data_tensor, iterations=10, learning_rate=0.01):
    """
    Train diffusion purification module
    
    Args:
        purifier: DiffusionPurification module
        encoder: Encoder module
        data_tensor: Input data tensor
        iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        
    Returns:
        losses: List of loss values during training
    """
    optimizer = optim.Adam(purifier.parameters(), lr=learning_rate)
    losses = []
    
    for it in range(iterations):
        optimizer.zero_grad()
        
        latent_codes = encoder(data_tensor)
        purified = purifier(latent_codes)
        
        loss = ((purified - latent_codes)**2).mean()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses
