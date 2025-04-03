"""
Training module for ANGAS experiments.

This module contains the implementation of the ScoreModel and functions for model training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class ScoreModel(nn.Module):
    """
    ScoreModel is a lightweight convolutional neural network for scoring/diffusion models.
    
    In a real implementation, this would be replaced with a more sophisticated
    score-based diffusion network.
    """
    def __init__(self, in_channels=3, hidden_channels=16):
        super(ScoreModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch of images)
            
        Returns:
            Score estimate for the input
        """
        return torch.tanh(self.net(x))


def train_score_model(model, dataloader, num_epochs=10, lr=1e-4, device='cuda'):
    """
    Train the score model.
    
    Args:
        model: ScoreModel instance
        dataloader: DataLoader containing training data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        Trained model and list of losses per epoch
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x = batch.to(device)
            target = x + torch.randn_like(x) * 0.1  # Dummy target
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
    
    return model, losses
