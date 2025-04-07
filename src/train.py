"""
Training module for RobustPurify-Backdoor Diffusion experiment.

This module contains:
- DiffusionModel: A simple convolutional neural network
- train_model: Function to train the model with single-loss or dual-loss regime
- save_sample_images: Function to save sample image pairs for visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from preprocess import simulate_purification


class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return self.conv(x)


def train_model(model, dataloader, dual_loss=False, lambda_consistency=0.5, num_epochs=10):
    """
    Train the diffusion model using single-loss (reconstruction) or dual-loss regime.
    Prints epoch losses to standard output.
    
    Args:
        model: The DiffusionModel to train
        dataloader: DataLoader containing training data
        dual_loss: Whether to use dual-loss training
        lambda_consistency: Weight for consistency loss
        num_epochs: Number of training epochs
    
    Returns:
        List of average losses per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    reconstruction_loss_fn = nn.MSELoss()
    consistency_loss_fn = nn.MSELoss()
    
    loss_history = []
    
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for raw_images, purified_images in dataloader:
            optimizer.zero_grad()
            raw_images = raw_images.unsqueeze(1).float()
            purified_images = purified_images.unsqueeze(1).float()
            
            output_raw = model(raw_images)
            output_purified = model(purified_images)
            loss_rec = reconstruction_loss_fn(output_raw, raw_images)
            if dual_loss:
                loss_consistency = consistency_loss_fn(output_raw, output_purified)
                loss = loss_rec + lambda_consistency * loss_consistency
            else:
                loss = loss_rec
            
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    return loss_history


def save_sample_images(raw, purified, filename="sample_images_pair1.pdf"):
    """
    Plot a pair of images (raw poisoning image and its purified version),
    then save the figure as a .pdf file.
    
    Args:
        raw: Raw image
        purified: Purified version of the image
        filename: Output filename (must be .pdf)
    """
    plt.figure(figsize=(8, 4))
    plt.subplot(1,2,1)
    plt.imshow(raw, cmap='gray')
    plt.title("Raw Poisoned Image")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(purified, cmap='gray')
    plt.title("Purified Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close()
    print(f"Saved sample images as {filename}")
