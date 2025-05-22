#!/usr/bin/env python3
"""
Data preprocessing module for the Adaptive Characteristic Simulation (ACS) experiments.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def ensure_directories():
    """Ensure that the necessary directories exist."""
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

def generate_synthetic_data(size=32, channels=3, batch_size=1):
    """
    Generate synthetic image data for testing.
    
    Args:
        size: Height and width of the image
        channels: Number of color channels
        batch_size: Number of images to generate
        
    Returns:
        Tensor of shape [batch_size, channels, height, width]
    """
    return torch.randn(batch_size, channels, size, size)

def save_plot(plt_figure, filename, directory="./logs"):
    """
    Save a matplotlib figure as a high-quality PDF.
    
    Args:
        plt_figure: Matplotlib figure to save
        filename: Name of the file (without extension)
        directory: Directory to save the file in
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{filename}.pdf")
    plt_figure.savefig(filepath, bbox_inches="tight", format="pdf", dpi=300)
    print(f"Saved plot as: {filepath}")
    plt.close(plt_figure)
