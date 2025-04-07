"""
Utility functions for visualizing results.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from config.experiment_config import FIGURES_DIR

def plot_image_grid(images, titles=None, figsize=(12, 8)):
    """
    Plot a grid of images.
    
    Args:
        images: List of images (numpy arrays or torch tensors)
        titles: List of titles for each image
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    n = len(images)
    fig = plt.figure(figsize=figsize)
    
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        
        if isinstance(images[i], torch.Tensor):
            img = images[i].cpu().detach().numpy()
            if img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        else:
            img = images[i]
            
        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        
        if titles is not None and i < len(titles):
            plt.title(titles[i])
            
    plt.tight_layout()
    return fig

def save_image_comparison(original, triggered, purified, filename):
    """
    Save a comparison of original, triggered, and purified images.
    
    Args:
        original: Original image tensor
        triggered: Triggered image tensor
        purified: Purified image tensor
        filename: Name of the file to save (without extension)
    """
    images = [original, triggered, purified]
    titles = ['Original', 'Triggered', 'Purified']
    
    fig = plot_image_grid(images, titles)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, f"{filename}.pdf")
    fig.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved comparison: {filepath}")
    plt.close(fig)
