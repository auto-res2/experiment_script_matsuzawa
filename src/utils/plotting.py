"""
Utility functions for creating and saving plots in high-quality PDF format.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchvision import utils as tv_utils
import numpy as np

sns.set_theme(style="whitegrid")

def save_loss_curve(epochs, losses, filename, title, multiple_curves=False, labels=None):
    """
    Creates and saves a loss curve plot in high-quality PDF format.
    
    Args:
        epochs (list): List of epoch numbers.
        losses (list or list of lists): Loss values per epoch. If multiple_curves is True,
                                       this should be a list of lists.
        filename (str): Name of the file to save the plot to.
        title (str): Title of the plot.
        multiple_curves (bool): Whether to plot multiple curves.
        labels (list): List of labels for multiple curves.
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    if multiple_curves:
        for i, loss_values in enumerate(losses):
            label = labels[i] if labels else f"Curve {i+1}"
            sns.lineplot(x=epochs, y=loss_values, marker='o', label=label)
    else:
        sns.lineplot(x=epochs, y=losses, marker='o')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    
    if multiple_curves:
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(f"logs/{filename}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    
def save_image_grid(images, filename, nrow=8):
    """
    Saves a grid of images in high-quality PDF format.
    
    Args:
        images (torch.Tensor): Tensor of images to save.
        filename (str): Name of the file to save the grid to.
        nrow (int): Number of images in each row of the grid.
    """
    if images.min() < 0:
        images = (images + 1) / 2
        
    images = torch.clamp(images, 0, 1)
    
    grid = tv_utils.make_grid(images, nrow=nrow)
    
    grid_np = grid.cpu().numpy().transpose((1, 2, 0))
    
    plt.figure(figsize=(10, 10), dpi=300)
    plt.axis('off')
    plt.imshow(grid_np)
    plt.tight_layout()
    plt.savefig(f"logs/{filename}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    
def save_scatter_plot(x_values, y_values, filename, title, xlabel, ylabel):
    """
    Creates and saves a scatter plot in high-quality PDF format.
    
    Args:
        x_values (list): X-axis values.
        y_values (list): Y-axis values.
        filename (str): Name of the file to save the plot to.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(10, 6), dpi=300)
    sns.scatterplot(x=x_values, y=y_values, s=100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"logs/{filename}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
