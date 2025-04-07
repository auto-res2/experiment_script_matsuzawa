"""
Utility functions for ANCD experiments.
"""
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def generate_samples(model, dataloader, n_batches=5, save_folder='./logs'):
    """
    Generate sample images and save grid as a pdf.
    
    Args:
        model: Trained model
        dataloader: DataLoader with input data
        n_batches: Number of batches to generate samples from
        save_folder: Folder to save the samples
        
    Returns:
        samples: Generated samples tensor
    """
    os.makedirs(save_folder, exist_ok=True)
    model.eval()
    all_samples = []
    
    with torch.no_grad():
        iter_count = 0
        for inputs, _ in dataloader:
            inputs = inputs.to(next(model.parameters()).device)
            outputs = model(inputs)
            all_samples.append(outputs.cpu())
            iter_count += 1
            if iter_count >= n_batches:
                break
                
    samples = torch.cat(all_samples, dim=0)
    
    temp_filename = os.path.join(save_folder, 'generated_samples.png')
    save_image(samples, temp_filename, nrow=8, normalize=True)
    
    plt.figure(figsize=(8,8))
    img = plt.imread(temp_filename)
    plt.imshow(img)
    plt.axis('off')
    pdf_filename = os.path.join(save_folder, 'generated_samples.pdf')
    plt.savefig(pdf_filename, bbox_inches='tight')
    plt.close()
    
    return samples

def measure_consistency(model, inputs, noise_std=0.05):
    """
    Measure consistency loss by comparing output on the original inputs and perturbed inputs.
    
    Args:
        model: Model to evaluate
        inputs: Input tensor
        noise_std: Standard deviation of noise to add
        
    Returns:
        mse_loss: Mean squared error between outputs
    """
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        outputs_orig = model(inputs.to(device))
        
    noise = torch.randn_like(inputs) * noise_std
    inputs_perturbed = inputs + noise
    
    with torch.no_grad():
        outputs_perturbed = model(inputs_perturbed.to(device))
        
    mse_loss = torch.nn.MSELoss()(outputs_orig, outputs_perturbed)
    return mse_loss.item()

def save_plot(x, y, xlabel, ylabel, title, filename, figsize=(6, 4), color='blue', marker='o'):
    """
    Create and save a plot as PDF.
    
    Args:
        x: X-axis data
        y: Y-axis data
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
        figsize: Figure size tuple
        color: Line color
        marker: Marker style
    """
    plt.figure(figsize=figsize)
    plt.plot(x, y, color=color, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def save_bar_plot(x_labels, y_values, xlabel, ylabel, title, filename, figsize=(6, 4)):
    """
    Create and save a bar plot as PDF.
    
    Args:
        x_labels: X-axis labels
        y_values: Y-axis values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    plt.bar(x_labels, y_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
