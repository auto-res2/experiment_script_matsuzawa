"""
Script for training models for UPR Defense.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
from config.experiment_config import DEVICE, ITERATIONS, LEARNING_RATE

class DiffusionPurifier(nn.Module):
    """
    A diffusion purifier model which simulates the diffusion process.
    In a real-world scenario, this could be replaced by a learnable denoising network.
    """
    def __init__(self):
        super(DiffusionPurifier, self).__init__()
        self.layer = nn.Identity()

    def forward(self, x, noise_level):
        """
        Forward pass of the diffusion purifier.
        
        Args:
            x: Input image tensor
            noise_level: Level of noise to add
            
        Returns:
            torch.Tensor: Denoised image
        """
        noise = torch.randn_like(x) * noise_level
        x_noisy = x + noise
        x_denoised = self.layer(x_noisy)
        return x_denoised

def dual_consistency_loss(ref_image, purified_image, noise):
    """
    Compute the dual consistency loss as the sum of the image-space MSE loss 
    and an estimated noise MSE loss.
    
    Args:
        ref_image: Reference image tensor
        purified_image: Purified image tensor
        noise: Noise tensor
        
    Returns:
        torch.Tensor: Combined loss value
    """
    mse_loss = nn.MSELoss()
    image_loss = mse_loss(purified_image, ref_image)
    noise_est = purified_image - ref_image
    noise_loss = mse_loss(noise_est, noise)
    return image_loss + noise_loss

def terp_purification(image, trigger_intensity):
    """
    A baseline purification function (e.g., TERD) using a simple diffusion process.
    
    Args:
        image: Input image tensor
        trigger_intensity: Intensity of the trigger
        
    Returns:
        torch.Tensor: Purified image
    """
    purifier = DiffusionPurifier().to(image.device)
    noise_level = 0.1 * trigger_intensity
    return purifier(image, noise_level)

def upr_purification(image, trigger_intensity, iterations=ITERATIONS):
    """
    UPR Defense using iterative reverse diffusion with dual consistency loss.
    
    Args:
        image: Input image tensor
        trigger_intensity: Intensity of the trigger
        iterations: Number of iterations for purification
        
    Returns:
        torch.Tensor: Purified image
    """
    purifier = DiffusionPurifier().to(image.device)
    purified = image.clone().requires_grad_(True)
    latent_noise = torch.randn_like(image) * (0.1 * trigger_intensity)
    lr = LEARNING_RATE
    
    for it in range(iterations):
        purified_temp = purifier(purified, latent_noise.std())
        loss = dual_consistency_loss(image, purified_temp, latent_noise)
        purified_grad = torch.autograd.grad(loss, purified, retain_graph=True)[0]
        with torch.no_grad():
            purified -= lr * purified_grad
        purified.requires_grad_(True)
        
    return purified.detach()

def terp_multiplesampler_purification(image, trigger_intensity, max_steps=50, loss_threshold=1e-3):
    """
    Purification function using a standard multi-step sampler (TERD like).
    
    Args:
        image: Input image tensor
        trigger_intensity: Intensity of the trigger
        max_steps: Maximum number of steps
        loss_threshold: Threshold for stopping
        
    Returns:
        tuple: (Purified image, Number of steps)
    """
    purifier = DiffusionPurifier().to(image.device)
    purified = image.clone().detach().requires_grad_(False)
    step = 0
    loss_val = float('inf')
    
    while step < max_steps and loss_val > loss_threshold:
        noise_level = 0.1 * trigger_intensity
        purified = purifier(purified, noise_level)
        loss = dual_consistency_loss(image, purified, torch.zeros_like(image))
        loss_val = loss.item()
        step += 1
        
    return purified, step
