"""
Script for evaluating the UPR Defense model.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
from skimage.metrics import structural_similarity as ssim
from config.experiment_config import DEVICE, MAX_STEPS, LOSS_THRESHOLD, MIX_COEFFICIENTS, NOISE_SCHEDULES, FIGURES_DIR

def compute_psnr_ssim(img1, img2):
    """
    Compute PSNR and SSIM between two images.
    
    Args:
        img1, img2: torch.Tensor of shape (C, H, W) in [0,1]
        
    Returns:
        tuple: (psnr, ssim_value) computed over the image
    """
    img1_np = img1.cpu().detach().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img2_np = img2.cpu().detach().numpy().transpose(1, 2, 0)  # CHW -> HWC
    
    mse = np.mean((img1_np - img2_np) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))
    
    ssim_value = ssim(img1_np, img2_np, multichannel=True, data_range=1.0, win_size=3)
    
    return psnr, ssim_value

def upr_heun_purification(image, trigger_intensity, max_steps=MAX_STEPS, loss_threshold=LOSS_THRESHOLD):
    """
    UPR purification employing a Heun's method based sampler.
    
    Args:
        image: Input image tensor
        trigger_intensity: Intensity of the trigger
        max_steps: Maximum number of steps
        loss_threshold: Threshold for stopping
        
    Returns:
        tuple: (Purified image, Number of steps)
    """
    from src.train import DiffusionPurifier, dual_consistency_loss
    
    purifier = DiffusionPurifier().to(image.device)
    purified = image.clone().detach()
    step = 0
    loss_val = float('inf')
    
    while step < max_steps and loss_val > loss_threshold:
        noise_level = 0.1 * trigger_intensity
        pred1 = purifier(purified, noise_level)
        pred2 = purifier(pred1, noise_level)
        purified = purified + 0.5 * (pred1 - purified) + 0.5 * (pred2 - pred1)
        loss = dual_consistency_loss(image, purified, torch.zeros_like(image))
        loss_val = loss.item()
        step += 1
        
    return purified, step

def upr_adaptive_purification(image, trigger_intensity, mix_coef=0.5, noise_schedule='linear', max_steps=MAX_STEPS):
    """
    UPR purification with controlled randomness and adaptive noise scheduling.
    
    Args:
        image: Input image tensor
        trigger_intensity: Intensity of the trigger
        mix_coef: Mixing coefficient for controlled randomness
        noise_schedule: Type of noise schedule ('linear', 'exponential', 'constant')
        max_steps: Maximum number of steps
        
    Returns:
        tuple: (Purified image, Loss history)
    """
    from src.train import DiffusionPurifier, dual_consistency_loss
    
    purifier = DiffusionPurifier().to(image.device)
    purified = image.clone().detach()
    loss_history = []
    
    for step in range(max_steps):
        if noise_schedule == 'linear':
            noise_factor = (max_steps - step) / max_steps
        elif noise_schedule == 'exponential':
            noise_factor = np.exp(-step / max_steps)
        else:
            noise_factor = 1.0  # constant schedule

        noise = torch.randn_like(image) * (0.1 * trigger_intensity * noise_factor)
        det_update = purifier(purified, noise.std())
        purified = purified + mix_coef * noise + (1 - mix_coef) * (det_update - purified)
        loss = dual_consistency_loss(image, purified, noise)
        loss_history.append(loss.item())
        
    return purified, loss_history

def save_plot(plt_figure, filename):
    """
    Save a matplotlib figure as a high-quality PDF.
    
    Args:
        plt_figure: Matplotlib figure to save
        filename: Name of the file to save (without extension)
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, f"{filename}.pdf")
    plt_figure.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")
    plt.close(plt_figure)
