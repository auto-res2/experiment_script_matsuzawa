"""
Evaluation module for ANGAS experiments.

This module contains the implementation of the diffusion scheduler, samplers,
and evaluation functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


class MyDiffusionScheduler:
    """
    Diffusion scheduler implementing both standard update and ANGAS updates.
    """
    def __init__(self, dt=0.1):
        """
        Initialize the diffusion scheduler.
        
        Args:
            dt: Time step size
        """
        self.dt = dt

    def compute_update(self, x, model):
        """
        Compute standard update (e.g., as in DDIM/DDPM).
        
        Args:
            x: Current state
            model: Score model
            
        Returns:
            Update direction
        """
        update = self.dt * model(x)
        return update

    def ideal_update(self, x):
        """
        Compute the ideal update based on the true Fokker–Planck dynamics.
        
        In a real experiment, this would be computed from the ideal dynamics.
        Here we simulate it as a scaled version of x itself (for demonstration).
        
        Args:
            x: Current state
            
        Returns:
            Ideal update direction
        """
        ideal = self.dt * (0.5 * x)
        return ideal

    def compute_highorder_update(self, x, model):
        """
        Compute high-order update for accelerated convergence.
        
        Args:
            x: Current state
            model: Score model
            
        Returns:
            High-order update direction
        """
        update = self.dt * 1.2 * model(x)
        return update

    def compute_nonlinear_correction(self, x, update, scale=1.0):
        """
        Compute adaptive nonlinear correction.
        
        Args:
            x: Current state
            update: Update direction
            scale: Scale factor for the correction
            
        Returns:
            Nonlinear correction term
        """
        correction = scale * (-0.05 * update)
        return correction


def base_sampler(model, z, scheduler, num_steps, device='cuda'):
    """
    Implements the base method sampling using a conventional update.
    
    Args:
        model: Score model
        z: Initial noise
        scheduler: Diffusion scheduler
        num_steps: Number of sampling steps
        device: Device to compute on
        
    Returns:
        Samples and error metrics
    """
    model.eval()
    x = z.clone()
    errors = []
    
    print("\n[Experiment 1] Running Base Method sampler:")
    for step in tqdm(range(num_steps), desc="Base Method"):
        update = scheduler.compute_update(x, model)
        x_next = x + update
        ref_update = scheduler.ideal_update(x)
        error = torch.norm(x_next - (x + ref_update))
        errors.append(error.item())
        x = x_next
        print(f"  Step {step+1:02d}/{num_steps}, Error Norm: {error.item():.4f}")
    
    return x, errors


def angas_sampler(model, z, scheduler, num_steps, device='cuda'):
    """
    Implements the ANGAS sampler with high‐order updates and nonlinear correction.
    
    Args:
        model: Score model
        z: Initial noise
        scheduler: Diffusion scheduler
        num_steps: Number of sampling steps
        device: Device to compute on
        
    Returns:
        Samples and error metrics
    """
    model.eval()
    x = z.clone()
    errors = []
    
    print("\n[Experiment 1] Running ANGAS sampler:")
    for step in tqdm(range(num_steps), desc="ANGAS Method"):
        update = scheduler.compute_highorder_update(x, model)
        correction = scheduler.compute_nonlinear_correction(x, update)
        x_next = x + update + correction
        ref_update = scheduler.ideal_update(x)
        error = torch.norm(x_next - (x + ref_update))
        errors.append(error.item())
        x = x_next
        print(f"  Step {step+1:02d}/{num_steps}, Error Norm: {error.item():.4f}")
    
    return x, errors


def angas_variant_sampler(model, z, scheduler, num_steps, apply_correction=True, device='cuda'):
    """
    Sampler that can toggle adaptive nonlinear correction on/off.
    
    Args:
        model: Score model
        z: Initial noise
        scheduler: Diffusion scheduler
        num_steps: Number of sampling steps
        apply_correction: Whether to apply the nonlinear correction
        device: Device to compute on
        
    Returns:
        Samples and residual metrics
    """
    model.eval()
    x = z.clone()
    residuals = []
    
    variant = "with correction" if apply_correction else "without correction"
    print(f"\n[Experiment 2] Running ANGAS variant sampler ({variant}):")
    
    for step in tqdm(range(num_steps), desc=f"ANGAS {variant}"):
        update = scheduler.compute_highorder_update(x, model)
        if apply_correction:
            correction = scheduler.compute_nonlinear_correction(x, update)
        else:
            correction = 0.0  # no correction applied
        x_next = x + update + correction
        ref_update = scheduler.ideal_update(x)
        residual = torch.norm(x_next - (x + ref_update))
        residuals.append(residual.item())
        x = x_next
        print(f"  Step {step+1:02d}/{num_steps}, Residual Norm: {residual.item():.4f}")
    
    return x, residuals


def angas_dynamic_sampler(model, z, scheduler, num_steps, threshold, device='cuda'):
    """
    Sampler with dynamic adjustment of nonlinear correction based on a threshold.
    
    Args:
        model: Score model
        z: Initial noise
        scheduler: Diffusion scheduler
        num_steps: Number of sampling steps
        threshold: Error threshold for dynamic adjustment
        device: Device to compute on
        
    Returns:
        Samples, correction magnitudes, and error norms
    """
    model.eval()
    x = z.clone()
    correction_magnitudes = []
    error_norms = []
    
    for step in tqdm(range(num_steps), desc=f"ANGAS Dynamic (threshold={threshold})"):
        update = scheduler.compute_highorder_update(x, model)
        local_error = torch.norm(update)
        if local_error > threshold:
            correction = scheduler.compute_nonlinear_correction(x, update, scale=2.0)
        else:
            correction = scheduler.compute_nonlinear_correction(x, update, scale=1.0)
        x_next = x + update + correction

        correction_magnitude = torch.norm(correction).item()
        correction_magnitudes.append(correction_magnitude)

        ref_update = scheduler.ideal_update(x)
        error_norm = torch.norm(x_next - (x + ref_update)).item()
        error_norms.append(error_norm)

        x = x_next
    
    return x, correction_magnitudes, error_norms


def save_pdf_plot(data, labels, xlabel, ylabel, title, filename):
    """
    Save a plot as a high-quality PDF file.
    
    Args:
        data: List of data series to plot
        labels: List of labels for each data series
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename (will be saved in logs directory)
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    for i, series in enumerate(data):
        plt.plot(range(len(series)), series, label=labels[i])
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    output_path = f"logs/{filename}"
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as: {output_path}")
