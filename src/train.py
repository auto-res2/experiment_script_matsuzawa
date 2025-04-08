"""
Training module for the Optimized Characteristic Resampling (OCR) experiment.

This module contains the model definition and training functions for the OCR method.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class DummyDDPM(nn.Module):
    """
    Dummy Denoising Diffusion Probabilistic Model for testing OCR method.
    
    This is a simplified model that mimics the behavior of a diffusion model.
    """
    def __init__(self):
        super(DummyDDPM, self).__init__()
        self.layer = nn.Linear(3*32*32, 3*32*32)

    def forward(self, x, guidance_scale):
        """
        Forward pass for the dummy DDPM model.
        
        Args:
            x (torch.Tensor): Input tensor [batch, 3, 32, 32]
            guidance_scale (float): Guidance scale for the model
            
        Returns:
            torch.Tensor: Updated tensor after one reverse diffusion step
        """
        x_flat = x.view(x.size(0), -1)
        update = self.layer(x_flat)
        update = update.view_as(x)
        return x - guidance_scale * update * 1e-3

def base_method_step(model, x, guidance_scale, max_iters=50, loss_threshold=1e-3):
    """
    Fixed-point iteration using the Base Method.
    
    Args:
        model (nn.Module): The diffusion model
        x (torch.Tensor): Input tensor
        guidance_scale (float): Guidance scale for the model
        max_iters (int): Maximum number of iterations
        loss_threshold (float): Threshold for convergence
        
    Returns:
        tuple: (updated tensor, iteration count, final loss)
    """
    iter_count = 0
    loss = float('inf')
    while loss > loss_threshold and iter_count < max_iters:
        x_next = model(x, guidance_scale)
        loss_tensor = torch.nn.functional.mse_loss(x_next, x)
        loss = loss_tensor.item()
        x = x_next.detach()
        iter_count += 1
    return x, iter_count, loss

def ocr_method_step(model, x, guidance_scale, optimizer_cls=optim.Adam, max_iters=50, loss_threshold=1e-3):
    """
    OCR method that uses an inner optimization loop.
    
    Args:
        model (nn.Module): The diffusion model
        x (torch.Tensor): Input tensor
        guidance_scale (float): Guidance scale for the model
        optimizer_cls: Optimizer class to use
        max_iters (int): Maximum number of iterations
        loss_threshold (float): Threshold for convergence
        
    Returns:
        tuple: (updated tensor, iteration count, final loss)
    """
    x = x.clone().detach().requires_grad_(True)
    optimizer = optimizer_cls([x], lr=1e-2)
    iter_count = 0
    loss = float('inf')
    while iter_count < max_iters:
        optimizer.zero_grad()
        x_next = model(x, guidance_scale)
        composite_loss = torch.nn.functional.mse_loss(x_next, x)
        composite_loss.backward()
        optimizer.step()
        loss = composite_loss.item()
        if loss < loss_threshold:
            break
        iter_count += 1
    return x.detach(), iter_count, loss

def composite_loss(x, x_next, lambdas):
    """
    A composite loss function that simulates L_FP, L_guidance, and L_noise.
    
    Args:
        x (torch.Tensor): Original tensor
        x_next (torch.Tensor): Updated tensor
        lambdas (dict): Dictionary with keys "fp", "guidance", "noise" and corresponding weight values
        
    Returns:
        torch.Tensor: Weighted sum of the component losses
    """
    l_fp = torch.nn.functional.mse_loss(x_next, x)
    l_guidance = torch.nn.functional.l1_loss(x_next, x)
    l_noise = torch.mean((x_next - x)**2)
    total_loss = lambdas["fp"] * l_fp + lambdas["guidance"] * l_guidance + lambdas["noise"] * l_noise
    return total_loss

def ocr_method_step_with_optimizer(model, x, guidance_scale, optimizer_name="adam",
                                  max_iters=50, loss_threshold=1e-3):
    """
    OCR method inner loop that allows switching between optimizers.
    
    Args:
        model (nn.Module): The diffusion model
        x (torch.Tensor): Input tensor
        guidance_scale (float): Guidance scale for the model
        optimizer_name (str): Name of the optimizer to use ("adam", "rmsprop", or "anderson")
        max_iters (int): Maximum number of iterations
        loss_threshold (float): Threshold for convergence
        
    Returns:
        tuple: (updated tensor, iteration count, final loss)
    """
    x = x.clone().detach().requires_grad_(True)

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam([x], lr=1e-2)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = optim.RMSprop([x], lr=1e-2)
    elif optimizer_name.lower() == "anderson":
        optimizer = optim.Adam([x], lr=1e-2)
        print("Using Anderson acceleration (placeholder using Adam)")
    else:
        raise ValueError("Unknown optimizer")
      
    iter_count = 0
    loss = float('inf')
    while iter_count < max_iters:
        optimizer.zero_grad()
        x_next = model(x, guidance_scale)
        loss_tensor = torch.nn.functional.mse_loss(x_next, x)
        loss = loss_tensor.item()
        loss_tensor.backward()
        optimizer.step()
        if loss < loss_threshold:
            break
        iter_count += 1
    return x.detach(), iter_count, loss

def train_with_hyperparams(hyperparams, model, optimizer_cls=optim.Adam, iterations=30):
    """
    Train a dummy model with specified hyperparameters.
    
    Args:
        hyperparams (tuple): Hyperparameters (lambda_fp, lambda_guidance, lambda_noise)
        model (nn.Module): The diffusion model
        optimizer_cls: Optimizer class to use
        iterations (int): Number of iterations
        
    Returns:
        list: Loss values over iterations
    """
    lambdas = {"fp": hyperparams[0], "guidance": hyperparams[1], "noise": hyperparams[2]}
    device = next(model.parameters()).device
    x = torch.randn(16, 3, 32, 32).to(device).requires_grad_(True)
    optimizer = optimizer_cls([x], lr=1e-2)
    loss_values = []
    for _ in range(iterations):
        optimizer.zero_grad()
        x_next = model(x, guidance_scale=10)
        loss = composite_loss(x, x_next, lambdas)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
    return loss_values
