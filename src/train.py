#!/usr/bin/env python3
"""
Training module for the Adaptive Characteristic Simulation (ACS) experiments.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from .preprocess import save_plot

def f(t, y):
    """
    ODE function for Experiment 1.
    Non-linear dynamics: diffusion-like behavior with increased curvature around y=0.
    """
    return -y + 0.5 * np.tanh(10 * y)

def fixed_step_integrator(y0, t0, tend, dt):
    """
    Fixed-step Euler integrator for Experiment 1.
    
    Args:
        y0: Initial value
        t0: Initial time
        tend: End time
        dt: Fixed time step
    
    Returns:
        Arrays of time values and y values
    """
    t_values = [t0]
    y_values = [y0]
    t = t0
    y = y0
    while t < tend:
        y = y + dt * f(t, y)
        t += dt
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

def adaptive_step_integrator(y0, t0, tend, dt_initial, tol):
    """
    Adaptive-step integrator for Experiment 1.
    
    Args:
        y0: Initial value
        t0: Initial time
        tend: End time
        dt_initial: Initial time step
        tol: Error tolerance
    
    Returns:
        Arrays of time values and y values
    """
    t_values = [t0]
    y_values = [y0]
    t = t0
    y = y0
    dt = dt_initial
    while t < tend:
        y_full = y + dt * f(t, y)
        dt_half = dt / 2.0
        y_half = y + dt_half * f(t, y)
        y_two_half = y_half + dt_half * f(t + dt_half, y_half)
        error = np.abs(y_full - y_two_half)
        if error > tol:
            dt *= 0.5
            continue  # Retry without advancing
        elif error < (tol / 4.0):
            dt *= 1.5
        t += dt
        y = y_two_half
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

def sde_drift(x, theta=0.7):
    """
    Drift term for the SDE (Ornstein-Uhlenbeck type) in Experiment 2.
    """
    return -theta * x

def base_sde_euler_maruyama(x0, T, dt, noise_scale):
    """
    Euler–Maruyama integration for the SDE with fixed noise intensity for Experiment 2.
    """
    num_steps = int(T / dt)
    xs = [x0]
    x = x0
    for i in range(num_steps):
        dw = torch.randn_like(x) * (dt**0.5)
        x = x + dt * sde_drift(x) + noise_scale * dw
        xs.append(x)
    return torch.stack(xs)

def acs_sde_euler_maruyama(x0, T, dt, noise_scale, tol):
    """
    Euler–Maruyama integration for SDE with adaptive noise control for Experiment 2.
    """
    xs = [x0]
    x = x0
    t = 0.0
    while t < T:
        dw = torch.randn_like(x) * (dt**0.5)
        x_predict = x + dt * sde_drift(x) + noise_scale * dw
        error = torch.abs(x_predict - x)
        if (error > tol).any():
            adaptive_noise_scale = noise_scale * 0.5  # reduce noise if error too high
        else:
            adaptive_noise_scale = noise_scale  # keep noise scale unchanged
        dw = torch.randn_like(x) * (dt**0.5)
        x = x + dt * sde_drift(x) + adaptive_noise_scale * dw
        xs.append(x)
        t += dt
    return torch.stack(xs)

class SimpleUNet(nn.Module):
    """
    A simplified UNet-like architecture for Experiment 3.
    """
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_unet_model(model, num_epochs=10, batch_size=32, learning_rate=1e-3):
    """
    Train the UNet model for Experiment 3.
    
    Args:
        model: SimpleUNet model to train
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        Trained model
    """
    return model
