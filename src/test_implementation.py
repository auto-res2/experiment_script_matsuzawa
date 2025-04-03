"""
Test script to verify the ANGAS implementation.

This script runs a quick test to ensure all components are working correctly.
"""

import torch
import numpy as np
import time
import os

from train import ScoreModel
from evaluate import MyDiffusionScheduler
from preprocess import prepare_initial_noise


def test_components():
    """Test individual components of the implementation."""
    
    print("Testing components of the ANGAS implementation...\n")
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ScoreModel().to(device)
    print(f"Model created: {model.__class__.__name__}")
    
    scheduler = MyDiffusionScheduler(dt=0.1)
    print(f"Scheduler created: {scheduler.__class__.__name__}")
    
    z_init = prepare_initial_noise(batch_size=2, size=8, device=device)
    print(f"Initial noise shape: {z_init.shape}")
    
    output = model(z_init)
    print(f"Model output shape: {output.shape}")
    
    update = scheduler.compute_update(z_init, model)
    print(f"Standard update norm: {torch.norm(update).item():.4f}")
    
    highorder_update = scheduler.compute_highorder_update(z_init, model)
    print(f"High-order update norm: {torch.norm(highorder_update).item():.4f}")
    
    correction = scheduler.compute_nonlinear_correction(z_init, update)
    print(f"Nonlinear correction norm: {torch.norm(correction).item():.4f}")
    
    print("\nAll components tested successfully!")


if __name__ == "__main__":
    test_components()
