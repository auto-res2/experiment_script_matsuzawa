#!/usr/bin/env python
"""
Utility functions for the HEDA experiments.
"""
import torch
import numpy as np

def initialize_states(shape, seed=42):
    """Initialize a tensor of given shape with random values and fixed seed."""
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.randn(shape, device=device)

def compute_error(prev, current):
    """Compute L2 norm error between two states."""
    return torch.norm(current - prev).item()

def base_method_update(state, step_size=0.1, **kwargs):
    """
    Base Method update: a simple fixed‐point iteration.
    For example, assume convergence toward zero (gradient descent on 1/2||state||^2)
    """
    grad = -state  # gradient is -state so update drives state towards zero
    return state + step_size * grad

def heda_update(state, step_size=0.1, memory_size=1, **kwargs):
    """
    HEDA update: first perform a discrete projection (simulated via rounding),
    then a continuous update with an extra term that (in our simulation)
    uses the memory_size parameter to "accelerate" or smooth the update.
    """
    # Discrete projection (simulate smoothing by rounding)
    discrete_state = torch.round(state * 10) / 10.0
    # The difference between continuous and discrete state:
    diff = discrete_state - state
    # Use memory_size to scale the acceleration effect (the larger the memory, the less aggressive)
    memory_factor = 1.0 / memory_size if memory_size > 0 else 1.0
    # Update: similar to base update but with additional correction based on discrete state.
    return state + step_size * (-diff) + step_size * (-memory_factor * discrete_state)
