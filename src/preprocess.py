"""
Preprocessing module for RobustPurify-Backdoor Diffusion experiment.

This module contains functions for:
- Simulating purification (forward-reverse diffusion process)
- Creating dual-path datasets (raw and purified versions)
- Creating poisoned datasets with varying poisoning ratios
"""

import numpy as np
from scipy.integrate import ode
import torch
from torch.utils.data import TensorDataset
import random


def simulate_purification(image, t0=0, t1=1, dt=0.1):
    """
    Simulate a simple purification (forward-reverse diffusion) process 
    using an ODE solver (Heun's method via dopri5 integrator).
    
    Args:
        image: Input image to purify
        t0: Start time for simulation
        t1: End time for simulation
        dt: Time step for simulation
    
    Returns:
        Purified version of the input image
    """
    def diffusion_dynamics(t, y):
        return -0.1 * y

    y0 = image.flatten()
    solver = ode(diffusion_dynamics)
    solver.set_integrator('dopri5')
    solver.set_initial_value(y0, t0)
    while solver.successful() and solver.t < t1:
        solver.integrate(solver.t + dt)
    purified = solver.y.reshape(image.shape)
    return purified


def create_dual_path_dataset(num_samples=100, image_size=(64,64), seed=42):
    """
    Create a dataset of poisoning images with both raw and purified versions.
    Some images are "poisoned" with an artificial signal (e.g., a patch in the corner)
    while the purification process is simulated.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of the images (height, width)
        seed: Random seed for reproducibility
    
    Returns:
        PyTorch dataset with pairs of raw and purified images
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    raw_images = []
    purified_images = []
    
    for i in range(num_samples):
        img = np.random.rand(*image_size)
        if random.random() < 0.5:
            img[:5, :5] += 0.5  # backdoor signal added to top left corner
            
        raw_images.append(img)
        purified = simulate_purification(img)
        purified_images.append(purified)
    
    raw_data = torch.tensor(np.array(raw_images)).float()
    purified_data = torch.tensor(np.array(purified_images)).float()
    
    dataset = TensorDataset(raw_data, purified_data)
    print(f"Dual-path dataset created: {num_samples} samples")
    return dataset


def create_poisoned_dataset(total_samples, poisoning_ratio, image_size=(64,64), seed=42):
    """
    Create a dataset with a given poisoning ratio. 
    Clean images are random; poisoned images include a backdoor trigger (a patch).
    Each image is paired with a purified-version (simulate a smoothed version).
    
    Args:
        total_samples: Total number of samples to generate
        poisoning_ratio: Fraction of samples to poison (0.0 to 1.0)
        image_size: Size of the images (height, width)
        seed: Random seed for reproducibility
    
    Returns:
        PyTorch dataset with pairs of raw and purified images
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_poisoned = int(total_samples * poisoning_ratio)
    num_clean = total_samples - num_poisoned
    
    clean_images = np.random.rand(num_clean, *image_size)
    poisoned_images = np.random.rand(num_poisoned, *image_size)
    poisoned_images[:, :5, :5] += 0.5  # backdoor trigger
    
    all_images = np.concatenate([clean_images, poisoned_images], axis=0)
    purified_images = np.array([simulate_purification(img) for img in all_images])
    
    raw_data = torch.tensor(all_images).float()
    purified_data = torch.tensor(purified_images).float()
    
    dataset = TensorDataset(raw_data, purified_data)
    print(f"Poisoned dataset created: {total_samples} samples with poisoning ratio {poisoning_ratio*100:.2f}%")
    return dataset
