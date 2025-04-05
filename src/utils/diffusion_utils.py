"""
Utility functions for diffusion models and KL divergence calculation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """
    Generate adversarial examples using the Fast Gradient Sign Method (FGSM).
    
    Args:
        image: Input image
        epsilon: Perturbation magnitude
        data_grad: Gradient of the loss with respect to the input image
        
    Returns:
        Perturbed image (adversarial example)
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def insert_trigger(image: torch.Tensor, trigger_value: float = 1.0) -> torch.Tensor:
    """
    Insert a trigger (a small patch in the top-left corner) into an image.
    
    Args:
        image: Input image with shape [C,H,W]
        trigger_value: Value to use for the trigger patch
        
    Returns:
        Triggered image
    """
    triggered = image.clone()
    triggered[:, :4, :4] = trigger_value
    return triggered

def kl_divergence(latent_sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Compute approximate KL divergence between a sample and a diagonal Gaussian.
    
    Args:
        latent_sample: Sample from a latent distribution
        mean: Mean of the reference Gaussian distribution
        std: Standard deviation of the reference Gaussian distribution
        
    Returns:
        KL divergence value
    """
    latent_mu = latent_sample.mean(dim=0)
    latent_std = latent_sample.std(dim=0) + 1e-8
    kl = torch.sum(torch.log(std/latent_std) + 
                  (latent_std**2 + (latent_mu - mean)**2) / (2*std**2) - 0.5)
    return kl

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
