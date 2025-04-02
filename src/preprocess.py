import numpy as np
import torch


def generate_synthetic_data(n_samples=1000, dim=2, seed=42):
    """
    Generate synthetic data for diffusion experiments.
    
    Args:
        n_samples: Number of samples to generate
        dim: Dimension of each sample
        seed: Random seed for reproducibility
        
    Returns:
        torch.Tensor: Generated samples
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.randn(n_samples, dim)


def prepare_initial_samples(n_samples=500, dim=2, seed=43):
    """
    Prepare initial samples for diffusion process.
    
    Args:
        n_samples: Number of samples to generate
        dim: Dimension of each sample
        seed: Random seed for reproducibility
        
    Returns:
        torch.Tensor: Initial samples
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.randn(n_samples, dim)
