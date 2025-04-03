import torch
import torch.nn as nn
import numpy as np

def diffusion_step(inputs, noise_level=0.001):
    """Simulate one diffusion (or purification) step.
    
    Args:
        inputs: Input tensor (adversarial examples)
        noise_level: Level of noise to add
        
    Returns:
        Diffused tensor
    """
    return inputs * (1.0 - noise_level) + torch.randn_like(inputs) * noise_level

def calculate_adaptive_lambda(inputs):
    """Calculate adaptive lambda based on input characteristics.
    
    Args:
        inputs: Input tensor
        
    Returns:
        Adaptive lambda value
    """
    variance = torch.var(inputs, dim=[1, 2, 3], keepdim=True)
    return 1.0 + torch.tanh(variance)

def double_tweedie_transform(inputs, score_model, noise_level):
    """Apply double-Tweedie transformation for improved denoising.
    
    Args:
        inputs: Input tensor (adversarial examples)
        score_model: Model that estimates the score function
        noise_level: Current noise level
        
    Returns:
        Transformed tensor
    """
    score_estimate = score_model(inputs, noise_level)
    first_estimate = inputs + (noise_level**2) * score_estimate
    
    second_noise_level = noise_level * 0.9
    second_score = score_model(first_estimate, second_noise_level)
    second_estimate = first_estimate + (second_noise_level**2) * second_score
    
    return second_estimate

def check_consistency(current, previous, threshold=0.01):
    """Check if consistency criterion is met for adaptive step scheduling.
    
    Args:
        current: Current denoised output
        previous: Previous denoised output
        threshold: Consistency threshold
        
    Returns:
        Boolean indicating if consistency is met
    """
    consistency_metric = torch.mean((current - previous)**2)
    return consistency_metric < threshold
