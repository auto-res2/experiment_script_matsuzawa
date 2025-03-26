"""
Utility functions for DiffuSynerMix experiments.
Contains implementations of mixup variants and helper functions.
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

def standard_mixup(x, y, alpha=1.0):
    """Standard mixup implementation."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def synermix(x, y, alpha=1.0):
    """
    SynerMix implementation that combines intra-class mixup with standard mixup.
    """
    return standard_mixup(x, y, alpha)

class DiffuSynerMixModule(nn.Module):
    """
    DiffuSynerMix module with a trainable direction predictor and a multi-step diffusion process.
    """
    def __init__(self, in_channels, hidden_dim=64, num_steps=5, noise_std=0.1):
        super(DiffuSynerMixModule, self).__init__()
        self.direction_predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        )
        self.num_steps = num_steps
        self.noise_std = noise_std

    def forward(self, x, y):
        direction = self.direction_predictor(x)
        mixed = x.clone()
        for _ in range(self.num_steps):
            noise = torch.randn_like(mixed) * self.noise_std
            mixed = mixed + direction - noise
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        lam = torch.distributions.Beta(1.0, 1.0).sample().item()
        mixed_x = lam * mixed + (1 - lam) * mixed[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

def apply_mixup(x, y, mode='standard', **kwargs):
    """
    Dispatch function that applies the desired mixup variant.
    """
    if mode == 'standard':
        return standard_mixup(x, y, **kwargs)
    elif mode == 'synermix':
        return synermix(x, y, **kwargs)
    elif mode == 'diffusynermix':
        return kwargs['diffu_module'](x, y)
    else:
        raise ValueError(f"Invalid mixup mode: {mode}")

def get_diffusynermix_module(use_direction_predictor=True, num_steps=5, noise_std=0.1):
    """
    Returns a DiffuSynerMix module with modifications for ablation studies.
    """
    class AblatedDiffuSynerMixModule(DiffuSynerMixModule):
        def forward(self, x, y):
            if not use_direction_predictor:
                direction = torch.zeros_like(x)
            else:
                direction = self.direction_predictor(x)
            mixed = x.clone()
            for _ in range(num_steps):
                noise = torch.randn_like(mixed) * noise_std
                mixed = mixed + direction - noise
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            lam = torch.distributions.Beta(1.0, 1.0).sample().item()
            mixed_x = lam * mixed + (1 - lam) * mixed[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam
    return AblatedDiffuSynerMixModule(in_channels=3, num_steps=num_steps, noise_std=noise_std)

def extract_features(model, loader, device):
    """
    Extract features from the penultimate layer of the model using a forward hook.
    """
    model.eval()
    features_list = []
    labels_list = []
    extracted_features = []

    def hook(module, input, output):
        extracted_features.append(input[0].detach().cpu().numpy())

    hook_handle = model.fc.register_forward_hook(hook)
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            labels_list.append(labels.numpy())
    hook_handle.remove()
    features = np.concatenate(extracted_features, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels
