"""
Model definitions for ANCD experiments.
"""
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.experiment_config import FEATURE_DIM

class BaselineTripleNetwork(nn.Module):
    """
    Baseline Triple Network (representing the traditional approach).
    """
    def __init__(self, feature_dim=FEATURE_DIM):
        super(BaselineTripleNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, feature_dim, 3, padding=1),
            nn.ReLU()
        )
        self.diffusion = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU()
        )
        self.aux = nn.Sequential(
            nn.Conv2d(feature_dim, 3, 3, padding=1)  # Predict noise map, same image dimensions
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        diffused = self.diffusion(features)
        out = self.aux(diffused)
        return out

class ANCDNetwork(nn.Module):
    """
    Adaptive Noise and Consistency Distillation (ANCD) Network.
    """
    def __init__(self, feature_dim=FEATURE_DIM):
        super(ANCDNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, feature_dim, 3, padding=1),
            nn.ReLU()
        )
        self.integrator = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 3, 3, padding=1)
        )
        
    def forward(self, x):
        shared_out = self.shared(x)
        
        with torch.no_grad():
            gradient_variance = torch.mean(torch.abs(shared_out))
            noise_level = 0.1 * torch.exp(-0.1 * gradient_variance)
        
        noise = torch.randn_like(shared_out) * noise_level
        shared_out = shared_out + noise
        
        out_initial = self.integrator(shared_out)
        
        with torch.no_grad():
            correction = 0.1 * (out_initial - x)
            shared_out_corrected = shared_out - correction
        
        out = self.integrator(shared_out_corrected)
        return out

class ANCDNetworkVariants(nn.Module):
    """
    ANCD Network with toggles for adaptive components (for ablation study).
    """
    def __init__(self, feature_dim=FEATURE_DIM, use_adaptive_noise=True, 
                use_high_order=True, use_consistency_loss=True):
        super(ANCDNetworkVariants, self).__init__()
        self.use_adaptive_noise = use_adaptive_noise
        self.use_high_order = use_high_order
        self.use_consistency_loss = use_consistency_loss
        
        self.shared = nn.Sequential(
            nn.Conv2d(3, feature_dim, 3, padding=1),
            nn.ReLU()
        )
        
        if self.use_high_order:
            self.integrator = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim, 3, 3, padding=1)
            )
        else:
            self.integrator = nn.Conv2d(feature_dim, 3, 3, padding=1)  # Simple Euler step simulation
            
    def forward(self, x):
        shared_out = self.shared(x)
        
        if self.use_adaptive_noise:
            with torch.no_grad():
                gradient_variance = torch.mean(torch.abs(shared_out))
                noise_level = 0.1 * torch.exp(-0.1 * gradient_variance)
            
            noise = torch.randn_like(shared_out) * noise_level
            shared_out = shared_out + noise
        
        if self.use_high_order:
            out_initial = self.integrator(shared_out)
            
            if self.use_consistency_loss:
                with torch.no_grad():
                    correction = 0.1 * (out_initial - x)
                    shared_out_corrected = shared_out - correction
                
                out = self.integrator(shared_out_corrected)
            else:
                out = out_initial
        else:
            out = self.integrator(shared_out)
            
        return out
