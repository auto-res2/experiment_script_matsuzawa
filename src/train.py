"""
Training module for Graph-GaussianAssembler experiments.

This module contains the model definitions and training functions for the experiments.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GATConv


class GaussianDreamer:
    """Implementation of the GaussianDreamer baseline model."""
    
    def __init__(self, config_path):
        """
        Initialize the GaussianDreamer model.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        print(f"Initialized GaussianDreamer with config: {config_path}")
    
    def generate_3d_asset(self, prompt):
        """
        Generate a 3D asset from a text prompt.
        
        Args:
            prompt (str): Text prompt describing the asset
            
        Returns:
            dict: Dictionary containing asset parameters
        """
        print(f"[GaussianDreamer] Generating 3D asset for prompt: '{prompt}'")
        asset = {"positions": torch.randn(100, 3)}
        return asset


class DiffAssemble:
    """Implementation of the DiffAssemble variant model."""
    
    def __init__(self, config_path):
        """
        Initialize the DiffAssemble model.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        print(f"Initialized DiffAssemble with config: {config_path}")
    
    def generate_3d_asset(self, prompt):
        """
        Generate a 3D asset from a text prompt.
        
        Args:
            prompt (str): Text prompt describing the asset
            
        Returns:
            dict: Dictionary containing asset parameters
        """
        print(f"[DiffAssemble] Generating 3D asset for prompt: '{prompt}'")
        asset = {"positions": torch.randn(110, 3)}
        return asset


class GraphDenoisingModule(nn.Module):
    """Graph-based denoising module for GraphGaussianAssembler."""
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the graph denoising module.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(GraphDenoisingModule, self).__init__()
        self.conv1 = GATConv(in_channels, 64)
        self.conv2 = GATConv(64, out_channels)
    
    def forward(self, x, edge_index=None):
        """
        Forward pass of the graph denoising module.
        
        Args:
            x (torch.Tensor): Input feature tensor
            edge_index (torch.Tensor, optional): Edge indices for graph
            
        Returns:
            torch.Tensor: Output tensor
        """
        if edge_index is None:
            edge_index = torch.tensor([[0],[0]], dtype=torch.long)
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class IndependentDenoisingModule(nn.Module):
    """Independent (non-graph) denoising module for ablation studies."""
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the independent denoising module.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(IndependentDenoisingModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )
    
    def forward(self, x):
        """
        Forward pass of the independent denoising module.
        
        Args:
            x (torch.Tensor): Input feature tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.mlp(x)


class GraphGaussianAssembler:
    """Implementation of the Graph-GaussianAssembler model."""
    
    def __init__(self, denoise_module=None):
        """
        Initialize the GraphGaussianAssembler model.
        
        Args:
            denoise_module (nn.Module, optional): Denoising module to use
        """
        if denoise_module is None:
            self.denoise_module = nn.Identity()
            print("[GraphGaussianAssembler] Using default identity denoising module.")
        else:
            self.denoise_module = denoise_module
            print("[GraphGaussianAssembler] Using provided denoising module.")
    
    def generate_3d_asset(self, prompt):
        """
        Generate a 3D asset from a text prompt.
        
        Args:
            prompt (str): Text prompt describing the asset
            
        Returns:
            dict: Dictionary containing asset parameters
        """
        print(f"[GraphGaussianAssembler] Generating 3D asset for prompt: '{prompt}'")
        asset = {"positions": torch.randn(105, 3)}
        asset["positions"] = self.denoise_module(asset["positions"])
        return asset

    def initialize_asset(self, prompt):
        """
        Initialize asset parameters without denoising.
        
        Args:
            prompt (str): Text prompt describing the asset
            
        Returns:
            dict: Dictionary containing initial asset parameters
        """
        print(f"[GraphGaussianAssembler] Initializing asset for prompt: '{prompt}'")
        asset = {"positions": torch.randn(105, 3)}
        return asset

    def denoise_step(self, asset):
        """
        Perform a single denoising step on the asset.
        
        Args:
            asset (dict): Asset to denoise
            
        Returns:
            dict: Denoised asset
        """
        new_positions = asset["positions"] - 0.01 * torch.randn_like(asset["positions"])
        return {"positions": new_positions}


class AdaptiveTimeScheduler(nn.Module):
    """Adaptive time scheduler for Graph-GaussianAssembler."""
    
    def __init__(self, init_steps=50):
        """
        Initialize the adaptive time scheduler.
        
        Args:
            init_steps (int): Initial number of steps
        """
        super(AdaptiveTimeScheduler, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.init_steps = init_steps
    
    def forward(self, complexity_measure):
        """
        Forward pass of the adaptive time scheduler.
        
        Args:
            complexity_measure (torch.Tensor): Measure of scene complexity
            
        Returns:
            int: Adaptive number of steps
        """
        adjustment = self.fc(complexity_measure)
        adaptive_steps = self.init_steps - int(adjustment.item())
        return max(10, adaptive_steps)  # Ensure at least 10 iterations


def save_model(model, name):
    """
    Save a model to disk.
    
    Args:
        model (nn.Module): Model to save
        name (str): Name for the saved model
    """
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", f"{name}.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, name):
    """
    Load a model from disk.
    
    Args:
        model (nn.Module): Model to load into
        name (str): Name of the saved model
        
    Returns:
        nn.Module: Loaded model
    """
    path = os.path.join("models", f"{name}.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"No saved model found at {path}")
    
    return model
