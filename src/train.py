"""
Script for training models for Joint-Guided Bayesian Flow Networks (JG-BFN) experiments.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.neighbors import KernelDensity
import os
import matplotlib.pyplot as plt

class ScoreModel(nn.Module):
    """
    A small convolutional network to simulate a score-based model.
    It also includes a helper function get_latent_info to simulate auxiliary guidance information.
    """
    def __init__(self, in_channels=1, hidden_dim=32):
        super(ScoreModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        
        self.fc = nn.Linear(28*28, 28*28)
        
    def forward(self, x, t):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        score = self.conv3(out)
        return score

    def get_latent_info(self, x, t):
        """
        Simulate extraction of latent trajectory and guidance info.
        For demonstration, we simply flatten and pass through a linear layer.
        """
        batch_size = x.size(0)
        input_size = x.view(batch_size, -1).size(1)
        if not hasattr(self, 'fc') or self.fc.in_features != input_size:
            self.fc = nn.Linear(input_size, input_size).to(x.device)
        
        latent_traj = x.view(batch_size, -1)
        guidance_info = self.fc(latent_traj)
        guidance_info = guidance_info.view_as(x)
        return latent_traj, guidance_info

class SDERverseLoss(nn.Module):
    def forward(self, model, x_noisy, x_true, t):
        score = model(x_noisy, t)
        loss = ((score - (x_true - x_noisy))**2).mean()
        return loss

class GuidanceLoss(nn.Module):
    def forward(self, latent_traj, guidance_info):
        guidance_info_flat = guidance_info.view(latent_traj.shape)
        loss = ((latent_traj - guidance_info_flat)**2).mean()
        return loss

def train_ablation_experiment(dataloader, num_epochs, batch_size, weight_guidance, device):
    """
    Train two model variants for ablation study:
    - Full Loss Model (with auxiliary guidance)
    - Ablation Model (without auxiliary guidance)
    
    Args:
        dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        weight_guidance: Weight for guidance loss
        device: Device to run training on
        
    Returns:
        model_full: Trained full loss model
        model_ablation: Trained ablation model
        loss_history_full: Loss history for full model
        loss_history_ablation: Loss history for ablation model
    """
    sde_loss = SDERverseLoss()
    guidance_loss = GuidanceLoss()
    
    model_full = ScoreModel(in_channels=1).to(device)
    model_ablation = ScoreModel(in_channels=1).to(device)
    
    optimizer_full = optim.Adam(model_full.parameters(), lr=1e-3)
    optimizer_ablation = optim.Adam(model_ablation.parameters(), lr=1e-3)

    loss_history_full = []
    loss_history_ablation = []
    
    for epoch in range(num_epochs):
        running_loss_full = 0.0
        running_loss_ablation = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            batch = x.size(0)
            
            t = torch.rand(batch, 1, device=device)
            noise = torch.randn_like(x)
            x_noisy = x + noise
            
            optimizer_full.zero_grad()
            loss_val_full = sde_loss(model_full, x_noisy, x, t)
            latent_traj, guidance_info = model_full.get_latent_info(x_noisy, t)
            guidance_val = guidance_loss(latent_traj, guidance_info)
            loss_val_full += weight_guidance * guidance_val
            loss_val_full.backward()
            optimizer_full.step()
            running_loss_full += loss_val_full.item()
            
            optimizer_ablation.zero_grad()
            loss_val_ab = sde_loss(model_ablation, x_noisy, x, t)
            loss_val_ab.backward()
            optimizer_ablation.step()
            running_loss_ablation += loss_val_ab.item()
            
        avg_loss_full = running_loss_full / len(dataloader)
        avg_loss_ab = running_loss_ablation / len(dataloader)
        loss_history_full.append(avg_loss_full)
        loss_history_ablation.append(avg_loss_ab)
        print(f"Epoch {epoch+1}/{num_epochs}, Full Loss Model: {avg_loss_full:.4f}, Ablation Model: {avg_loss_ab:.4f}")
    
    return model_full, model_ablation, loss_history_full, loss_history_ablation

def train_adaptive_experiment(dataloader, num_epochs, batch_size, fixed_weight, device):
    """
    Train two model variants for adaptive weighting study:
    - Fixed Weight Model (with fixed guidance weight)
    - Adaptive Weight Model (with density-based adaptive guidance weight)
    
    Args:
        dataloader: DataLoader for synthetic data
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        fixed_weight: Fixed weight for guidance loss
        device: Device to run training on
        
    Returns:
        model_fixed: Trained fixed weight model
        model_adapt: Trained adaptive weight model
        loss_history_fixed: Loss history for fixed model
        loss_history_adapt: Loss history for adaptive model
    """
    class SyntheticScoreModel(nn.Module):
        def __init__(self, hidden_dim=16):
            super(SyntheticScoreModel, self).__init__()
            input_size = 4  # 1*2*2 = 4
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_size)
            )
            self.fc = nn.Linear(input_size, input_size)
            
        def forward(self, x, t):
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            score = self.net(x_flat)
            return score.view_as(x)
            
        def get_latent_info(self, x, t):
            batch_size = x.size(0)
            latent_traj = x.view(batch_size, -1)
            guidance_info = self.fc(latent_traj)
            return latent_traj, guidance_info.view_as(x)
    
    model_fixed = SyntheticScoreModel(hidden_dim=16).to(device)
    model_adapt = SyntheticScoreModel(hidden_dim=16).to(device)
    optimizer_fixed = optim.Adam(model_fixed.parameters(), lr=1e-3)
    optimizer_adapt = optim.Adam(model_adapt.parameters(), lr=1e-3)
    
    sde_loss = SDERverseLoss()
    guidance_loss = GuidanceLoss()
    
    loss_history_fixed = []
    loss_history_adapt = []
    
    for epoch in range(num_epochs):
        running_fixed = 0.0
        running_adapt = 0.0
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            batch = batch_x.size(0)
            t = torch.rand(batch, 1, device=device)
            noise = torch.randn_like(batch_x)
            x_noisy = batch_x + noise
            
            optimizer_fixed.zero_grad()
            loss_fixed = sde_loss(model_fixed, x_noisy, batch_x, t)
            latent_traj, guidance_info = model_fixed.get_latent_info(x_noisy, t)
            fixed_guidance = guidance_loss(latent_traj, guidance_info)
            loss_fixed += fixed_weight * fixed_guidance
            loss_fixed.backward()
            optimizer_fixed.step()
            running_fixed += loss_fixed.item()
            
            optimizer_adapt.zero_grad()
            loss_adapt = sde_loss(model_adapt, x_noisy, batch_x, t)
            x_np = batch_x.view(batch, -1).detach().cpu().numpy()
            densities = compute_local_density(x_np)
            adaptive_weights_np = 1.0 / (densities + 1e-5)
            adaptive_weights = torch.tensor(adaptive_weights_np, dtype=torch.float32, device=device)
            adaptive_weights = adaptive_weights.view(batch, 1, 1, 1)
            
            latent_traj_adapt, guidance_info_adapt = model_adapt.get_latent_info(x_noisy, t)
            guidance_val_adapt = guidance_loss(latent_traj_adapt, guidance_info_adapt)
            loss_adapt += fixed_weight * guidance_val_adapt * adaptive_weights.mean()
            loss_adapt.backward()
            optimizer_adapt.step()
            running_adapt += loss_adapt.item()
            
        avg_fixed = running_fixed / len(dataloader)
        avg_adapt = running_adapt / len(dataloader)
        loss_history_fixed.append(avg_fixed)
        loss_history_adapt.append(avg_adapt)
        print(f"Epoch {epoch+1}/{num_epochs}, Fixed: {avg_fixed:.4f}, Adaptive: {avg_adapt:.4f}")
    
    return model_fixed, model_adapt, loss_history_fixed, loss_history_adapt

def compute_local_density(x_samples_np):
    """
    Compute a simple Kernel Density Estimate for the given samples.
    x_samples_np is assumed to be 2D: [num_samples, features].
    """
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x_samples_np)
    log_density = kde.score_samples(x_samples_np)
    density = np.exp(log_density)
    return density
