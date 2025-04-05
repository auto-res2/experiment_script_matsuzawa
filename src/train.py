"""
Training module for TEDP (Trigger-Eradicating Diffusion Purification).
Implements the diffusion model and training pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class DiffusionModel(nn.Module):
    """
    A simple convolutional autoencoder. For demonstration, it simply tries to reconstruct the input.
    An intermediate layer is selected for latent extraction.
    """
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # downsample
        self.latent_layer = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # upsample
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        latent = F.relu(self.latent_layer(x))
        x = F.relu(self.deconv1(latent))
        x = torch.sigmoid(self.conv3(x))  # output in [0,1]
        return x


def train_pipeline(model, data_loader, purification=None, num_epochs=5, adaptive=False, 
                   record_func=None, device='cuda'):
    """
    model: the diffusion model
    data_loader: provided training data
    purification: an instance of PurificationModule (or None for baseline)
    adaptive: if True adapt the purification strength with a simple update rule.
    record_func: optional callback to record metrics each epoch.
    """
    model.to(device)
    if purification is not None: 
        purification.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_history = []
    
    purification_strength = purification.noise_std_init if purification is not None else None
    
    latent_features_mean = None

    hook_handle = None
    latent_collection = []   # list to store latent representations
    def latent_hook(module, input, output):
        latent_collection.append(output.view(output.size(0), -1).detach().cpu().numpy())
    
    if record_func is not None and record_func.__name__ in ['record_latents', 'experiment2_recorder']:
        hook_handle = model.latent_layer.register_forward_hook(latent_hook)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, _ in data_loader:
            images = images.to(device)
            if adaptive and (purification is not None):
                latent_stat = images.mean().item()
                if latent_stat > 0.5:
                    purification_strength = min(purification_strength*1.05, 1.0)
                else:
                    purification_strength = max(purification_strength*0.95, 0.01)
                purification.noise_std_init = purification_strength

            if purification is not None:
                images = purification(images)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.mse_loss(outputs, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss/len(data_loader)
        loss_history.append(avg_loss)
        print("Epoch {}: Loss {:.4f}".format(epoch, avg_loss))
        if record_func is not None:
            record_func(epoch, avg_loss, purification_strength if adaptive and purification is not None else None)
    
    if hook_handle is not None:
        hook_handle.remove()
    return loss_history, latent_collection
