import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class DiffusionModel(nn.Module):
    """
    A simple two-layer network for the neural correction module.
    In practice, you should use an architecture that fits your data.
    """
    def __init__(self, in_features):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, in_features)
        )
        
    def forward(self, x):
        return self.net(x)

def harmonic_ansatz_update(x, guidance_scale):
    """
    Dummy harmonic ansatz update: in practice,
    replace with your fixed-point iteration scheme.
    """
    return x - guidance_scale * (x - 0.5)

def neural_correction_update(x, correction_module, guidance_scale):
    """
    Combines the analytic update and a learned neural correction.
    Here we use a simple obtained sum.
    """
    analytic = harmonic_ansatz_update(x, guidance_scale)
    neural_correction = correction_module(x)
    return analytic + guidance_scale * neural_correction

def train_correction_module(data, guided_only=True, num_epochs=50, batch_size=64, lr=1e-3):
    """
    Trains the correction module on either guided-only (analytic target)
    or a joint-space target that includes an additional non-linear term.
    """
    model = DiffusionModel(in_features=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def target_func(x):
        base = harmonic_ansatz_update(x, guidance_scale=0.1)
        if guided_only:
            return base
        else:
            joint_effect = 0.1 * torch.sin(x)
            return base + joint_effect

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for (batch,) in dataloader:
            optimizer.zero_grad()
            pred = model(batch)
            target = target_func(batch)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(data)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4e}")
    
    save_path = '../models/correction_module_guided_only.pt' if guided_only else '../models/correction_module_joint_space.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Trained model saved to {save_path}")
    
    return model, losses
