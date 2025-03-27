"""
Scripts for training models.

This module implements the training procedures for D-DAME and baseline diffusion models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

class UNet(nn.Module):
    def __init__(self, T, ch=64, ch_mult=(1, 2, 4, 8), attn=False, num_res_blocks=2, dropout=0.1, num_labels=None):
        super(UNet, self).__init__()
        self.T = T
        self.ch = ch
        self.conv1 = nn.Conv2d(3, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(ch, 3, kernel_size=3, padding=1)
    
    def forward(self, x, t, labels=None):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.out_conv(x)
        return x

class DMRE(nn.Module):
    def __init__(self, input_dim):
        super(DMRE, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Predict risk score
        )
    def forward(self, features):
        risk = self.fc(features)
        damping = torch.sigmoid(risk)
        return risk, damping

class DDAMEWrapper(nn.Module):
    def __init__(self, base_model, dmre):
        super(DDAMEWrapper, self).__init__()
        self.base_model = base_model
        self.dmre = dmre  # for computing risk scores and damping factors
        
    def forward(self, x, t, labels=None, extract_features=False):
        features = torch.randn(x.size(0), 64, device=x.device)  # placeholder features
        risk, damping = self.dmre(features)
        out = self.base_model(x, t, labels)
        if extract_features:
            return out, risk, damping, features
        return out, risk, damping

def train_epoch_baseline(model, optimizer, dataloader, T, lambda_threshold=0.5, epoch_num=0, writer=None, max_iters=10):
    model.train()
    losses, grad_norms = [], []
    iter_count = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        if iter_count >= max_iters:
            break
        
        device = next(model.parameters()).device
        data = data.to(device)
        t = torch.randint(0, T, (data.size(0),), device=device).float()

        optimizer.zero_grad()
        output = model(data, t)
        loss = ((output - data)**2).mean()
        if loss.item() < lambda_threshold:
            loss = torch.tensor(0.0, requires_grad=True, device=device)
            print(f"[Baseline][Epoch {epoch_num}] Batch {batch_idx}: Loss below threshold, skipping grad update.")
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm)
        if writer is not None:
            writer.add_scalar("Baseline/loss", loss.item(), epoch_num*max_iters + iter_count)
            writer.add_scalar("Baseline/grad_norm", grad_norm, epoch_num*max_iters + iter_count)
        print(f"[Baseline][Epoch {epoch_num}][Batch {batch_idx}] Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
        iter_count += 1

    return {"loss": losses, "grad_norm": grad_norms}

def train_epoch_ddame(model, optimizer, dataloader, T, use_probe=False, probe_intensity=0.1, 
                     epoch_num=0, writer=None, max_iters=10, variant="full"):
    """
    Train a D-DAME model for one epoch.
    
    Args:
        model: The D-DAME model
        optimizer: Optimizer for training
        dataloader: DataLoader providing training data
        T: Number of diffusion steps
        use_probe: Whether to apply memorization probes
        probe_intensity: Intensity of probes (if used)
        epoch_num: Current epoch number
        writer: TensorBoard SummaryWriter (optional)
        max_iters: Maximum number of iterations per epoch
        variant: D-DAME variant ("full", "no_dmre", or "no_ensemble")
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    losses, grad_norms, risk_values, damping_values = [], [], [], []
    iter_count = 0
    
    for batch_idx, (data, _) in enumerate(dataloader):
        if iter_count >= max_iters:
            break
        
        device = next(model.parameters()).device
        data = data.to(device)
        
        if use_probe:
            from utils.model_utils import add_memorization_probe
            data = add_memorization_probe(data.clone(), probe_intensity=probe_intensity)
            print(f"[D-DAME][Epoch {epoch_num}][Batch {batch_idx}] Applied memorization probe.")
        
        t = torch.randint(0, T, (data.size(0),), device=device).float()

        optimizer.zero_grad()
        out, risk, damping = model(data, t)
        loss = ((out - data)**2).mean()

        if variant == "no_dmre":
            if loss.item() < 0.5:
                loss = torch.tensor(0.0, requires_grad=True, device=device)
            print(f"[D-DAME (no_dmre)][Epoch {epoch_num}][Batch {batch_idx}] Loss: {loss.item():.4f}")
        else:
            modulated_loss = damping.mean() * loss
            loss = modulated_loss
            print(f"[D-DAME][Epoch {epoch_num}][Batch {batch_idx}] Raw Loss: {loss.item():.4f}, "
                  f"Damping Mean: {damping.mean().item():.4f}, Risk Mean: {risk.mean().item():.4f}")
        
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm)
        risk_values.append(risk.mean().item())
        damping_values.append(damping.mean().item())
        
        if writer is not None:
            global_step = epoch_num*max_iters + iter_count
            writer.add_scalar("D-DAME/loss", loss.item(), global_step)
            writer.add_scalar("D-DAME/grad_norm", grad_norm, global_step)
            writer.add_scalar("D-DAME/risk", risk.mean().item(), global_step)
            writer.add_scalar("D-DAME/damping", damping.mean().item(), global_step)
        
        iter_count += 1
    
    return {"loss": losses, "grad_norm": grad_norms, "risk": risk_values, "damping": damping_values}
