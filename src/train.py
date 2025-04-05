"""
Script for training models.

This module implements the ASID-M model architecture and training procedures.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
    
    def generate_trajectory(self, x, num_steps=20):
        """
        Simulates an iterative refinement (denoising) process.
        
        Args:
            x (torch.Tensor): Input tensor.
            num_steps (int): Number of refinement steps.
            
        Returns:
            torch.Tensor: Refined tensor.
        """
        traj = x.clone()
        for step in range(num_steps):
            traj = traj - 0.05 * traj   # iterative denoising simulation
        return traj

class MomentumModule(nn.Module):
    def __init__(self, dim, momentum=0.9):
        super(MomentumModule, self).__init__()
        self.momentum = momentum
        self.register_buffer('historical', torch.zeros(dim))
    
    def forward(self, x):
        """
        Applies momentum correction to the input.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Momentum-corrected tensor.
        """
        current = x.mean(dim=0, keepdim=True)
        self.historical = self.momentum * self.historical + (1 - self.momentum) * current
        return x + self.historical

class OneStepStudent(nn.Module):
    def __init__(self, input_dim=3*32*32, hidden_dim=512):
        super(OneStepStudent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.momentum_module = MomentumModule(hidden_dim)
        
    def forward(self, x):
        """
        Forward pass through the one-step student model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        x_flat = x.view(x.size(0), -1)
        x_flat = self.relu(self.fc1(x_flat))
        x_flat = self.momentum_module(x_flat)
        x_flat = self.fc2(x_flat)
        return x_flat.view(-1, 3, 32, 32)

class SimpleOneStepStudent(nn.Module):
    def __init__(self, input_dim=3*32*32, hidden_dim=512):
        super(SimpleOneStepStudent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        """
        Forward pass through the simple one-step student model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        x_flat = x.view(x.size(0), -1)
        x_flat = self.relu(self.fc1(x_flat))
        x_flat = self.fc2(x_flat)
        return x_flat.view(-1, 3, 32, 32)

def train_model(model, dataloader, teacher_model, criterion, optimizer, 
               device, epochs=2, noise_std=0.1, max_batches=None):
    """
    Trains a student model to mimic a teacher model.
    
    Args:
        model (nn.Module): Student model to train.
        dataloader (DataLoader): DataLoader containing training data.
        teacher_model (nn.Module): Teacher model to mimic.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for training.
        device (torch.device): Device to train on.
        epochs (int): Number of epochs to train for.
        noise_std (float): Standard deviation of noise to add to images.
        max_batches (int): Maximum number of batches to use per epoch.
        
    Returns:
        list: Training loss history.
    """
    loss_history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            images = images.to(device)
            noisy_images = images + torch.randn_like(images) * noise_std
            
            with torch.no_grad():
                teacher_target = teacher_model.generate_trajectory(images, num_steps=20)
            
            student_output = model(noisy_images)
            loss = criterion(student_output, teacher_target)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / batch_count if batch_count > 0 else 0
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} finished with average loss {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")
    
    return loss_history

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_data = torch.randn(16, 3, 32, 32).to(device)
    dummy_dataloader = [(input_data, torch.zeros(16)) for _ in range(2)]
    
    teacher_model = TeacherModel().to(device)
    student_model = OneStepStudent().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    
    loss_history = train_model(
        student_model, dummy_dataloader, teacher_model, criterion, optimizer, 
        device, epochs=1, noise_std=0.1
    )
    
    print("Test training completed successfully")
