import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import time
import os
from utils.diffusion import diffusion_step, double_tweedie_transform, calculate_adaptive_lambda
from utils.consistency import ConsistencyLoss, AdaptiveConsistency

def load_model(model_name='resnet18', num_classes=10, pretrained=True):
    """Load a model architecture.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        model: PyTorch model
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    return model

def save_model(model, filename='models/model.pth'):
    """Save a trained model.
    
    Args:
        model: PyTorch model
        filename: Path to save the model
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, 
               device='cuda:0', save_path='models/model.pth'):
    """Train a classifier model.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation/test data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save the best model
        
    Returns:
        model: Trained model
        history: Training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, save_path)
    
    model.load_state_dict(torch.load(save_path))
    
    return model, history

class SimpleScoreModel(nn.Module):
    """A simple score model for the Tweedie estimator."""
    def __init__(self, input_channels=3):
        super(SimpleScoreModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, noise_level):
        """Forward pass.
        
        Args:
            x: Input tensor
            noise_level: Noise level for conditioning
            
        Returns:
            Score estimate
        """
        x = x * (1.0 + noise_level)
        return self.net(x)

def train_score_model(score_model, train_loader, epochs=5, lr=0.0001, 
                     device='cuda:0', save_path='models/score_model.pth'):
    """Train a score model for the Tweedie estimator.
    
    Args:
        score_model: Score model
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save the model
        
    Returns:
        score_model: Trained score model
    """
    score_model = score_model.to(device)
    optimizer = optim.Adam(score_model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            noise_level = torch.rand(batch_size, 1, 1, 1, device=device) * 0.1
            noise = torch.randn_like(inputs) * noise_level
            noisy_inputs = inputs + noise
            
            optimizer.zero_grad()
            predicted_noise = score_model(noisy_inputs, noise_level.view(-1, 1, 1, 1))
            loss = nn.MSELoss()(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}: score_model_loss={avg_loss:.6f}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(score_model.state_dict(), save_path)
    
    return score_model
