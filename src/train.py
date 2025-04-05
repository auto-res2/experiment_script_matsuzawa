import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np

from src.models.tcdp_models import TCDP, TCDP_NoConsistency, TCDP_FixedNoise, TCDP_Adaptive
from src.preprocess import get_testloader
from config.experiment_config import TRAINING_CONFIG

def train_tcdp_models(save_dir='./models', epochs=5, batch_size=32, learning_rate=0.001):
    """
    Train TCDP models on CIFAR-10 dataset
    
    This is a simplified training function for demonstration purposes.
    In a real-world scenario, you would use a proper training dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    testloader = get_testloader(batch_size=batch_size)
    
    tcdp = TCDP().to(device)
    tcdp_no_consistency = TCDP_NoConsistency().to(device)
    tcdp_fixed = TCDP_FixedNoise().to(device)
    tcdp_adaptive = TCDP_Adaptive().to(device)
    
    optimizer_tcdp = optim.Adam(tcdp.parameters(), lr=learning_rate)
    optimizer_no_cons = optim.Adam(tcdp_no_consistency.parameters(), lr=learning_rate)
    optimizer_fixed = optim.Adam(tcdp_fixed.parameters(), lr=learning_rate)
    optimizer_adaptive = optim.Adam(tcdp_adaptive.parameters(), lr=learning_rate)
    
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for i, (images, _) in enumerate(testloader):
            if i >= 1:  # Just one batch for quick testing
                break
                
            images = images.to(device)
            
            noisy_images = images + 0.1 * torch.randn_like(images)
            
            optimizer_tcdp.zero_grad()
            purified_tc, consistency_loss = tcdp(noisy_images)
            loss_tcdp = criterion(purified_tc, images) + 0.1 * consistency_loss
            loss_tcdp.backward()
            optimizer_tcdp.step()
            
            optimizer_no_cons.zero_grad()
            purified_no_cons = tcdp_no_consistency(noisy_images)
            loss_no_cons = criterion(purified_no_cons, images)
            loss_no_cons.backward()
            optimizer_no_cons.step()
            
            print(f"Batch {i+1}: TCDP Loss: {loss_tcdp.item():.4f}, No Consistency Loss: {loss_no_cons.item():.4f}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(tcdp.state_dict(), os.path.join(save_dir, 'tcdp.pth'))
    torch.save(tcdp_no_consistency.state_dict(), os.path.join(save_dir, 'tcdp_no_consistency.pth'))
    torch.save(tcdp_fixed.state_dict(), os.path.join(save_dir, 'tcdp_fixed.pth'))
    torch.save(tcdp_adaptive.state_dict(), os.path.join(save_dir, 'tcdp_adaptive.pth'))
    
    return tcdp, tcdp_no_consistency, tcdp_fixed, tcdp_adaptive
