"""
Training script for video super-resolution models.
"""

import torch
import yaml
from utils.models import StableVSR, ATRD, ATRD_NoOTAR

def train_model(model_name, dataloader, config_path):
    """
    Train a video super-resolution model.
    
    Args:
        model_name: Name of the model to train
        dataloader: Dataloader for training
        config_path: Path to configuration file
        
    Returns:
        model: Trained model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_params = config['model']
    
    if model_name == 'StableVSR':
        model = StableVSR()
    elif model_name == 'ATRD':
        model = ATRD(diffusion_steps=model_params['diffusion_steps'])
    elif model_name == 'ATRD_NoOTAR':
        model = ATRD_NoOTAR(diffusion_steps=model_params['diffusion_steps'])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    
    return model
