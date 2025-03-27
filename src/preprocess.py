"""
Data preprocessing for video super-resolution.
"""

import yaml
from utils.data import get_dummy_dataloader

def preprocess_data(config_path):
    """
    Preprocess data for video super-resolution.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dataloader: Dataloader for training and evaluation
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_params = config['data']
    
    dataloader = get_dummy_dataloader(
        num_sequences=data_params['num_sequences'],
        num_frames=data_params['num_frames'],
        channels=data_params['channels'],
        height=data_params['height'],
        width=data_params['width']
    )
    
    return dataloader
