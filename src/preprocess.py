import torch
import numpy as np

def load_data(batch_size=2, image_size=16, feature_dim=32, in_channels=3):
    """
    Generate synthetic data for AFiT experiments.
    
    Args:
        batch_size (int): Number of samples in each batch
        image_size (int): Size of synthetic images (image_size x image_size)
        feature_dim (int): Dimension of features
        in_channels (int): Number of input channels
        
    Returns:
        images: Synthetic images
        ground_truth_tokens: Synthetic ground truth tokens
    """
    
    images = torch.randn(batch_size, in_channels, image_size, image_size)
    
    seq_length = image_size * image_size  # flattened spatial dimensions
    ground_truth_tokens = torch.randn(seq_length, batch_size, feature_dim)
    
    return images, ground_truth_tokens

def preprocess_data(config):
    """
    Preprocess data based on configuration parameters.
    
    Args:
        config: Configuration parameters
        
    Returns:
        Preprocessed data for experiments
    """
    images, ground_truth_tokens = load_data(
        batch_size=config.batch_size,
        image_size=config.image_size,
        feature_dim=config.feature_dim,
        in_channels=config.in_channels
    )
    
    return {
        'images': images,
        'ground_truth_tokens': ground_truth_tokens
    }
