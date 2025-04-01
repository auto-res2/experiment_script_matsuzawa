import torch
import torchvision.transforms as transforms
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def setup_transforms():
    """
    Set up image transformations for the model.
    
    Returns:
        torchvision.transforms.Compose: Composition of image transformations
    """
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return image_transform

def generate_synthetic_data(num_samples=100, text_length=10, test_mode=False):
    """
    Generate synthetic data for testing the model.
    
    Args:
        num_samples: Number of samples to generate
        text_length: Length of text sequences
        test_mode: If True, generate a smaller dataset for testing
        
    Returns:
        dict: Dictionary containing synthetic text and image data
    """
    print("Generating synthetic data...")
    
    if test_mode:
        num_samples = 10
        
    image_data = torch.rand(num_samples, 3, 224, 224)
    
    text_data = torch.randint(0, 1000, (num_samples, text_length))
    
    labels = torch.randint(0, 5, (num_samples,))
    
    print(f"Generated {num_samples} synthetic samples")
    
    return {
        'images': image_data,
        'text_ids': text_data,
        'labels': labels
    }

def prepare_data(config, test_mode=False):
    """
    Prepare data for training and evaluation.
    
    Args:
        config: Configuration dictionary
        test_mode: If True, use minimal data for testing
        
    Returns:
        dict: Dictionary containing the prepared data
    """
    data = generate_synthetic_data(
        num_samples=config['data']['num_samples'],
        text_length=config['data']['text_length'],
        test_mode=test_mode
    )
    
    num_samples = len(data['images'])
    indices = np.random.permutation(num_samples)
    
    if test_mode:
        train_idx = indices[:6]
        val_idx = indices[6:8]
        test_idx = indices[8:]
    else:
        train_idx = indices[:int(0.7*num_samples)]
        val_idx = indices[int(0.7*num_samples):int(0.9*num_samples)]
        test_idx = indices[int(0.9*num_samples):]
        
    train_data = {
        'images': data['images'][train_idx],
        'text_ids': data['text_ids'][train_idx],
        'labels': data['labels'][train_idx]
    }
    
    val_data = {
        'images': data['images'][val_idx],
        'text_ids': data['text_ids'][val_idx],
        'labels': data['labels'][val_idx]
    }
    
    test_data = {
        'images': data['images'][test_idx],
        'text_ids': data['text_ids'][test_idx],
        'labels': data['labels'][test_idx]
    }
    
    print(f"Data split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

if __name__ == "__main__":
    transform = setup_transforms()
    test_data = generate_synthetic_data(test_mode=True)
    print("Preprocessing module test successful!")
