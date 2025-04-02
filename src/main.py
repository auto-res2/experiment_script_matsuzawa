import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
from types import SimpleNamespace

from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model

def load_config(config_path=None):
    """
    Load configuration from a file or use default values.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        config: Configuration parameters
    """
    default_config = {
        'batch_size': 2,
        'image_size': 16,  # Use a small image size for demonstration
        'in_channels': 3,
        'feature_dim': 32,
        
        'seq_length': 10,  # Number of tokens in a sequence
        'momentum': 0.9,  # Momentum for token acceleration
        
        'num_epochs': 5,  # For a quick test
        'learning_rate': 1e-3,
        'acceleration_loss_weight': 1.0,  # Weight for secondary loss
        
        'num_iterations': 5,  # For token acceleration evaluation
        'num_steps': 10,  # For ODE solver evaluation
        'step_size': 0.1,  # Step size for diffusion process
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    config = SimpleNamespace(**default_config)
    
    if config_path and os.path.exists(config_path):
        pass
    
    return config

def setup_environment():
    """
    Set up the environment for the experiment.
    
    Returns:
        config: Configuration parameters
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU instead.")
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    config = load_config()
    config.device = str(device)
    
    return config

def run_experiment():
    """
    Run the complete AFiT experiment.
    """
    print("Starting AFiT Experiment")
    print("=" * 50)
    
    config = setup_environment()
    print("Configuration loaded.")
    
    start_time = time.time()
    
    print("\nStep 1: Preprocessing data...")
    data = preprocess_data(config)
    print("Data preprocessing complete.")
    
    print("\nStep 2: Training the model...")
    training_results = train_model(config, data)
    print("Model training complete.")
    
    print("\nStep 3: Evaluating the model...")
    evaluation_results = evaluate_model(config, training_results)
    print("Model evaluation complete.")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\nExperiment Summary:")
    print("=" * 50)
    print(f"Total experiment time: {total_time:.2f} seconds")
    print(f"Device used: {config.device}")
    print(f"Final training loss: {training_results['loss_history'][-1]:.4f}")
    print("Saved plots:")
    print("- logs/training_loss_joint_full.pdf")
    print("- logs/training_loss_dynamicToken_small.pdf")
    print("- logs/fid_vs_timesteps_highOrder_small.pdf")
    print("=" * 50)
    
    return {
        'config': config,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'total_time': total_time
    }

if __name__ == '__main__':
    run_experiment()
