"""
Main module for the Optimized Characteristic Resampling (OCR) experiment.

This script runs the entire OCR experiment workflow, from data preparation to model
training and evaluation.
"""

import os
import torch
import matplotlib.pyplot as plt
import yaml
import json
import argparse
from datetime import datetime

from preprocess import prepare_data
from train import DummyDDPM
from evaluate import run_experiment1, run_experiment2, run_experiment3

def load_config(config_path=None):
    """
    Load configuration from a YAML file or use default configuration.
    
    Args:
        config_path (str, optional): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    default_config = {
        "batch_size": 16,
        "image_size": (3, 32, 32),
        "dataset_size": 64,
        "guidance_scales": [5, 10, 50],
        "max_iterations": 50,
        "loss_threshold": 1e-3,
        "learning_rate": 1e-2,
        "optimizers": ["adam", "rmsprop", "anderson"],
        "plot_path": "plots",
        "log_path": "logs"
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
    else:
        config = default_config
        os.makedirs("config", exist_ok=True)
        with open("config/default_config.yaml", 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    return config

def main():
    """
    Main function to run the OCR experiment.
    """
    parser = argparse.ArgumentParser(description="Run OCR experiment")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    config = load_config(args.config)
    
    print("Running OCR experiment with configuration:")
    print(json.dumps(config, indent=2))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    dataloader = prepare_data(config)
    print("Data prepared successfully")
    
    model = DummyDDPM().to(device)
    print("Model initialized successfully")
    
    print("\nStarting experiments...\n")
    
    results1 = run_experiment1(model, dataloader, device, save_path=config["plot_path"])
    
    results2 = run_experiment2(model, device, save_path=config["plot_path"])
    
    results3 = run_experiment3(model, device, save_path=config["plot_path"])
    
    print("\nAll experiments completed successfully!")
    print(f"Results saved to {config['plot_path']} directory")

if __name__ == "__main__":
    main()
