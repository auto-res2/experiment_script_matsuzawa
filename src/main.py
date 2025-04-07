"""
Main script for RobustPurify-Backdoor Diffusion experiment.

This script orchestrates the entire experiment by:
1. Setting up the environment
2. Creating the logs directory if it doesn't exist
3. Running all three experiments:
   - Dual-Path Signal Embedding and Trigger Robustness
   - Efficacy of Dual-Loss Training
   - Impact of Poisoning Ratio on Backdoor Efficacy

All figures and plots are saved in high-quality PDF format in the logs directory.
"""

import os
import numpy as np
import torch
import random
import time

from evaluate import (
    experiment_dual_path_signal_embedding,
    experiment_dual_loss_training,
    experiment_poisoning_ratio_effect
)


def setup_environment():
    """Set up the environment for the experiment."""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("logs/experiment3"):
        os.makedirs("logs/experiment3")
        
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU.")


if __name__ == "__main__":
    start_time = time.time()
    print("Starting RobustPurify-Backdoor Diffusion Experiment")
    
    setup_environment()
    
    experiment_dual_path_signal_embedding()
    experiment_dual_loss_training()
    experiment_poisoning_ratio_effect()
    
    print(f"\nExperiment completed in {time.time() - start_time:.2f} seconds")
