"""
Utility functions for models and training.
"""
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def add_memorization_probe(data, probe_intensity=0.1, probe_ratio=0.2):
    """Add perturbations to a fraction of data (simulate memorization probes)."""
    batch_size = data.size(0)
    num_probes = max(1, int(probe_ratio * batch_size))
    indices = torch.randperm(batch_size)[:num_probes]
    probe = probe_intensity * torch.randn_like(data[indices])
    data[indices] += probe
    return data

def plot_metrics(metric_dict, filename_prefix="result"):
    """
    Plot training metrics.
    
    Args:
        metric_dict: Dictionary with keys (e.g., "loss", "grad_norm", "risk") 
                  and each value a list of metric values per training iteration.
        filename_prefix: Prefix for the output file name.
    """
    plt.figure(figsize=(10, 6))
    for key, values in metric_dict.items():
        plt.plot(values, label=key)
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    filename = f"{filename_prefix}_metrics.pdf"
    plt.savefig(filename, format='pdf')
    plt.close()
    print("Plot saved to", filename)
