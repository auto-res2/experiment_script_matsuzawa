"""
Evaluation module for ANCD experiments.
"""
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch_fidelity import calculate_metrics
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.experiment_utils import generate_samples, measure_consistency, save_bar_plot

def run_experiment2(model, train_loader, test_mode=False):
    """
    Run Experiment 2: Sample Quality and Stability Evaluation.
    
    Args:
        model: Trained model
        train_loader: DataLoader with training data
        test_mode: If True, use simplified evaluation
        
    Returns:
        results: Dictionary with experiment results
    """
    print("\n=== Running Experiment 2: Sample Quality and Stability Evaluation ===")
    
    samples = generate_samples(
        model, 
        train_loader, 
        n_batches=1 if test_mode else 3, 
        save_folder='./logs'
    )
    
    inputs, _ = next(iter(train_loader))
    consistency_loss = measure_consistency(model, inputs)
    print(f"Consistency Loss (Lower is better): {consistency_loss:.4f}")
    
    if test_mode:
        fid_score = 30.0 + np.random.rand() * 10
        is_score = 6.0 + np.random.rand() * 2
        fid_isc = {
            'frechet_inception_distance': fid_score,
            'inception_score_mean': is_score
        }
        print("Test Mode: Using mock FID and IS scores")
    else:
        fid_isc = calculate_metrics(
            input1='./logs/generated_samples.png',
            input2='cifar10', 
            isc=True, 
            fid=True
        )
        
    print("FID: ", fid_isc.get('frechet_inception_distance', None))
    print("IS Mean: ", fid_isc.get('inception_score_mean', None))
    
    metrics_names = ['FID', 'IS']
    metrics_values = [
        fid_isc.get('frechet_inception_distance', 0),
        fid_isc.get('inception_score_mean', 0)
    ]
    
    save_bar_plot(
        metrics_names, 
        metrics_values, 
        "", 
        "Metric Value", 
        "Sample Quality Metrics", 
        "logs/sample_quality_exp2.pdf"
    )
    
    print("Experiment 2 plot saved as: logs/sample_quality_exp2.pdf")
    
    return {
        "consistency_loss": consistency_loss, 
        "metrics": fid_isc
    }

def run_experiment3(variant_losses):
    """
    Complete Experiment 3: Ablation Study by plotting variant loss curves.
    
    Args:
        variant_losses: Dictionary with loss values for each variant
        
    Returns:
        None
    """
    print("\n=== Completing Experiment 3: Ablation Study ===")
    
    plt.figure(figsize=(8,6))
    epochs = np.arange(1, len(next(iter(variant_losses.values()))) + 1)
    
    for variant_name, losses in variant_losses.items():
        plt.plot(epochs, losses, marker='o', label=variant_name)
        
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Ablation Study: Training Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/training_loss_ablation_exp3.pdf")
    
    print("Experiment 3 plot saved as: logs/training_loss_ablation_exp3.pdf")
