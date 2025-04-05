"""
Main entry point for TEDP (Trigger-Eradicating Diffusion Purification) experiment.
Implements the full pipeline from data preprocessing to evaluation.
"""

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

from preprocess import SyntheticDataset, PurificationModule
from train import DiffusionModel, train_pipeline
from evaluate import experiment1, experiment2


def experiment3(num_epochs=5, device='cuda'):
    """
    Run an ablation study that compares two training variants:
        a. Fixed purification strength.
        b. Adaptive purification strength (updated batch-by-batch).
    We record the evolution of the purification strength (for adaptive) and a simple loss metric.
    """
    print("Starting Experiment 3: Adaptive Tuning and Ablation Study")
    dataset = SyntheticDataset(size=500, poison_ratio=0.05)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model_fixed = DiffusionModel()
    model_adaptive = DiffusionModel()
    
    purification_fixed = PurificationModule(noise_std_init=0.05, variance_explosion=1.1, step_size=0.1)
    purification_adaptive = PurificationModule(noise_std_init=0.05, variance_explosion=1.1, step_size=0.1)
    
    fixed_loss_history = []
    adaptive_loss_history = []
    adaptive_strength_history = []  # record purification strength per epoch
    
    def record_fixed(epoch, loss, _):
        fixed_loss_history.append(loss)
        print("Fixed Epoch {}: Loss {:.4f}".format(epoch, loss))
    
    def record_adaptive(epoch, loss, curr_strength):
        adaptive_loss_history.append(loss)
        adaptive_strength_history.append(curr_strength)
        print("Adaptive Epoch {}: Loss {:.4f} - Purification Strength {:.4f}".format(epoch, loss, curr_strength))
    
    print("Training with fixed purification strength...")
    train_pipeline(model_fixed, loader, purification=purification_fixed, num_epochs=num_epochs, 
                   adaptive=False, record_func=record_fixed, device=device)
    
    print("Training with adaptive purification strength...")
    train_pipeline(model_adaptive, loader, purification=purification_adaptive, num_epochs=num_epochs, 
                   adaptive=True, record_func=record_adaptive, device=device)
    
    epochs = list(range(num_epochs))
    plt.figure(figsize=(6,4))
    plt.plot(epochs, fixed_loss_history, label="Fixed")
    plt.plot(epochs, adaptive_loss_history, label="Adaptive", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss: Fixed vs Adaptive")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/training_loss_comparison.pdf")
    print("Saved plot: logs/training_loss_comparison.pdf")
    
    plt.figure(figsize=(6,4))
    plt.plot(epochs, adaptive_strength_history, marker='o', color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("Purification Strength")
    plt.title("Adaptive Purification Strength Evolution")
    plt.tight_layout()
    plt.savefig("logs/purification_strength_adaptive.pdf")
    print("Saved plot: logs/purification_strength_adaptive.pdf")
    
    print("Experiment 3 finished.\n")


def test(device='cuda'):
    """
    The test function will run the three experiments quickly.
    This is intended to verify that the code executes without errors.
    """
    print("\n" + "="*80)
    print("TEDP TEST MODE - Running Quick Verification Tests")
    print("="*80)
    
    dataset_size = 100
    poison_ratio = 0.05
    batch_size = 16
    
    print("\nTest Configuration:")
    print("-"*50)
    print(f"Dataset: {dataset_size} samples, {poison_ratio:.2%} poison ratio")
    print(f"Device: {device}")
    print(f"Running with reduced epochs for quick verification")
    print("-"*50)
    
    print("\nInitializing test dataset...")
    dataset = SyntheticDataset(size=dataset_size, poison_ratio=poison_ratio)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    poisoned_count = sum(label for _, label in dataset)
    print(f"Test dataset created with {dataset_size} samples ({poisoned_count} poisoned, {dataset_size-poisoned_count} clean)")
    
    print("\nInitializing models for testing...")
    model_tedp = DiffusionModel()
    model_baseline = DiffusionModel()
    purification_module = PurificationModule(noise_std_init=0.05, variance_explosion=1.1, step_size=0.1)
    
    print("\n" + "-"*80)
    print("TEST 1: Advanced Diffusion Purification Benchmark (2 epochs)")
    print("-"*80)
    experiment1(model_tedp, model_baseline, purification_module, loader, num_epochs=2, device=device)
    
    print("\n" + "-"*80)
    print("TEST 2: Latent-Space Regularization Analysis (1 epoch)")
    print("-"*80)
    experiment2(model_tedp, loader, num_epochs=1, device=device)
    
    print("\n" + "-"*80)
    print("TEST 3: Adaptive Tuning and Ablation Study (2 epochs)")
    print("-"*80)
    experiment3(num_epochs=2, device=device)
    
    print("\n" + "="*80)
    print("TEST SUMMARY: All tests completed successfully")
    print("="*80)


def run_full_experiments(device='cuda'):
    """
    Run all three experiments with full settings.
    """
    print("\n" + "="*80)
    print("TEDP (Trigger-Eradicating Diffusion Purification) Experiment Suite")
    print("="*80)
    
    dataset_size = 500
    poison_ratio = 0.05
    batch_size = 16
    
    noise_std_init = 0.05
    variance_explosion = 1.1
    step_size = 0.1
    
    exp1_epochs = 5
    exp2_epochs = 3
    exp3_epochs = 5
    
    print("\nExperiment Configuration:")
    print("-"*50)
    print(f"Dataset Parameters:")
    print(f"  - Size: {dataset_size} samples")
    print(f"  - Poison ratio: {poison_ratio:.2%}")
    print(f"  - Batch size: {batch_size}")
    print(f"\nPurification Parameters:")
    print(f"  - Initial noise std: {noise_std_init}")
    print(f"  - Variance explosion: {variance_explosion}")
    print(f"  - Step size: {step_size}")
    print(f"\nTraining Parameters:")
    print(f"  - Experiment 1: {exp1_epochs} epochs")
    print(f"  - Experiment 2: {exp2_epochs} epochs")
    print(f"  - Experiment 3: {exp3_epochs} epochs")
    print(f"  - Device: {device}")
    print("-"*50)
    
    print("\nInitializing dataset and models...")
    dataset = SyntheticDataset(size=dataset_size, poison_ratio=poison_ratio)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    poisoned_count = sum(label for _, label in dataset)
    print(f"Dataset created with {dataset_size} samples ({poisoned_count} poisoned, {dataset_size-poisoned_count} clean)")
    
    print("\nInitializing models...")
    model_tedp = DiffusionModel()
    model_baseline = DiffusionModel()
    purification_module = PurificationModule(noise_std_init=noise_std_init, 
                                            variance_explosion=variance_explosion, 
                                            step_size=step_size)
    
    print("\nModel architecture summary:")
    print(f"  - Input channels: 3")
    print(f"  - Latent dimensions: 32 channels")
    print(f"  - Total parameters: {sum(p.numel() for p in model_tedp.parameters()):,}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 1: Advanced Diffusion Purification Benchmark")
    print("="*80)
    print("This experiment compares TEDP purification with baseline (no purification)")
    print("Metrics: FID Score, PSNR, Trigger Detection Rate")
    print("-"*80)
    
    experiment1(model_tedp, model_baseline, purification_module, loader, num_epochs=exp1_epochs, device=device)
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: Latent-Space Regularization Analysis")
    print("="*80)
    print("This experiment analyzes the latent space to detect clustering of poisoned samples")
    print("Metrics: Silhouette Score, PCA Visualization")
    print("-"*80)
    
    model_for_exp2 = DiffusionModel()
    experiment2(model_for_exp2, loader, num_epochs=exp2_epochs, device=device)
    
    print("\n" + "="*80)
    print("EXPERIMENT 3: Adaptive Tuning and Ablation Study")
    print("="*80)
    print("This experiment compares fixed vs. adaptive purification strength")
    print("Metrics: Training Loss, Purification Strength Evolution")
    print("-"*80)
    
    experiment3(num_epochs=exp3_epochs, device=device)
    
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print("All experiments completed successfully.")
    print("Generated plots:")
    print("  - logs/FID_comparison_pair1.pdf")
    print("  - logs/PSNR_comparison_pair1.pdf")
    print("  - logs/TriggerDetection_comparison_pair1.pdf")
    print("  - logs/latent_space_projection.pdf")
    print("  - logs/training_loss_comparison.pdf")
    print("  - logs/purification_strength_adaptive.pdf")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEDP Experiment Runner')
    parser.add_argument('--test', action='store_true', help='Run a quick test of all experiments')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available() and device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    start_time = time.time()
    
    if args.test:
        test(device)
    else:
        run_full_experiments(device)
    
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
