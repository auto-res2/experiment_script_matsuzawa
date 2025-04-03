"""
Main script for running Joint-Guided Bayesian Flow Networks (JG-BFN) experiments.
"""
import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

from preprocess import load_mnist_data, generate_synthetic_data
from train import (ScoreModel, train_ablation_experiment, 
                  train_adaptive_experiment, compute_local_density)
from evaluate import (sample_with_solver, compute_dummy_fid, 
                    run_efficiency_experiment, plot_loss_curve, 
                    plot_loss_comparison, plot_sampling_tradeoff)

def setup_directories():
    """Create the necessary directories for experiment outputs"""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

def run_experiment1_ablation(num_epochs=3, batch_size=64, weight_guidance=1.0, device=None):
    """
    Run Experiment 1: Ablation Study on Loss Components.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        weight_guidance: Weight for guidance loss
        device: Device to run the experiment on
        
    Returns:
        model_full: Trained full loss model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nRunning Experiment 1: Ablation Study on Loss Components")
    
    train_loader, _ = load_mnist_data(root='./data', batch_size=batch_size, subset_size=1024)
    
    model_full, model_ablation, loss_history_full, loss_history_ablation = train_ablation_experiment(
        train_loader, num_epochs, batch_size, weight_guidance, device
    )
    
    plot_loss_comparison(
        loss_history_full, 
        loss_history_ablation,
        labels=["Full Loss Model", "Ablation Model"],
        filename="logs/training_loss_baseline_pair1.pdf",
        title="Experiment 1: Training Loss Comparison"
    )
    
    torch.save(model_full.state_dict(), "models/model_full.pt")
    torch.save(model_ablation.state_dict(), "models/model_ablation.pt")
    
    return model_full

def run_experiment2_adaptive(num_epochs=3, batch_size=64, fixed_weight=1.0, device=None):
    """
    Run Experiment 2: Adaptive Weighting and Noise Schedule Robustness.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        fixed_weight: Fixed weight for guidance loss
        device: Device to run the experiment on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nRunning Experiment 2: Adaptive Weighting and Noise Schedule Robustness")
    
    dataloader = generate_synthetic_data(n_samples=1024, noise=0.1, batch_size=batch_size)
    
    model_fixed, model_adapt, loss_history_fixed, loss_history_adapt = train_adaptive_experiment(
        dataloader, num_epochs, batch_size, fixed_weight, device
    )
    
    plot_loss_comparison(
        loss_history_fixed,
        loss_history_adapt,
        labels=["Fixed Weight", "Adaptive Weight"],
        filename="logs/training_loss_adaptive_pair1.pdf",
        title="Experiment 2: Fixed vs Adaptive Guidance Weight"
    )
    
    torch.save(model_fixed.state_dict(), "models/model_fixed.pt")
    torch.save(model_adapt.state_dict(), "models/model_adaptive.pt")

def run_experiment3_efficiency(model, real_data, steps_list=[10, 20, 50, 100], device=None):
    """
    Run Experiment 3: Efficiency and Fast Sampling Analysis.
    
    Args:
        model: Trained model to evaluate
        real_data: Real data for FID computation
        steps_list: List of discretization steps to evaluate
        device: Device to run the experiment on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sampling_results, fid_list, time_list = run_efficiency_experiment(
        model, real_data, steps_list, device
    )
    
    plot_sampling_tradeoff(
        steps_list, fid_list, time_list, 
        filename="logs/inference_latency_pair1.pdf"
    )

def test_code(device=None):
    """
    Run a quick test of the experiments to ensure that the code executes.
    The test runs minimal epochs/steps to finish quickly.
    
    Args:
        device: Device to run the test on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nStarting test run on device: {device}")
    
    model_full = run_experiment1_ablation(num_epochs=1, batch_size=128, weight_guidance=0.5, device=device)
    
    run_experiment2_adaptive(num_epochs=1, batch_size=128, fixed_weight=0.5, device=device)
    
    _, test_loader = load_mnist_data(root='./data', batch_size=128, subset_size=256)
    real_data = next(iter(test_loader))[0]
    run_experiment3_efficiency(model_full, real_data, steps_list=[10, 20], device=device)
    
    print("\nTest run completed successfully.")

def main():
    """Main function to run the JG-BFN experiments"""
    parser = argparse.ArgumentParser(description='Run JG-BFN experiments')
    parser.add_argument('--test', action='store_true', help='Run a quick test of the code')
    parser.add_argument('--device', type=str, default=None, help='Device to run on (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    args = parser.parse_args()
    
    setup_directories()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device is not None:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    if args.test:
        test_code(device)
    else:
        print("\nRunning full JG-BFN experiments")
        
        model_full = run_experiment1_ablation(
            num_epochs=args.epochs, 
            batch_size=args.batch_size, 
            weight_guidance=1.0, 
            device=device
        )
        
        run_experiment2_adaptive(
            num_epochs=args.epochs, 
            batch_size=args.batch_size, 
            fixed_weight=1.0, 
            device=device
        )
        
        _, test_loader = load_mnist_data(root='./data', batch_size=args.batch_size)
        real_data = next(iter(test_loader))[0]
        run_experiment3_efficiency(
            model_full, 
            real_data, 
            steps_list=[10, 20, 50, 100], 
            device=device
        )
        
        print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()
