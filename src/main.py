#!/usr/bin/env python3
"""
Main script for running Iso-LWGAN experiments.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from preprocess import prepare_dataset
from train import (
    train_iso_lwgan, 
    train_stochastic_generator, 
    train_comparison_model
)
from evaluate import (
    evaluate_isometric_regularizer,
    evaluate_stochastic_generator,
    evaluate_mnist_models
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_params import (
    EXP1_PARAMS, 
    EXP2_PARAMS, 
    EXP3_PARAMS, 
    TEST_PARAMS,
    RANDOM_SEED,
    DEVICE,
    STATUS_ENUM
)

def setup_environment():
    """Set up the environment for experiments."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    return device

def experiment1_isometric_regularization(device, test_mode=False):
    """
    Experiment 1: Impact of the Isometric Regularization Term
    """
    print("\n" + "="*80)
    print("Experiment 1: Impact of the Isometric Regularization Term")
    print("="*80)
    
    params = TEST_PARAMS if test_mode else EXP1_PARAMS
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"logs/exp1_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    data_loader = prepare_dataset('synthetic', params)
    
    lambda_values = params['test_lambda_iso']
    results = {}
    
    for lambda_iso in lambda_values:
        loss_history, encoder, generator = train_iso_lwgan(
            lambda_iso=lambda_iso,
            num_epochs=params['num_epochs'],
            batch_size=params['batch_size'],
            z_dim=params['z_dim'],
            data_loader=data_loader,
            device=device,
            log_dir=f"{output_dir}/lambda_{lambda_iso}",
            save_dir=output_dir
        )
        
        avg_diff, max_diff = evaluate_isometric_regularizer(
            encoder, generator, data_loader, save_dir=output_dir
        )
        
        results[lambda_iso] = {
            'loss_history': loss_history,
            'avg_diff': avg_diff,
            'max_diff': max_diff
        }
        
        torch.save({
            'encoder': encoder.state_dict(),
            'generator': generator.state_dict(),
            'lambda_iso': lambda_iso,
            'params': params
        }, f"models/iso_lwgan_lambda{lambda_iso}.pth")
    
    plt.figure(figsize=(10, 6))
    for lambda_iso, result in results.items():
        plt.plot(result['loss_history'], label=f"λ_iso = {lambda_iso}")
    plt.title("Training Loss for Different Isometric Regularization Weights")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/lambda_comparison.pdf", bbox_inches="tight", format="pdf")
    plt.close()
    
    lambda_values_list = list(results.keys())
    avg_diffs = [results[lam]['avg_diff'] for lam in lambda_values_list]
    
    plt.figure(figsize=(8, 5))
    plt.bar([str(lam) for lam in lambda_values_list], avg_diffs)
    plt.title("Average Distance Difference for Different λ_iso Values")
    plt.xlabel("λ_iso")
    plt.ylabel("Average Distance Difference")
    plt.savefig(f"{output_dir}/avg_diff_comparison.pdf", bbox_inches="tight", format="pdf")
    plt.close()
    
    print(f"Experiment 1 completed. Results saved to {output_dir}")
    return results

def experiment2_stochastic_generator(device, test_mode=False):
    """
    Experiment 2: Evaluating the Partially Stochastic Generator
    """
    print("\n" + "="*80)
    print("Experiment 2: Evaluating the Partially Stochastic Generator")
    print("="*80)
    
    params = TEST_PARAMS if test_mode else EXP2_PARAMS
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"logs/exp2_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    data_loader = prepare_dataset('multimodal', params)
    
    sigma_values = params['test_sigma_noise']
    results = {}
    
    for sigma_noise in sigma_values:
        loss_history, encoder, generator = train_stochastic_generator(
            sigma_noise=sigma_noise,
            num_epochs=params['num_epochs'],
            batch_size=params['batch_size'],
            z_dim=params['z_dim'],
            data_loader=data_loader,
            device=device,
            log_dir=f"{output_dir}/sigma_{sigma_noise}",
            save_dir=output_dir
        )
        
        diversity_metrics = evaluate_stochastic_generator(
            encoder, generator, data_loader, 
            sigma_noise=sigma_noise, 
            save_dir=output_dir
        )
        
        results[sigma_noise] = {
            'loss_history': loss_history,
            'diversity_metrics': diversity_metrics,
            'avg_diversity': np.mean(diversity_metrics)
        }
        
        torch.save({
            'encoder': encoder.state_dict(),
            'generator': generator.state_dict(),
            'sigma_noise': sigma_noise,
            'params': params
        }, f"models/stochastic_generator_sigma{sigma_noise}.pth")
    
    plt.figure(figsize=(10, 6))
    for sigma_noise, result in results.items():
        plt.plot(result['loss_history'], label=f"σ_noise = {sigma_noise}")
    plt.title("Training Loss for Different Noise Magnitudes")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/sigma_loss_comparison.pdf", bbox_inches="tight", format="pdf")
    plt.close()
    
    sigma_values_list = list(results.keys())
    avg_diversities = [results[sigma]['avg_diversity'] for sigma in sigma_values_list]
    
    plt.figure(figsize=(8, 5))
    plt.bar([str(sigma) for sigma in sigma_values_list], avg_diversities)
    plt.title("Average Output Diversity for Different σ_noise Values")
    plt.xlabel("σ_noise")
    plt.ylabel("Average Diversity")
    plt.savefig(f"{output_dir}/diversity_comparison.pdf", bbox_inches="tight", format="pdf")
    plt.close()
    
    print(f"Experiment 2 completed. Results saved to {output_dir}")
    return results

def experiment3_mnist_comparison(device, test_mode=False):
    """
    Experiment 3: Overall Performance Comparison between Iso-LWGAN and Base LWGAN
    """
    print("\n" + "="*80)
    print("Experiment 3: Overall Performance Comparison between Iso-LWGAN and Base LWGAN")
    print("="*80)
    
    params = TEST_PARAMS if test_mode else EXP3_PARAMS
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"logs/exp3_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    data_loader = prepare_dataset('mnist', params)
    
    print("\nTraining Base LWGAN model...")
    base_encoder, base_generator = train_comparison_model(
        model_type="Base",
        num_epochs=params['num_epochs'],
        z_dim=params['z_dim'],
        lambda_iso=0.0,  # No isometric regularization for base model
        sigma_noise=0.0,  # No noise injection for base model
        data_loader=data_loader,
        device=device,
        log_dir=output_dir,
        save_dir=output_dir
    )
    
    print("\nTraining Iso-LWGAN model...")
    iso_encoder, iso_generator = train_comparison_model(
        model_type="Iso",
        num_epochs=params['num_epochs'],
        z_dim=params['z_dim'],
        lambda_iso=params['lambda_iso'],
        sigma_noise=params['sigma_noise'],
        data_loader=data_loader,
        device=device,
        log_dir=output_dir,
        save_dir=output_dir
    )
    
    base_recon_error, iso_recon_error = evaluate_mnist_models(
        base_encoder, base_generator,
        iso_encoder, iso_generator,
        data_loader, save_dir=output_dir
    )
    
    plt.figure(figsize=(6, 4))
    plt.bar(['Base LWGAN', 'Iso-LWGAN'], [base_recon_error, iso_recon_error])
    plt.title("Reconstruction Error Comparison")
    plt.ylabel("Reconstruction Error")
    plt.savefig(f"{output_dir}/recon_error_comparison.pdf", bbox_inches="tight", format="pdf")
    plt.close()
    
    print(f"Experiment 3 completed. Results saved to {output_dir}")
    return {
        'base_recon_error': base_recon_error,
        'iso_recon_error': iso_recon_error
    }

def test_all(device):
    """
    Run quick tests for each experiment.
    """
    print("\n" + "="*80)
    print("Running quick tests for each experiment...")
    print("="*80)
    
    experiment1_isometric_regularization(device, test_mode=True)
    experiment2_stochastic_generator(device, test_mode=True)
    experiment3_mnist_comparison(device, test_mode=True)
    
    print("\nQuick test run completed. Check printed output and generated PDF plots.")

def main():
    """
    Main function to run all experiments.
    """
    print("Starting Iso-LWGAN experiments...")
    start_time = time.time()
    
    device = setup_environment()
    
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    
    if test_mode:
        test_all(device)
    else:
        experiment1_isometric_regularization(device)
        experiment2_stochastic_generator(device)
        experiment3_mnist_comparison(device)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    
    global STATUS_ENUM
    STATUS_ENUM = "stopped"
    print(f"STATUS_ENUM set to: {STATUS_ENUM}")
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
