#!/usr/bin/env python
"""
Main script for running the HEDA experiments.
This script orchestrates the execution of experiments comparing the HEDA method
with a base method across different evaluation metrics.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from preprocess import prepare_experiment_data, load_config
from train import run_convergence_experiment, run_sampling_with_timing
from evaluate import (
    run_hyperparameter_grid_search, 
    plot_convergence_comparison,
    plot_trajectory_comparison,
    plot_runtime_comparison,
    plot_hyperparameter_heatmaps
)
from utils import base_method_update, heda_update

def run_convergence_and_stability_experiment(config):
    """Execute the convergence and stability experiment."""
    print("=== Convergence and Stability Experiment ===")
    
    # Prepare data
    data = prepare_experiment_data('convergence', config)
    
    # Run Base Method
    errors_base, states_base = run_convergence_experiment(
        base_method_update, 
        data['state_init'], 
        max_iter=data['max_iter'], 
        tol=data['tol'], 
        method_name="Base Method"
    )
    
    # Run HEDA
    errors_heda, states_heda = run_convergence_experiment(
        heda_update, 
        data['state_init'], 
        max_iter=data['max_iter'], 
        tol=data['tol'], 
        method_name="HEDA", 
        hyperparams={'step_size': data['step_size'], 'memory_size': data['memory_size']}
    )
    
    # Plot and save results
    plot_convergence_comparison(errors_base, errors_heda)
    plot_trajectory_comparison(states_base, states_heda)
    
    return errors_base, errors_heda, states_base, states_heda

def run_resource_efficiency_experiment(config):
    """Execute the computational resource efficiency experiment."""
    print("=== Computational Resource Efficiency Experiment ===")
    
    # Prepare data
    data = prepare_experiment_data('resource', config)
    
    # Run both methods
    total_time_base, times_base, peak_mem_base = run_sampling_with_timing(
        base_method_update, 
        data['state_init'], 
        max_iter=data['max_iter'], 
        method_name="Base Method"
    )
    
    total_time_heda, times_heda, peak_mem_heda = run_sampling_with_timing(
        heda_update, 
        data['state_init'], 
        max_iter=data['max_iter'], 
        method_name="HEDA", 
        hyperparams={'step_size': data['step_size'], 'memory_size': data['memory_size']}
    )
    
    # Plot runtime comparison
    plot_runtime_comparison(times_base, times_heda)
    
    # Print memory comparison
    if peak_mem_base is not None and peak_mem_heda is not None:
        memory_reduction = (peak_mem_base - peak_mem_heda) / peak_mem_base * 100
        print(f"Memory usage reduction with HEDA: {memory_reduction:.2f}%")
    
    return total_time_base, total_time_heda, times_base, times_heda, peak_mem_base, peak_mem_heda

def run_hyperparameter_sensitivity_experiment(config):
    """Execute the hyperparameter sensitivity and robustness evaluation."""
    print("=== Hyperparameter Sensitivity and Robustness Evaluation ===")
    
    # Prepare data
    data = prepare_experiment_data('hyperparameter', config)
    
    # Run grid search
    results_df = run_hyperparameter_grid_search(
        data['step_sizes'], 
        data['memory_sizes'], 
        data['state_init'], 
        max_iter=data['max_iter'], 
        tol=data['tol']
    )
    
    # Plot heatmaps
    plot_hyperparameter_heatmaps(results_df)
    
    return results_df

def test_code(config):
    """Run minimal tests to verify implementation."""
    print("Starting minimal tests...\n")
    
    # Get test data
    test_data = prepare_experiment_data('test', config)
    
    # Test Convergence Experiment (Base Method)
    print("Testing Convergence Experiment (Base Method)")
    run_convergence_experiment(
        base_method_update, 
        test_data['state_init_conv'], 
        max_iter=test_data['max_iter'], 
        tol=test_data['tol'], 
        method_name="Base Method Test"
    )
    
    # Test Convergence Experiment (HEDA)
    print("\nTesting Convergence Experiment (HEDA)")
    run_convergence_experiment(
        heda_update, 
        test_data['state_init_conv'], 
        max_iter=test_data['max_iter'], 
        tol=test_data['tol'], 
        method_name="HEDA Test", 
        hyperparams={'step_size': test_data['step_size'], 'memory_size': test_data['memory_size']}
    )
    
    # Test Computational Efficiency
    print("\nTesting Computational Efficiency (Base Method)")
    run_sampling_with_timing(
        base_method_update, 
        test_data['state_init_time'], 
        max_iter=5, 
        method_name="Base Method Timing Test"
    )
    
    print("\nTesting Computational Efficiency (HEDA)")
    run_sampling_with_timing(
        heda_update, 
        test_data['state_init_time'], 
        max_iter=5, 
        method_name="HEDA Timing Test", 
        hyperparams={'step_size': test_data['step_size'], 'memory_size': test_data['memory_size']}
    )
    
    print("\nMinimal tests finished.")
    return True

def main():
    """Main entry point for HEDA experiments."""
    # Print hardware information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    # Load configuration
    config = load_config()
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run test to verify code works
    if config['experiments']['test']['enable']:
        test_code(config)
    
    # Run full experiments if enabled
    results = {}
    
    if config['experiments']['convergence']['enable']:
        results['convergence'] = run_convergence_and_stability_experiment(config)
    
    if config['experiments']['resource']['enable']:
        results['resource'] = run_resource_efficiency_experiment(config)
    
    if config['experiments']['hyperparameter']['enable']:
        results['hyperparameter'] = run_hyperparameter_sensitivity_experiment(config)
    
    # Save experiment summary
    summary = {
        'device': str(device),
        'experiments_run': [k for k, v in config['experiments'].items() if v['enable']],
        'timestamp': str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'
    }
    
    # Save summary to file
    with open('logs/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAll experiments completed. Results saved to logs directory.")

if __name__ == "__main__":
    main()
