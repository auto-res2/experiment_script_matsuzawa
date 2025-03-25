#!/usr/bin/env python
"""
Evaluation module for the HEDA experiments.
This module contains functions to evaluate and compare methods.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sns
import os
from utils import base_method_update, heda_update
from train import evaluate_method, run_convergence_experiment, run_sampling_with_timing

def run_hyperparameter_grid_search(step_sizes, memory_sizes, state_init, max_iter=100, tol=1e-3):
    """
    Run a grid search over selected hyperparameters for both methods.
    
    Args:
        step_sizes: List of step sizes to evaluate
        memory_sizes: List of memory sizes to evaluate
        state_init: Initial state tensor
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        results_df: DataFrame containing evaluation results
    """
    results = []
    
    # Loop over both methods.
    for method_name, method_update in zip(["Base Method", "HEDA"], [base_method_update, heda_update]):
        for ss, ms in itertools.product(step_sizes, memory_sizes):
            hyperparams = {'step_size': ss, 'memory_size': ms}
            print(f"\nEvaluating {method_name} with step_size={ss} and memory_size={ms}")
            iterations, final_error = evaluate_method(method_update, hyperparams, state_init, max_iter=max_iter, tol=tol)
            results.append({
                'method': method_name,
                'step_size': ss,
                'memory_size': ms,
                'iterations': iterations,
                'final_error': final_error
            })
    
    results_df = pd.DataFrame(results)
    print("\nHyperparameter Evaluation Results:")
    print(results_df)
    
    return results_df

def plot_convergence_comparison(errors_base, errors_heda, save_dir='logs'):
    """Plot and save convergence curves comparison."""
    plt.figure(figsize=(10, 5))
    plt.plot(errors_base, label='Base Method')
    plt.plot(errors_heda, label='HEDA')
    plt.xlabel('Iteration')
    plt.ylabel('Error (L2-Norm)')
    plt.title('Convergence Comparison')
    plt.legend()
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/convergence_comparison.png')
    plt.close()

def plot_trajectory_comparison(states_base, states_heda, save_dir='logs'):
    """Plot and save trajectory comparison."""
    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(states_base, axis=1), label='Base Method Mean Trajectory')
    plt.plot(np.mean(states_heda, axis=1), label='HEDA Mean Trajectory')
    plt.xlabel('Iteration')
    plt.ylabel('Mean State Value')
    plt.title('Trajectory Comparison')
    plt.legend()
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/trajectory_comparison.png')
    plt.close()

def plot_runtime_comparison(times_base, times_heda, save_dir='logs'):
    """Plot and save runtime comparison."""
    plt.figure(figsize=(10, 5))
    plt.plot(times_base, label='Base Method')
    plt.plot(times_heda, label='HEDA')
    plt.xlabel('Iteration')
    plt.ylabel('Time (sec)')
    plt.title('Runtime per Iteration Comparison')
    plt.legend()
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/runtime_comparison.png')
    plt.close()

def plot_hyperparameter_heatmaps(results_df, save_dir='logs'):
    """Plot and save hyperparameter evaluation heatmaps."""
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize final error heatmaps for each method.
    for method in results_df.method.unique():
        sub_df = results_df[results_df.method == method].pivot(index="step_size", columns="memory_size", values="final_error")
        plt.figure(figsize=(6, 4))
        sns.heatmap(sub_df, annot=True, cmap="viridis")
        plt.title(f'{method} - Final Error Heatmap')
        plt.xlabel('Memory Size')
        plt.ylabel('Step Size')
        plt.savefig(f'{save_dir}/{method.replace(" ", "_").lower()}_heatmap.png')
        plt.close()
