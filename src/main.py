"""
Main script for running RARD experiments.
This script implements the Regime-Adaptive Accelerated Reproducible Diffusion (RARD) method
and runs three experiments to demonstrate its effectiveness.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.integrate import solve_ivp
import time

os.makedirs('logs', exist_ok=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.experiment_config import (
    EXPERIMENT1_CONFIG, 
    EXPERIMENT2_CONFIG, 
    EXPERIMENT3_CONFIG,
    VERBOSE
)
from src.preprocess import generate_synthetic_data, prepare_initial_samples
from src.train import rard_solver, dynamic_regime_solver, fixed_threshold_solver, diffusion_sampling
from src.evaluate import evaluate_experiment1, evaluate_experiment2, evaluate_experiment3


def experiment1():
    """
    Experiment 1: Accelerated Coarse Exploration via Momentum‐Augmented ODE Solvers
    """
    print("Running Experiment 1: Accelerated Coarse Exploration via Momentum‐Augmented ODE Solvers")
    
    x0 = EXPERIMENT1_CONFIG['x0']
    t_final = EXPERIMENT1_CONFIG['t_final']
    
    results = evaluate_experiment1(x0, t_final)
    
    print("Experiment 1 (Base RK45): Final error = {:.6e}".format(results['error_rk'][-1]))
    print("Experiment 1 (RARD-inspired): Final error = {:.6e}".format(results['error_rard_custom'][-1]))

    plt.figure(figsize=(8,5))
    plt.semilogy(results['t_rk'], results['error_rk'], label='Base RK45 (Fixed-Step)', linestyle='--', marker='o', markersize=3)
    plt.semilogy(results['t_rard'], results['error_rard_custom'], label='RARD-inspired Momentum Augmented', linestyle='-', marker='x', markersize=3)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.title('Convergence Comparison in Coarse Exploration Phase', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle=':')
    plt.savefig("logs/convergence_comparison_experiment1.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("Experiment 1 plot saved as 'logs/convergence_comparison_experiment1.pdf'\n")


def experiment2():
    """
    Experiment 2: Regime-Specific Correction with Dynamic Step-Size Tuning
    """
    print("Running Experiment 2: Regime-Specific Correction with Dynamic Step-Size Tuning")
    
    x0 = EXPERIMENT2_CONFIG['x0']
    t_final = EXPERIMENT2_CONFIG['t_final']
    
    results = evaluate_experiment2(x0, t_final)
    
    fig, axs = plt.subplots(1, 2, figsize=(12,5))

    axs[0].plot(results['times_dynamic'], results['errors_dynamic'], label='Dynamic Regime-specific', color='blue', marker='o', markersize=3)
    axs[0].plot(results['times_fixed'], results['errors_fixed'], label='Fixed Threshold', color='red', marker='x', markersize=3)
    axs[0].set_xlabel('Time', fontsize=12)
    axs[0].set_ylabel('Absolute Error', fontsize=12)
    axs[0].set_title('Error Dynamics', fontsize=14)
    axs[0].legend(fontsize=10)
    axs[0].grid(True, which='both', linestyle=':')

    axs[1].plot(results['scores_dynamic'], label='RP Score (Dynamic)', color='green', marker='o', markersize=3)
    axs[1].plot(results['scores_fixed'], label='RP Score (Fixed)', color='purple', marker='x', markersize=3)
    axs[1].set_xlabel('Iteration', fontsize=12)
    axs[1].set_ylabel('RP Score', fontsize=12)
    axs[1].set_title('Consistency Metrics', fontsize=14)
    axs[1].legend(fontsize=10)
    axs[1].grid(True, which='both', linestyle=':')

    plt.tight_layout()
    plt.savefig("logs/error_and_consistency_experiment2.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("Experiment 2 plot saved as 'logs/error_and_consistency_experiment2.pdf'\n")


def experiment3():
    """
    Experiment 3: End-to-End Ablation Study of RARD Components
    """
    print("Running Experiment 3: End-to-End Ablation Study of RARD Components")
    
    n_samples = EXPERIMENT3_CONFIG['n_samples']
    n_steps = EXPERIMENT3_CONFIG['n_steps']
    
    results = evaluate_experiment3(n_samples, n_steps)
    
    plt.figure(figsize=(8,5))
    plt.plot(results['rp_full'], label="Full RARD", color='blue', marker='o', markersize=3)
    plt.plot(results['rp_no_dynamic'], label="No Dynamic Tuning", color='red', marker='x', markersize=3)
    plt.plot(results['rp_no_regime'], label="No Regime-Adaptation", color='green', marker='s', markersize=3)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean Distance to Target (RP Score)", fontsize=12)
    plt.title("Ablation Study: RP Score over Diffusion Sampling", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plt.savefig("logs/rp_score_ablation_experiment3.pdf", format="pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("Experiment 3: RP score plot saved as 'logs/rp_score_ablation_experiment3.pdf'")
    
    print("Final MAE (Full RARD): {:.6e}".format(results['final_mae_full']))
    print("Final MAE (No Dynamic Tuning): {:.6e}".format(results['final_mae_no_dynamic']))
    print("Final MAE (No Regime Adaptation): {:.6e}".format(results['final_mae_no_regime']))
    print()


def test_code():
    """
    Runs a quick test on each experiment – a short run to check that the code is
    executing without errors. This function returns immediately after execution.
    """
    print("==== Starting quick test run for all experiments ====")
    start_time = time.time()
    
    old_exp1_config = EXPERIMENT1_CONFIG.copy()
    old_exp2_config = EXPERIMENT2_CONFIG.copy()
    old_exp3_config = EXPERIMENT3_CONFIG.copy()
    
    EXPERIMENT1_CONFIG['t_final'] = 1.0  # Shorter time
    EXPERIMENT2_CONFIG['t_final'] = 1.0  # Shorter time
    EXPERIMENT3_CONFIG['n_samples'] = 50  # Fewer samples
    EXPERIMENT3_CONFIG['n_steps'] = 20   # Fewer steps
    
    experiment1()
    experiment2()
    experiment3()
    
    EXPERIMENT1_CONFIG.update(old_exp1_config)
    EXPERIMENT2_CONFIG.update(old_exp2_config)
    EXPERIMENT3_CONFIG.update(old_exp3_config)
    
    end_time = time.time()
    print(f"==== Quick test run completed in {end_time - start_time:.2f} seconds ====")


def main():
    """
    Main function to run all experiments.
    """
    print("Starting all experiments for RARD evaluation...\n")
    start_time = time.time()
    
    experiment1()
    experiment2()
    experiment3()
    
    end_time = time.time()
    print(f"All experiments completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_code()
    else:
        main()
