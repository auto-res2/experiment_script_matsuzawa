#!/usr/bin/env python3
"""
Main module for running the Adaptive Characteristic Simulation (ACS) experiments.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import EXP1_PARAMS, EXP2_PARAMS, EXP3_PARAMS, SAVE_DIR, STATUS_ENUM

from src.preprocess import ensure_directories, generate_synthetic_data
from src.train import (
    fixed_step_integrator, adaptive_step_integrator, 
    base_sde_euler_maruyama, acs_sde_euler_maruyama,
    SimpleUNet, train_unet_model
)
from src.evaluate import (
    experiment1_evaluate, experiment2_evaluate, experiment3_evaluate,
    ddim_sample_base, ddim_sample_acs
)

def update_status(status):
    """
    Update the status_enum in the config file.
    
    Args:
        status: New status value
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/config.py")
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.startswith("STATUS_ENUM"):
            lines[i] = f'STATUS_ENUM = "{status}"  # Will be set to "stopped" at the end of the experiment\n'
    
    with open(config_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Status updated to: {status}")

def experiment1():
    """
    Experiment 1: Adaptive Step Size Scheduler with Synthetic Diffusion Trajectories
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: ADAPTIVE STEP SIZE SCHEDULER WITH SYNTHETIC DIFFUSION TRAJECTORIES")
    print("="*80)
    
    print("\nExperiment Parameters:")
    print(f"  Initial value (y0): {EXP1_PARAMS['y0']}")
    print(f"  Initial time (t0): {EXP1_PARAMS['t0']}")
    print(f"  End time (tend): {EXP1_PARAMS['tend']}")
    print(f"  Fixed time step (dt_fixed): {EXP1_PARAMS['dt_fixed']}")
    print(f"  Initial adaptive time step (dt_initial): {EXP1_PARAMS['dt_initial']}")
    print(f"  Error tolerance (tol): {EXP1_PARAMS['tol']}")
    
    print("\nRunning fixed-step integration...")
    y0 = EXP1_PARAMS["y0"]
    t0 = EXP1_PARAMS["t0"]
    tend = EXP1_PARAMS["tend"]
    dt_fixed = EXP1_PARAMS["dt_fixed"]
    dt_initial = EXP1_PARAMS["dt_initial"]
    tol = EXP1_PARAMS["tol"]

    start_time = time.time()
    t_fixed, y_fixed = fixed_step_integrator(y0, t0, tend, dt_fixed)
    time_fixed = time.time() - start_time
    print(f"  Fixed-step integration completed in {time_fixed:.4f} seconds")
    print(f"  Number of iterations: {len(t_fixed)}")
    
    print("\nRunning adaptive-step integration...")
    start_time = time.time()
    t_adaptive, y_adaptive = adaptive_step_integrator(y0, t0, tend, dt_initial, tol)
    time_adaptive = time.time() - start_time
    print(f"  Adaptive-step integration completed in {time_adaptive:.4f} seconds")
    print(f"  Number of iterations: {len(t_adaptive)}")
    
    iteration_reduction = (len(t_fixed) - len(t_adaptive)) / len(t_fixed) * 100
    time_reduction = (time_fixed - time_adaptive) / time_fixed * 100
    print(f"\nEfficiency Improvement:")
    print(f"  Iteration reduction: {iteration_reduction:.2f}%")
    print(f"  Time reduction: {time_reduction:.2f}%")

    print("\nGenerating plots and evaluating results...")
    experiment1_evaluate(t_fixed, y_fixed, t_adaptive, y_adaptive, time_fixed, time_adaptive)
    
    print("\nExperiment 1 complete.")
    print("-"*80 + "\n")

def experiment2():
    """
    Experiment 2: Evaluating the Controlled Random Perturbations in the SDE Integration
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: CONTROLLED RANDOM PERTURBATIONS IN SDE INTEGRATION")
    print("="*80)
    
    print("\nExperiment Parameters:")
    print(f"  Total simulation time (T): {EXP2_PARAMS['T']}")
    print(f"  Time step (dt): {EXP2_PARAMS['dt']}")
    print(f"  Noise scale: {EXP2_PARAMS['noise_scale']}")
    print(f"  Error tolerance (tol): {EXP2_PARAMS['tol']}")
    
    T = EXP2_PARAMS["T"]
    dt = EXP2_PARAMS["dt"]
    noise_scale = EXP2_PARAMS["noise_scale"]
    tol = EXP2_PARAMS["tol"]
    x0 = torch.tensor([1.0])
    
    print("\nRunning base SDE simulation with fixed noise intensity...")
    start_time = time.time()
    traj_base = base_sde_euler_maruyama(x0, T, dt, noise_scale)
    time_base = time.time() - start_time
    print(f"  Base SDE simulation completed in {time_base:.4f} seconds")
    print(f"  Number of steps: {len(traj_base)}")
    
    print("\nRunning ACS SDE simulation with adaptive noise control...")
    start_time = time.time()
    traj_acs = acs_sde_euler_maruyama(x0, T, dt, noise_scale, tol)
    time_acs = time.time() - start_time
    print(f"  ACS SDE simulation completed in {time_acs:.4f} seconds")
    print(f"  Number of steps: {len(traj_acs)}")
    
    base_variance = torch.var(traj_base, dim=0).item()
    acs_variance = torch.var(traj_acs, dim=0).item()
    variance_reduction = (base_variance - acs_variance) / base_variance * 100
    
    print("\nVariance Statistics:")
    print(f"  Base SDE variance: {base_variance:.6f}")
    print(f"  ACS SDE variance: {acs_variance:.6f}")
    print(f"  Variance {'reduction' if variance_reduction > 0 else 'increase'}: {abs(variance_reduction):.2f}%")
    
    steps = np.arange(0, T + dt, dt)
    
    print("\nGenerating plots and evaluating results...")
    experiment2_evaluate(traj_base, traj_acs, steps)
    
    print("\nExperiment 2 complete.")
    print("-"*80 + "\n")

def experiment3():
    """
    Experiment 3: End-to-End Performance Comparison on a Benchmark Task
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: END-TO-END PERFORMANCE COMPARISON ON A BENCHMARK TASK")
    print("="*80)
    
    print("\nExperiment Parameters:")
    print(f"  Number of sampling steps: {EXP3_PARAMS['num_steps']}")
    print(f"  Eta parameter: {EXP3_PARAMS['eta']}")
    print(f"  Error tolerance (tol): {EXP3_PARAMS['tol']}")
    print(f"  Image size: {EXP3_PARAMS['image_size']}x{EXP3_PARAMS['image_size']}")
    print(f"  Number of channels: {EXP3_PARAMS['channels']}")
    print(f"  UNet features: {EXP3_PARAMS['features']}")
    
    num_steps = EXP3_PARAMS["num_steps"]
    eta = EXP3_PARAMS["eta"]
    tol = EXP3_PARAMS["tol"]
    image_size = EXP3_PARAMS["image_size"]
    channels = EXP3_PARAMS["channels"]
    features = EXP3_PARAMS["features"]
    
    print("\nGenerating synthetic image data...")
    x_init = generate_synthetic_data(size=image_size, channels=channels)
    print(f"  Created synthetic image with shape: {list(x_init.shape)}")
    
    print("\nInitializing UNet model...")
    model = SimpleUNet(in_channels=channels, out_channels=channels, features=features)
    model = train_unet_model(model)
    print("  Model initialized successfully")
    
    print("\nRunning Base DDIM sampling (fixed-step method)...")
    start_time = time.time()
    x_base, traj_base = ddim_sample_base(model, x_init, num_steps=num_steps, eta=eta)
    time_base = time.time() - start_time
    print(f"  Base DDIM sampling completed in {time_base:.4f} seconds")
    print(f"  Number of sampling steps: {len(traj_base)-1}")
    
    print("\nRunning ACS sampling (adaptive integration)...")
    start_time = time.time()
    x_acs, traj_acs = ddim_sample_acs(model, x_init, num_steps=num_steps, eta=eta, tol=tol)
    time_acs = time.time() - start_time
    print(f"  ACS sampling completed in {time_acs:.4f} seconds")
    print(f"  Number of sampling steps: {len(traj_acs)-1}")
    
    time_diff = ((time_base - time_acs) / time_base) * 100
    step_diff = ((len(traj_base) - len(traj_acs)) / len(traj_base)) * 100
    
    print("\nPerformance Comparison:")
    print(f"  Base DDIM sampling time: {time_base:.4f} seconds")
    print(f"  ACS sampling time: {time_acs:.4f} seconds")
    print(f"  Time {'reduction' if time_diff > 0 else 'increase'}: {abs(time_diff):.2f}%")
    print(f"  Step {'reduction' if step_diff > 0 else 'increase'}: {abs(step_diff):.2f}%")
    
    print("\nGenerating plots and evaluating results...")
    experiment3_evaluate(x_base, traj_base, x_acs, traj_acs)
    
    print("\nExperiment 3 complete.")
    print("-"*80 + "\n")

def run_all_tests():
    """
    Run all experiments and measure total execution time.
    """
    print("\n" + "="*80)
    print("ADAPTIVE CHARACTERISTIC SIMULATION (ACS) EXPERIMENTS")
    print("="*80)
    
    print("\nSystem Information:")
    print(f"  Python version: {sys.version.split()[0]}")
    print(f"  NumPy version: {np.__version__}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    
    print("\nExperiment Configuration:")
    print(f"  Save directory: {SAVE_DIR}")
    print(f"  Current status: {STATUS_ENUM}")
    
    print("\nRunning all experiments...\n")
    start = time.time()
    
    experiment1()
    experiment2()
    experiment3()
    
    end = time.time()
    total_time = end - start
    
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"All plots saved to: {os.path.abspath(SAVE_DIR)}")
    print("\nExperiment files generated:")
    for filename in sorted(os.listdir(SAVE_DIR)):
        if filename.endswith('.pdf'):
            file_path = os.path.join(SAVE_DIR, filename)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            print(f"  - {filename} ({file_size:.1f} KB)")
    
    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    ensure_directories()
    
    update_status("running")
    
    try:
        run_all_tests()
        
        update_status("stopped")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        update_status("stopped")
        sys.exit(1)
