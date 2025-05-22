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

from .preprocess import ensure_directories, generate_synthetic_data
from .train import (
    fixed_step_integrator, adaptive_step_integrator, 
    base_sde_euler_maruyama, acs_sde_euler_maruyama,
    SimpleUNet, train_unet_model
)
from .evaluate import (
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
    print("Starting Experiment 1: Adaptive Step Size Scheduler on Synthetic Diffusion Trajectories")
    y0 = EXP1_PARAMS["y0"]
    t0 = EXP1_PARAMS["t0"]
    tend = EXP1_PARAMS["tend"]
    dt_fixed = EXP1_PARAMS["dt_fixed"]
    dt_initial = EXP1_PARAMS["dt_initial"]
    tol = EXP1_PARAMS["tol"]

    start_time = time.time()
    t_fixed, y_fixed = fixed_step_integrator(y0, t0, tend, dt_fixed)
    time_fixed = time.time() - start_time

    start_time = time.time()
    t_adaptive, y_adaptive = adaptive_step_integrator(y0, t0, tend, dt_initial, tol)
    time_adaptive = time.time() - start_time

    experiment1_evaluate(t_fixed, y_fixed, t_adaptive, y_adaptive, time_fixed, time_adaptive)
    
    print("Experiment 1 complete.\n")

def experiment2():
    """
    Experiment 2: Evaluating the Controlled Random Perturbations in the SDE Integration
    """
    print("Starting Experiment 2: Controlled Random Perturbations in SDE Integration")
    T = EXP2_PARAMS["T"]
    dt = EXP2_PARAMS["dt"]
    noise_scale = EXP2_PARAMS["noise_scale"]
    tol = EXP2_PARAMS["tol"]
    x0 = torch.tensor([1.0])

    traj_base = base_sde_euler_maruyama(x0, T, dt, noise_scale)
    traj_acs = acs_sde_euler_maruyama(x0, T, dt, noise_scale, tol)

    steps = np.arange(0, T + dt, dt)

    experiment2_evaluate(traj_base, traj_acs, steps)
    
    print("Experiment 2 complete.\n")

def experiment3():
    """
    Experiment 3: End-to-End Performance Comparison on a Benchmark Task
    """
    print("Starting Experiment 3: End-to-End Performance Comparison on a Benchmark Task")
    num_steps = EXP3_PARAMS["num_steps"]
    eta = EXP3_PARAMS["eta"]
    tol = EXP3_PARAMS["tol"]
    image_size = EXP3_PARAMS["image_size"]
    channels = EXP3_PARAMS["channels"]
    features = EXP3_PARAMS["features"]

    x_init = generate_synthetic_data(size=image_size, channels=channels)

    model = SimpleUNet(in_channels=channels, out_channels=channels, features=features)
    model = train_unet_model(model)

    x_base, traj_base = ddim_sample_base(model, x_init, num_steps=num_steps, eta=eta)
    x_acs, traj_acs = ddim_sample_acs(model, x_init, num_steps=num_steps, eta=eta, tol=tol)

    experiment3_evaluate(x_base, traj_base, x_acs, traj_acs)
    
    print("Experiment 3 complete.\n")

def run_all_tests():
    """
    Run all experiments and measure total execution time.
    """
    print("Running all experiments...\n")
    start = time.time()
    
    experiment1()
    experiment2()
    experiment3()
    
    end = time.time()
    print("All experiments completed in {:.2f} seconds.".format(end - start))

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
