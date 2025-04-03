"""
Script for evaluating models for Joint-Guided Bayesian Flow Networks (JG-BFN) experiments.
"""
import torch
import time
import matplotlib.pyplot as plt
import os
import numpy as np

def sample_with_solver(model, num_steps, sample_shape, device):
    """
    A simplified reverse-time SDE solver that iterates with an Euler-like scheme.
    
    Args:
        model: The score model to use for sampling
        num_steps: Number of discretization steps
        sample_shape: Shape of samples to generate
        device: Device to run sampling on
        
    Returns:
        x: Generated samples
    """
    model.eval()
    with torch.no_grad():
        x = torch.randn(sample_shape, device=device)
        t_values = torch.linspace(1.0, 0.0, num_steps, device=device)
        for t in t_values:
            t_batch = t.expand(sample_shape[0], 1)
            score = model(x, t_batch)
            x = x + score * (1.0 / num_steps)
        return x

def compute_dummy_fid(fake_samples, real_samples):
    """
    Compute a dummy FID like metric as L2 difference between sample means.
    (In practice, use a proper FID implementation.)
    
    Args:
        fake_samples: Generated samples
        real_samples: Real data samples
        
    Returns:
        fid: Dummy FID score
    """
    fake_mean = fake_samples.mean().item()
    real_mean = real_samples.mean().item()
    fid = abs(fake_mean - real_mean)
    return fid

def run_efficiency_experiment(model, real_data, steps_list, device):
    """
    Evaluate sampling efficiency with various discretization steps.
    Compare the (dummy) FID metric and sampling runtime.
    
    Args:
        model: Trained model to evaluate
        real_data: Real data for FID computation
        steps_list: List of discretization steps to evaluate
        device: Device to run evaluation on
        
    Returns:
        sampling_results: Dictionary with evaluation results
        fid_list: List of FID scores
        time_list: List of sampling times
    """
    print("\nRunning Experiment 3: Efficiency and Fast Sampling Analysis")
    sampling_results = {}
    fid_list = []
    time_list = []
    
    sample_shape = (64,) + real_data.shape[1:]
    for steps in steps_list:
        start_time = time.time()
        samples = sample_with_solver(model, steps, sample_shape, device)
        duration = time.time() - start_time
        
        real_samples = real_data[:samples.size(0)].to(device)
        fid_score = compute_dummy_fid(samples, real_samples)
        sampling_results[steps] = {'time': duration, 'fid': fid_score}
        fid_list.append(fid_score)
        time_list.append(duration)
        print(f"Steps: {steps}, Time: {duration:.3f}s, Dummy FID: {fid_score:.3f}")
    
    return sampling_results, fid_list, time_list

def plot_loss_curve(loss_list, filename, title="Training Loss"):
    """
    Plot and save a loss curve.
    
    Args:
        loss_list: List of loss values
        filename: File to save the plot to
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(loss_list, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(filename, format='pdf')
    plt.close()
    print("Saved plot to", filename)

def plot_loss_comparison(loss_history_1, loss_history_2, labels, filename, title="Training Loss Comparison"):
    """
    Plot and save a comparison of two loss curves.
    
    Args:
        loss_history_1: First list of loss values
        loss_history_2: Second list of loss values
        labels: Labels for the two curves
        filename: File to save the plot to
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(loss_history_1, marker='o', label=labels[0])
    plt.plot(loss_history_2, marker='x', label=labels[1])
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format='pdf')
    plt.close()
    print("Saved training loss comparison plot to", filename)

def plot_sampling_tradeoff(steps_list, fid_list, time_list, filename):
    """
    Plot and save the sampling efficiency trade-off.
    
    Args:
        steps_list: List of discretization steps
        fid_list: List of FID scores
        time_list: List of sampling times
        filename: File to save the plot to
    """
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    color = 'tab:blue'
    ax1.set_xlabel("Discretization Steps")
    ax1.set_ylabel("FID", color=color)
    ax1.plot(steps_list, fid_list, marker='o', color=color, label="FID")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel("Sampling Time (s)", color=color)
    ax2.plot(steps_list, time_list, marker='x', color=color, label="Time (s)")
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title("Sampling Efficiency Trade-Off")
    plt.savefig(filename, format='pdf')
    plt.close()
    print("Saved plot to", filename)
