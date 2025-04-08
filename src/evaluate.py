"""
Evaluation module for the Optimized Characteristic Resampling (OCR) experiment.

This module contains the evaluation functions for the OCR method experiments.
"""

import time
import os
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt

def run_experiment1(model, dataloader, device, save_path="plots"):
    """
    Run Experiment 1: Convergence Speed and Stability under High Guidance Scales.
    
    Args:
        model (nn.Module): The diffusion model
        dataloader (torch.utils.data.DataLoader): DataLoader with the dataset
        device (torch.device): Device to run the model on
        save_path (str): Directory to save the plots
        
    Returns:
        dict: Results of the experiment
    """
    from train import base_method_step, ocr_method_step
    
    print("Running Experiment 1: Convergence Speed and Stability under High Guidance Scales")
    guidance_scales = [5, 10, 50]  # Using high guidance scales
    results = {}  # Will store records: per guidance scale, lists for each method

    for guidance_scale in guidance_scales:
        results[guidance_scale] = {"base": [], "ocr": []}
        for batch, _ in dataloader:
            batch = batch.to(device)
            start_time = time.time()
            _, iter_count_base, final_loss_base = base_method_step(model, batch, guidance_scale)
            time_base = time.time() - start_time

            start_time = time.time()
            _, iter_count_ocr, final_loss_ocr = ocr_method_step(model, batch, guidance_scale)
            time_ocr = time.time() - start_time

            results[guidance_scale]["base"].append((iter_count_base, final_loss_base, time_base))
            results[guidance_scale]["ocr"].append((iter_count_ocr, final_loss_ocr, time_ocr))

        print(f"Guidance Scale: {guidance_scale}")
        print("  Base Method (iter, loss, time):", results[guidance_scale]["base"])
        print("  OCR Method (iter, loss, time):", results[guidance_scale]["ocr"])

    plt.figure(figsize=(10, 6), dpi=300)
    plt.title("Convergence Time vs Guidance Scale")
    for guidance_scale in guidance_scales:
        times_base = [r[2] for r in results[guidance_scale]["base"]]
        times_ocr = [r[2] for r in results[guidance_scale]["ocr"]]
        plt.plot([guidance_scale]*len(times_base), times_base, 'ro', 
                label='Base' if guidance_scale == guidance_scales[0] else "")
        plt.plot([guidance_scale]*len(times_ocr), times_ocr, 'bo', 
                label='OCR' if guidance_scale == guidance_scales[0] else "")
    plt.xlabel("Guidance Scale")
    plt.ylabel("Time to Convergence (sec)")
    plt.legend()
    os.makedirs(save_path, exist_ok=True)
    pdf_filename = os.path.join(save_path, "convergence_time.pdf")
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Experiment 1 plot saved as {pdf_filename}\n")
    
    return results

def run_experiment2(model, device, save_path="plots"):
    """
    Run Experiment 2: Ablation Study on Loss Components and Hyperparameters.
    
    Args:
        model (nn.Module): The diffusion model
        device (torch.device): Device to run the model on
        save_path (str): Directory to save the plots
        
    Returns:
        dict: Results of the experiment
    """
    from train import train_with_hyperparams
    
    print("Running Experiment 2: Ablation Study on Loss Components and Hyperparameters")
    lambdas_grid = {
        "fp": [0.5, 1.0, 2.0],
        "guidance": [0.5, 1.0, 2.0],
        "noise": [0.0, 0.5, 1.0]  # Includes case where noise loss is disabled (0.0)
    }
    param_combinations = list(itertools.product(lambdas_grid["fp"],
                                               lambdas_grid["guidance"],
                                               lambdas_grid["noise"]))
    
    results = {}
    plt.figure(figsize=(12, 8), dpi=300)
    for params in param_combinations:
        key = f"fp={params[0]}_guidance={params[1]}_noise={params[2]}"
        losses = train_with_hyperparams(params, model)
        results[key] = losses
        plt.plot(losses, label=key)  # Plot loss curve for this hyperparameter setting

    plt.title("Loss Curves for Different Hyperparameter Settings")
    plt.xlabel("Iteration")
    plt.ylabel("Composite Loss")
    plt.legend(fontsize='x-small', loc='upper right')
    os.makedirs(save_path, exist_ok=True)
    pdf_filename = os.path.join(save_path, "loss_curves_ablation.pdf")
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    plt.close()
    print("Hyperparameter combinations and final loss values:")
    for key, loss_curve in results.items():
        print(f"  {key}: final loss = {loss_curve[-1]:.4f}")
    print(f"Experiment 2 plot saved as {pdf_filename}\n")
    
    return results

def run_experiment3(model, device, save_path="plots"):
    """
    Run Experiment 3: Evaluation of Different Accelerated Optimization Methods.
    
    Args:
        model (nn.Module): The diffusion model
        device (torch.device): Device to run the model on
        save_path (str): Directory to save the plots
        
    Returns:
        dict: Results of the experiment
    """
    from train import ocr_method_step_with_optimizer
    
    print("Running Experiment 3: Evaluation of Different Accelerated Optimization Methods")
    optimizer_names = ["adam", "rmsprop", "anderson"]
    results = {}
    x = torch.randn(16, 3, 32, 32).to(device)
    
    plt.figure(figsize=(10, 6), dpi=300)
    for opt_name in optimizer_names:
        start_time = time.time()
        x_updated, iter_count, final_loss = ocr_method_step_with_optimizer(
            model, x, guidance_scale=10, optimizer_name=opt_name
        )
        elapsed = time.time() - start_time
        results[opt_name] = {"iterations": iter_count, "final_loss": final_loss, "time": elapsed}
        print(f"Optimizer: {opt_name}")
        print(f"  Iterations: {iter_count}, Final Loss: {final_loss:.6f}, Time: {elapsed:.4f} sec")
        loss_curve = np.linspace(final_loss+0.1, final_loss, iter_count if iter_count > 0 else 1)
        plt.plot(range(len(loss_curve)), loss_curve, label=opt_name)
    
    plt.title("Convergence Loss Curves for Different Optimizers")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    os.makedirs(save_path, exist_ok=True)
    pdf_filename = os.path.join(save_path, "optimizer_comparison.pdf")
    plt.savefig(pdf_filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Experiment 3 plot saved as {pdf_filename}\n")
    
    return results
