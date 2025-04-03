import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from preprocess import preprocess_data
from train import DiffusionModel, train_correction_module
from evaluate import (
    run_experiment, run_adaptive_experiment, evaluate_model_outputs,
    fixed_weight, confidence_weight, gradient_variance_weight,
    save_figure, test_experiments
)

def experiment_comparative_sampling(initial_data=None):
    """
    Experiment 1: Comparative Sampling Quality and Efficiency Experiment
    """
    print("\n========== Experiment 1: Comparative Sampling Quality and Efficiency ==========")
    torch.manual_seed(42)
    if initial_data is None:
        initial_data = torch.randn(32, 2)  # batch of 32 samples in 2D
    guidance_scale = 0.1

    correction_module = DiffusionModel(in_features=2)
    correction_module.eval()  # In practice, load a pretrained model

    print("Running baseline experiment (harmonic ansatz only)...")
    baseline_output, baseline_res, baseline_steps, baseline_time = run_experiment(
        correction_module=None,  # not used for baseline
        initial_data=initial_data,
        guidance_scale=guidance_scale,
        use_hncg=False
    )
    print(f"Baseline: Steps = {baseline_steps}, Total time = {baseline_time:.4f} sec")

    print("Running HNCG experiment (with neural correction)...")
    hncg_output, hncg_res, hncg_steps, hncg_time = run_experiment(
        correction_module=correction_module,
        initial_data=initial_data,
        guidance_scale=guidance_scale,
        use_hncg=True
    )
    print(f"HNCG: Steps = {hncg_steps}, Total time = {hncg_time:.4f} sec")

    fig = plt.figure(figsize=(10, 6))
    plt.plot(baseline_res, label="Baseline")
    plt.plot(hncg_res, label="HNCG")
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Convergence Curve Comparison")
    plt.legend()
    save_figure(fig, "convergence_curve_comparison_pair1.pdf")

def experiment_adaptive_weighting():
    """
    Experiment 2: Adaptive Weighting Module Ablation and Sensitivity Analysis
    """
    print("\n========== Experiment 2: Adaptive Weighting Module Ablation and Sensitivity Analysis ==========")
    torch.manual_seed(42)
    initial_data = torch.randn(32, 2)  # reinitialize for consistency
    guidance_scale = 0.15

    correction_module = DiffusionModel(in_features=2)
    correction_module.eval()

    weights = {
        "Fixed Weight": lambda x: fixed_weight(x, 0.5),
        "Confidence Threshold": confidence_weight,
        "Gradient Variance": gradient_variance_weight
    }

    results = {}
    for label, wf in weights.items():
        print(f"Running adaptive experiment with strategy: {label}")
        res = run_adaptive_experiment(correction_module, initial_data, wf, guidance_scale=guidance_scale)
        results[label] = res

    fig = plt.figure(figsize=(10, 6))
    for label, res in results.items():
        plt.plot(res, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Ablation: Convergence with Different Adaptive Weighting Strategies")
    plt.legend()
    save_figure(fig, "adaptive_weighting_convergence.pdf")

def experiment_joint_space_modeling():
    """
    Experiment 3: Joint-Space Modeling Contribution Study
    """
    print("\n========== Experiment 3: Joint-Space Modeling Contribution Study ==========")
    multimodal_data = preprocess_data(n_points=1000)

    print("Training correction module with guided-only data...")
    model_guided, losses_guided = train_correction_module(multimodal_data, guided_only=True, num_epochs=20)
    print("Training correction module with joint-space data...")
    model_joint, losses_joint = train_correction_module(multimodal_data, guided_only=False, num_epochs=20)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(losses_guided, label="Guided-Only")
    plt.plot(losses_joint, label="Joint-Space")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison for Correction Module Variants")
    plt.legend()
    save_figure(fig, "training_loss_comparison.pdf")

    pred_guided, pred_joint = evaluate_model_outputs(model_guided, model_joint)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axs[0].imshow(pred_guided, origin='lower', extent=(-5, 5, -5, 5))
    axs[0].set_title("Guided-Only Correction")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(pred_joint, origin='lower', extent=(-5, 5, -5, 5))
    axs[1].set_title("Joint-Space Correction")
    fig.colorbar(im1, ax=axs[1])
    plt.suptitle("Model Output Comparison on Evaluation Grid")
    save_figure(fig, "joint_space_modeling_pair1.pdf")

def main():
    """
    Main function to run all experiments
    """
    print("Starting HNCG Experiment on Tesla T4 GPU...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    os.makedirs('../logs', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    experiment_comparative_sampling()
    experiment_adaptive_weighting()
    experiment_joint_space_modeling()
    
    test_experiments()
    
    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()
