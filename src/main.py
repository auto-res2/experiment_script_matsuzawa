"""
Main experiment script for D2PTR (Dual-Stage Diffusion Purification with Trigger Reversion).

This script implements three experiments to validate D2PTR components:
1. Purification Robustness Against Adversarial Perturbations
2. Backdoor Trigger Detection and Reversion
3. Adaptive Parameter Tuning via Latent Distribution Divergence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from typing import Dict, Tuple, List, Optional, Any

from config.experiment_config import (
    DEVICE, RANDOM_SEED, LATENT_DIM, DIFFUSION_STEPS, STEP_SIZE,
    DIVERGENCE_THRESHOLD, REVERSION_LR, REVERSION_STEPS, MAX_ADAPTIVE_STEPS,
    STEP_SIZE_DECAY, DATASET, DATA_DIR
)
from src.utils.models import SimpleCNN, LatentEncoder, DiffusionPurifier
from src.utils.diffusion_utils import (
    fgsm_attack, insert_trigger, kl_divergence, set_seed
)
from src.preprocess import get_dataset
from src.train import train_classifier, train_latent_encoder
from src.evaluate import (
    evaluate_classifier, test_purification_robustness,
    trigger_reversion, adaptive_purification
)
from config.experiment_config import EPSILON

def test_experiment1():
    """
    Experiment 1: Purification Robustness Against Adversarial Perturbations
    
    This experiment tests the effectiveness of the diffusion purification
    stage in protecting against adversarial examples.
    """
    print("\n" + "="*80)
    print(" Experiment 1: Purification Robustness Against Adversarial Perturbations ")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    _, test_loader = get_dataset(DATASET, DATA_DIR)
    
    classifier = SimpleCNN(num_classes=10).to(DEVICE)
    purifier = DiffusionPurifier(num_steps=DIFFUSION_STEPS, step_size=STEP_SIZE).to(DEVICE)
    
    classifier.eval()
    
    results = test_purification_robustness(
        model=classifier,
        purifier=purifier,
        test_loader=test_loader
    )
    
    print("Experiment 1 completed.")
    return results

def test_experiment2():
    """
    Experiment 2: Backdoor Trigger Detection and Reversion
    
    This experiment tests the ability of the trigger reversion module
    to detect and remove backdoor triggers from inputs.
    """
    print("\n" + "="*80)
    print(" Experiment 2: Backdoor Trigger Detection and Reversion ")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    _, test_loader = get_dataset(DATASET, DATA_DIR)
    
    encoder = LatentEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    
    benign_mean = torch.zeros(LATENT_DIM).to(DEVICE)
    benign_std = torch.ones(LATENT_DIM).to(DEVICE)
    
    images, _ = next(iter(test_loader))
    images = images.to(DEVICE)
    
    triggered_images = torch.stack([insert_trigger(img, trigger_value=1.0) for img in images])
    triggered_images = triggered_images.to(DEVICE)
    
    latent_triggered = encoder(triggered_images)
    initial_kl = kl_divergence(latent_triggered, benign_mean, benign_std)
    print(f"Initial KL divergence (triggered): {initial_kl.item():.4f}")
    
    latent_reverted, divergence_list = trigger_reversion(
        latent_triggered, benign_mean, benign_std
    )
    post_kl = kl_divergence(latent_reverted, benign_mean, benign_std)
    print(f"Post-reversion KL divergence: {post_kl.item():.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [initial_kl.item(), post_kl.item()], marker='o', linewidth=2)
    plt.xticks([0, 1], ["Triggered", "Reverted"])
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence Before and After Trigger Reversion")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/kl_divergence_trigger.pdf", format="pdf", dpi=300)
    plt.close()
    
    print("Experiment 2 completed.")
    return {
        "initial_kl": initial_kl.item(),
        "post_kl": post_kl.item(),
        "improvement": initial_kl.item() - post_kl.item()
    }

def test_experiment3():
    """
    Experiment 3: Adaptive Parameter Tuning via Latent Distribution Divergence
    
    This experiment tests the adaptive parameter tuning mechanism based on
    latent distribution divergence.
    """
    print("\n" + "="*80)
    print(" Experiment 3: Adaptive Parameter Tuning via Latent Distribution Divergence ")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    _, test_loader = get_dataset(DATASET, DATA_DIR)
    
    classifier = SimpleCNN(num_classes=10).to(DEVICE)
    encoder = LatentEncoder(latent_dim=LATENT_DIM).to(DEVICE)
    purifier = DiffusionPurifier(num_steps=DIFFUSION_STEPS, step_size=STEP_SIZE).to(DEVICE)
    
    classifier.eval()
    encoder.eval()
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    
    images.requires_grad = True
    outputs = classifier(images)
    loss = F.cross_entropy(outputs, labels)
    classifier.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    adv_images = fgsm_attack(images, EPSILON, data_grad)
    
    purified_images, final_div, divergence_list = adaptive_purification(
        adv_images, purifier, encoder
    )
    print(f"Final divergence difference after adaptive tuning: {final_div:.4f}")
    
    with torch.no_grad():
        out_purified = classifier(purified_images)
        pred = out_purified.argmax(dim=1)
        acc = (pred == labels).float().mean().item() * 100
    print(f"Classification accuracy after adaptive tuning: {acc:.2f}%")
    
    plt.figure(figsize=(8, 6))
    steps = list(range(len(divergence_list)))
    plt.plot(steps, divergence_list, marker='o', linewidth=2)
    plt.xlabel("Adaptive Step")
    plt.ylabel("Divergence Difference")
    plt.title("Adaptive Tuning Divergence over Steps")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/divergence_adaptive.pdf", format="pdf", dpi=300)
    plt.close()
    
    print("Experiment 3 completed.")
    return {
        "final_divergence": final_div,
        "accuracy": acc,
        "num_adaptive_steps": len(divergence_list) - 1
    }

def run_all_experiments():
    """
    Run all three experiments to validate D2PTR components.
    """
    print("\n" + "="*80)
    print(" D2PTR: Dual-Stage Diffusion Purification with Trigger Reversion ")
    print("="*80)
    
    os.makedirs("logs", exist_ok=True)
    
    start_time = time.time()
    
    results = {}
    results["experiment1"] = test_experiment1()
    results["experiment2"] = test_experiment2()
    results["experiment3"] = test_experiment3()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*80)
    print(" Summary of Results ")
    print("="*80)
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print("\nExperiment 1 (Purification Robustness):")
    print(f"  Original accuracy: {results['experiment1']['original_accuracy']:.2f}%")
    print(f"  Adversarial accuracy: {results['experiment1']['adversarial_accuracy']:.2f}%")
    print(f"  Purified accuracy: {results['experiment1']['purified_accuracy']:.2f}%")
    
    print("\nExperiment 2 (Trigger Reversion):")
    print(f"  Initial KL divergence: {results['experiment2']['initial_kl']:.4f}")
    print(f"  Post-reversion KL divergence: {results['experiment2']['post_kl']:.4f}")
    print(f"  Improvement: {results['experiment2']['improvement']:.4f}")
    
    print("\nExperiment 3 (Adaptive Parameter Tuning):")
    print(f"  Final divergence: {results['experiment3']['final_divergence']:.4f}")
    print(f"  Classification accuracy: {results['experiment3']['accuracy']:.2f}%")
    print(f"  Number of adaptive steps: {results['experiment3']['num_adaptive_steps']}")
    
    return results

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Using CPU.")
        from config.experiment_config import DEVICE
        globals()["DEVICE"] = "cpu"
    
    run_all_experiments()
