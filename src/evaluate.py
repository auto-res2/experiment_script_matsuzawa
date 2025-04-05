"""
Evaluation functions for D2PTR experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Tuple, List, Optional, Any

from config.experiment_config import (
    DEVICE, EPSILON, DIVERGENCE_THRESHOLD, 
    REVERSION_LR, REVERSION_STEPS, MAX_ADAPTIVE_STEPS,
    STEP_SIZE_DECAY
)
from src.utils.models import SimpleCNN, LatentEncoder, DiffusionPurifier
from src.utils.diffusion_utils import fgsm_attack, insert_trigger, kl_divergence

def evaluate_classifier(model: nn.Module, 
                        test_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate a classifier model on a test dataset.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        
    Returns:
        accuracy: Classification accuracy
    """
    model = model.to(DEVICE)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy

def test_purification_robustness(model: nn.Module, 
                                purifier: nn.Module, 
                                test_loader: torch.utils.data.DataLoader,
                                num_samples: int = 100) -> Dict[str, float]:
    """
    Test the purification robustness against adversarial perturbations.
    
    Args:
        model: Classifier model
        purifier: Diffusion purifier model
        test_loader: DataLoader for test data
        num_samples: Number of samples to test
        
    Returns:
        results: Dictionary with classification accuracies
    """
    model = model.to(DEVICE)
    purifier = purifier.to(DEVICE)
    model.eval()
    purifier.eval()
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    adv_images = fgsm_attack(images, EPSILON, data_grad)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        orig_acc = (predicted == labels).float().mean().item() * 100
        
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        adv_acc = (predicted == labels).float().mean().item() * 100
        
        purified_images = purifier(adv_images)
        outputs = model(purified_images)
        _, predicted = torch.max(outputs.data, 1)
        purified_acc = (predicted == labels).float().mean().item() * 100
    
    results = {
        "original_accuracy": orig_acc,
        "adversarial_accuracy": adv_acc,
        "purified_accuracy": purified_acc
    }
    
    print(f"Original accuracy: {orig_acc:.2f}%")
    print(f"Adversarial accuracy: {adv_acc:.2f}%")
    print(f"Purified accuracy: {purified_acc:.2f}%")
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=["Original", "Adversarial", "Purified"], 
                y=[orig_acc, adv_acc, purified_acc])
    plt.ylabel("Accuracy (%)")
    plt.title("Classifier Accuracy Before and After Purification")
    plt.ylim(0, 100)
    plt.tight_layout()
    
    os.makedirs("logs", exist_ok=True)
    plt.savefig("logs/accuracy_purification.pdf", format="pdf", dpi=300)
    plt.close()
    
    return results

def trigger_reversion(latent: torch.Tensor, 
                      benign_mean: torch.Tensor, 
                      benign_std: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
    """
    Apply trigger reversion to a latent representation.
    
    Args:
        latent: Latent representation to revert
        benign_mean: Mean of the benign latent distribution
        benign_std: Standard deviation of the benign latent distribution
        
    Returns:
        reverted_latent, divergence_list: Reverted latent representation and 
                                        list of divergence values
    """
    latent = latent.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([latent], lr=REVERSION_LR)
    divergence_list = []
    
    for i in range(REVERSION_STEPS):
        optimizer.zero_grad()
        loss = kl_divergence(latent, benign_mean, benign_std)
        loss.backward()
        optimizer.step()
        
        divergence_list.append(loss.item())
        
        if loss.item() < DIVERGENCE_THRESHOLD:
            print(f"Early stop at step {i+1} with loss {loss.item():.4f}")
            break
    
    return latent.detach(), divergence_list

def adaptive_purification(x_adv: torch.Tensor, 
                          purifier: nn.Module, 
                          encoder: nn.Module) -> Tuple[torch.Tensor, float, List[float]]:
    """
    Apply adaptive purification with parameter tuning based on latent divergence.
    
    Args:
        x_adv: Adversarial input tensor
        purifier: Diffusion purifier model
        encoder: Latent encoder model
        
    Returns:
        purified, final_div, divergence_list: Purified image, final divergence value,
                                             and list of divergence values
    """
    with torch.no_grad():
        latent_initial = encoder(x_adv)
    
    benign_mean = torch.zeros(latent_initial.size(1)).to(DEVICE)
    benign_std = torch.ones(latent_initial.size(1)).to(DEVICE)
    
    current_step_size = purifier.step_size
    divergence_list = []
    
    purified = purifier(x_adv)
    latent_purified = encoder(purified)
    initial_div = kl_divergence(latent_initial, benign_mean, benign_std)
    purified_div = kl_divergence(latent_purified, benign_mean, benign_std)
    divergence = abs(purified_div - initial_div)
    divergence_list.append(divergence.item())
    print(f"Initial divergence difference: {divergence.item():.4f}")
    
    adaptive_steps = 0
    while divergence.item() > DIVERGENCE_THRESHOLD and adaptive_steps < MAX_ADAPTIVE_STEPS:
        current_step_size *= STEP_SIZE_DECAY
        purifier.step_size = current_step_size
        print(f"Adaptive tuning (step {adaptive_steps+1}): new step size = {current_step_size:.4f}")
        
        purified = purifier(x_adv)
        latent_purified = encoder(purified)
        purified_div = kl_divergence(latent_purified, benign_mean, benign_std)
        divergence = abs(purified_div - initial_div)
        divergence_list.append(divergence.item())
        adaptive_steps += 1
    
    return purified, divergence.item(), divergence_list
