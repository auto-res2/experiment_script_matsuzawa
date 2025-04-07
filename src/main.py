"""
Main script for ANCD (Adaptive Noise and Consistency Distillation) experiments.

This script implements three experiments:
1. Efficiency and Memory Overhead Comparison
2. Sample Quality and Stability Evaluation
3. Ablation Study: Effect of Adaptive Components
"""
import os
import torch
import numpy as np
import sys
import time

os.makedirs("logs", exist_ok=True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocess import get_dataloaders
from train import run_experiment1, train_model_for_experiment2, train_variants_for_experiment3
from evaluate import run_experiment2, run_experiment3
from utils.models import BaselineTripleNetwork, ANCDNetwork, ANCDNetworkVariants
from config.experiment_config import RANDOM_SEED, DEVICE

def test_experiments():
    """
    Run a quick test for each of the three experiments.
    The test uses a small subset of the data and one epoch with few iterations.
    """
    print("\n### Running Quick Test of All Experiments ###")
    
    torch.manual_seed(RANDOM_SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = get_dataloaders(test_mode=True)
    
    print("\nTesting Experiment 1 (ANCD Network)")
    ancd_model = ANCDNetwork()
    exp1_results_ancd = run_experiment1(ancd_model, train_loader, test_mode=True, use_ancd=True)
    
    print("\nTesting Experiment 1 (Baseline Network)")
    baseline_model = BaselineTripleNetwork()
    exp1_results_baseline = run_experiment1(baseline_model, train_loader, test_mode=True, use_ancd=False)
    
    print("\nTesting Experiment 2")
    ancd_model = ANCDNetwork()
    ancd_model = train_model_for_experiment2(ancd_model, train_loader, test_mode=True)
    exp2_results = run_experiment2(ancd_model, train_loader, test_mode=True)
    
    print("\nTesting Experiment 3")
    variants = {
        'full': ANCDNetworkVariants(use_adaptive_noise=True, use_high_order=True, use_consistency_loss=True),
        'no_adaptive_noise': ANCDNetworkVariants(use_adaptive_noise=False, use_high_order=True, use_consistency_loss=True),
        'no_high_order': ANCDNetworkVariants(use_adaptive_noise=True, use_high_order=False, use_consistency_loss=True),
        'no_consistency_loss': ANCDNetworkVariants(use_adaptive_noise=True, use_high_order=True, use_consistency_loss=False)
    }
    
    variant_losses = train_variants_for_experiment3(variants, train_loader, test_mode=True)
    run_experiment3(variant_losses)
    
    print("\nQuick test complete. Check saved .pdf figures and printed logs for details.")

def run_full_experiments():
    """
    Run the full set of experiments with complete training.
    """
    print("\n### Running Full Experiments ###")
    
    torch.manual_seed(RANDOM_SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = get_dataloaders(test_mode=False)
    
    print("\nRunning Experiment 1 (ANCD Network)")
    ancd_model = ANCDNetwork()
    exp1_results_ancd = run_experiment1(ancd_model, train_loader, test_mode=False, use_ancd=True)
    
    print("\nRunning Experiment 1 (Baseline Network)")
    baseline_model = BaselineTripleNetwork()
    exp1_results_baseline = run_experiment1(baseline_model, train_loader, test_mode=False, use_ancd=False)
    
    print("\nRunning Experiment 2")
    ancd_model = ANCDNetwork()
    ancd_model = train_model_for_experiment2(ancd_model, train_loader, test_mode=False)
    exp2_results = run_experiment2(ancd_model, train_loader, test_mode=False)
    
    print("\nRunning Experiment 3")
    variants = {
        'full': ANCDNetworkVariants(use_adaptive_noise=True, use_high_order=True, use_consistency_loss=True),
        'no_adaptive_noise': ANCDNetworkVariants(use_adaptive_noise=False, use_high_order=True, use_consistency_loss=True),
        'no_high_order': ANCDNetworkVariants(use_adaptive_noise=True, use_high_order=False, use_consistency_loss=True),
        'no_consistency_loss': ANCDNetworkVariants(use_adaptive_noise=True, use_high_order=True, use_consistency_loss=False)
    }
    
    variant_losses = train_variants_for_experiment3(variants, train_loader, test_mode=False)
    run_experiment3(variant_losses)
    
    print("\nFull experiments complete. Check saved .pdf figures and printed logs for details.")

if __name__ == "__main__":
    test_experiments()
