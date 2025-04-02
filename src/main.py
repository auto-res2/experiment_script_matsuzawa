"""
Main script for running the MEAB-DG experiments.
This script orchestrates the entire process from data preprocessing to model evaluation.
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import prepare_data
from train import train_multimodal_gating, train_context_splitting
from evaluate import (
    evaluate_multimodal_gating, 
    evaluate_context_splitting,
    evaluate_edge_precision_scaling
)
from utils.model_utils import set_seed, get_device

def setup_directories():
    """Create necessary directories for the experiments."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("config", exist_ok=True)

def run_experiment(args):
    """
    Run the MEAB-DG experiments.
    
    Args:
        args: Command-line arguments
    """
    setup_directories()
    
    set_seed(args.seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("\n" + "="*80)
    print("MEAB-DG: Multimodal Edge-Adapted BTLM with Dynamic Gating")
    print("="*80)
    
    print("\nPreparing data...")
    data_info = prepare_data()
    print(f"Data preparation completed. Info: {data_info}")
    
    results = {}
    
    if args.experiment == "all" or args.experiment == "1":
        if not args.eval_only:
            print("\nRunning Experiment 1: Training Dual Modal Encoder with Dynamic Gating...")
            train_results = train_multimodal_gating()
            results["experiment1_train"] = train_results
        
        print("\nEvaluating Experiment 1: Dynamic Gating Mechanism for Multimodal Fusion...")
        eval_results = evaluate_multimodal_gating()
        results["experiment1_eval"] = eval_results
    
    if args.experiment == "all" or args.experiment == "2":
        if not args.eval_only:
            print("\nRunning Experiment 2: Training Dynamic Context Splitting Model...")
            train_results = train_context_splitting()
            results["experiment2_train"] = train_results
        
        print("\nEvaluating Experiment 2: Dynamic Context Splitting for Long-Context Tasks...")
        eval_results = evaluate_context_splitting()
        results["experiment2_eval"] = eval_results
    
    if args.experiment == "all" or args.experiment == "3":
        print("\nRunning Experiment 3: Edge-Aware Dynamic Precision Scaling...")
        eval_results = evaluate_edge_precision_scaling()
        results["experiment3_eval"] = eval_results
    
    print("\n" + "="*80)
    print("MEAB-DG Experiments Summary")
    print("="*80)
    
    if "experiment1_eval" in results:
        exp1 = results["experiment1_eval"]
        print(f"\nExperiment 1 Results:")
        print(f"  Dynamic Gating Accuracy: {exp1['dynamic_accuracy']:.4f}")
        print(f"  Baseline Accuracy: {exp1['baseline_accuracy']:.4f}")
        print(f"  Improvement: {exp1['improvement']*100:.2f}%")
        print(f"  Average Text Gate: {exp1['avg_text_gate']:.4f}")
        print(f"  Average Image Gate: {exp1['avg_image_gate']:.4f}")
    
    if "experiment2_eval" in results:
        exp2 = results["experiment2_eval"]
        print(f"\nExperiment 2 Results:")
        print(f"  Dynamic Context Splitting MSE: {exp2['dynamic_mse']:.4f}")
        print(f"  Baseline MSE: {exp2['baseline_mse']:.4f}")
        print(f"  Improvement: {exp2['improvement']:.4f}")
    
    if "experiment3_eval" in results:
        exp3 = results["experiment3_eval"]
        print(f"\nExperiment 3 Results:")
        print(f"  Text-only Inference Time: {exp3['text_only_time']:.6f} seconds")
        print(f"  Multimodal Inference Time: {exp3['multimodal_time']:.6f} seconds")
        if exp3['speedup'] > 1:
            print(f"  Speedup: {exp3['speedup']:.2f}x faster")
        else:
            print(f"  Slowdown: {1/exp3['speedup']:.2f}x slower")
        
        if torch.cuda.is_available():
            print(f"  Text-only Memory Usage: {exp3['text_only_memory']:.2f} MB")
            print(f"  Multimodal Memory Usage: {exp3['multimodal_memory']:.2f} MB")
            print(f"  Memory Difference: {exp3['memory_savings']:.2f} MB")
    
    print("\nAll experiments completed successfully!")
    print(f"Results and plots saved in the 'logs' directory.")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run MEAB-DG experiments")
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="all", 
        choices=["all", "1", "2", "3"],
        help="Which experiment to run (1, 2, 3, or all)"
    )
    parser.add_argument(
        "--eval-only", 
        action="store_true",
        help="Run evaluation only (skip training)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
