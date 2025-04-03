"""
Main script for running the Consistent Adaptive Purification (CAP) experiments.

This script implements the entire process from data preprocessing to model training and evaluation.
It runs three experiments to evaluate the CAP method:
1. Comparison of Adversarial Robustness Performance
2. Ablation Study of CAP Components
3. Efficiency and Resource Usage Analysis

All figures and plots are saved in high-quality PDF format suitable for academic papers.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse

from src.preprocess import load_cifar10, generate_adversarial_examples
from src.train import load_model, train_model, SimpleScoreModel, train_score_model
from src.evaluate import (
    cap_defense, purifypp_defense, baseline_defense,
    experiment1, experiment2, experiment3
)

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('config', exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CAP experiments')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with reduced data')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for data loaders')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-training', action='store_true', help='Skip training and use pretrained models')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def quick_test(device='cuda:0'):
    """Run a quick test to verify the code execution.
    
    Args:
        device: Device to run on
    """
    print("Running quick test...")
    
    train_loader, test_loader = load_cifar10(batch_size=16)
    
    model = load_model(pretrained=False)
    model = model.to(device)
    
    score_model = SimpleScoreModel()
    score_model = score_model.to(device)
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    print("Testing CAP defense...")
    purified_cap, steps, duration = cap_defense(images, score_model, max_steps=3)
    print(f"CAP defense: steps={steps}, duration={duration:.4f}s, output shape={purified_cap.shape}")
    
    print("Testing Purify++ defense...")
    purified_pp = purifypp_defense(images, steps=3)
    print(f"Purify++ defense output shape: {purified_pp.shape}")
    
    print("Testing baseline defense...")
    purified_baseline = baseline_defense(images)
    print(f"Baseline defense output shape: {purified_baseline.shape}")
    
    print("Quick test completed successfully!")

def main():
    """Main function to run the CAP experiments."""
    args = parse_args()
    
    set_seed(args.seed)
    
    setup_directories()
    
    device = args.device
    print(f"Using device: {device}")
    if device.startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    if args.quick_test:
        quick_test(device)
        return
    
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10(batch_size=args.batch_size)
    
    model_path = 'models/classifier.pth'
    if args.skip_training and os.path.exists(model_path):
        print(f"Loading pretrained classifier from {model_path}...")
        model = load_model(pretrained=False)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
    else:
        print("Training classifier model...")
        model = load_model(pretrained=True)
        model, history = train_model(
            model, train_loader, test_loader,
            epochs=args.epochs, lr=args.lr,
            device=device, save_path=model_path
        )
    
    score_model_path = 'models/score_model.pth'
    if args.skip_training and os.path.exists(score_model_path):
        print(f"Loading pretrained score model from {score_model_path}...")
        score_model = SimpleScoreModel()
        score_model.load_state_dict(torch.load(score_model_path))
        score_model = score_model.to(device)
    else:
        print("Training score model...")
        score_model = SimpleScoreModel()
        score_model = train_score_model(
            score_model, train_loader,
            epochs=max(1, args.epochs // 2),
            lr=args.lr / 10,
            device=device,
            save_path=score_model_path
        )
    
    print("Generating adversarial examples...")
    adv_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            *generate_adversarial_examples(model, test_loader, attack_type='PGD')
        ),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print("\n" + "="*50)
    print("Running CAP Experiments")
    print("="*50 + "\n")
    
    exp1_results = experiment1(model, adv_loader, device, score_model)
    
    exp2_results = experiment2(model, adv_loader, device, score_model)
    
    exp3_results = experiment3(model, adv_loader, device, score_model)
    
    print("\n" + "="*50)
    print("All experiments completed successfully!")
    print("="*50 + "\n")
    
    print("Results summary:")
    print(f"Experiment 1 - CAP Accuracy: {exp1_results['CAP']:.4f}")
    print(f"Experiment 1 - Purify++ Accuracy: {exp1_results['Purify++']:.4f}")
    print(f"Experiment 1 - Baseline Accuracy: {exp1_results['Baseline']:.4f}")
    print("\nExperiment 2 - Full CAP Accuracy: {:.4f}".format(exp2_results["Full_CAP"]["Accuracy"]))
    print("\nExperiment 3 - Accuracy at ε=8/255: {:.4f}".format(exp3_results["Accuracy"][-1]))
    
    print("\nFigures saved to:")
    print("- logs/adversarial_robustness_performance.pdf")
    print("- logs/ablation_study.pdf")
    print("- logs/inference_efficiency.pdf")

if __name__ == '__main__':
    main()
