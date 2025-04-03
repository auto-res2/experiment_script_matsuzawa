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

from preprocess import load_cifar10, generate_adversarial_examples
from train import load_model, train_model, SimpleScoreModel, train_score_model
from evaluate import (
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
    print("\n" + "="*70)
    print("RUNNING QUICK TEST OF CONSISTENT ADAPTIVE PURIFICATION (CAP) METHOD")
    print("="*70 + "\n")
    
    print("Step 1: Loading CIFAR-10 dataset for testing...")
    train_loader, test_loader = load_cifar10(batch_size=16)
    print(f"Successfully loaded CIFAR-10 dataset with batch size 16")
    print(f"Number of test batches: {len(test_loader)}")
    
    print("\nStep 2: Initializing ResNet-18 model...")
    model = load_model(pretrained=False)
    model = model.to(device)
    print(f"Model architecture: ResNet-18")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model device: {next(model.parameters()).device}")
    
    print("\nStep 3: Initializing score model for Tweedie estimation...")
    score_model = SimpleScoreModel()
    score_model = score_model.to(device)
    print(f"Score model parameters: {sum(p.numel() for p in score_model.parameters()):,}")
    
    print("\nStep 4: Preparing test batch...")
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    print(f"Test batch shape: {images.shape}")
    print(f"Label distribution: {torch.bincount(labels, minlength=10)}")
    
    print("\nStep 5: Testing CAP defense (our proposed method)...")
    start_time = time.perf_counter()
    purified_cap, steps, duration = cap_defense(images, score_model, max_steps=3)
    total_time = time.perf_counter() - start_time
    print(f"CAP defense completed in {total_time:.4f}s")
    print(f"Internal timing: steps={steps}, duration={duration:.4f}s")
    print(f"Output shape: {purified_cap.shape}")
    print(f"Output statistics: min={purified_cap.min().item():.4f}, max={purified_cap.max().item():.4f}, mean={purified_cap.mean().item():.4f}")
    
    print("\nStep 6: Testing Purify++ defense (baseline comparison)...")
    start_time = time.perf_counter()
    purified_pp = purifypp_defense(images, steps=3)
    pp_time = time.perf_counter() - start_time
    print(f"Purify++ defense completed in {pp_time:.4f}s")
    print(f"Output shape: {purified_pp.shape}")
    print(f"Output statistics: min={purified_pp.min().item():.4f}, max={purified_pp.max().item():.4f}, mean={purified_pp.mean().item():.4f}")
    
    print("\nStep 7: Testing baseline defense...")
    start_time = time.perf_counter()
    purified_baseline = baseline_defense(images)
    baseline_time = time.perf_counter() - start_time
    print(f"Baseline defense completed in {baseline_time:.4f}s")
    print(f"Output shape: {purified_baseline.shape}")
    print(f"Output statistics: min={purified_baseline.min().item():.4f}, max={purified_baseline.max().item():.4f}, mean={purified_baseline.mean().item():.4f}")
    
    print("\nStep 8: Comparing defense methods...")
    print(f"CAP vs Purify++ MSE: {torch.mean((purified_cap - purified_pp)**2).item():.6f}")
    print(f"CAP vs Baseline MSE: {torch.mean((purified_cap - purified_baseline)**2).item():.6f}")
    print(f"Purify++ vs Baseline MSE: {torch.mean((purified_pp - purified_baseline)**2).item():.6f}")
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")

def main():
    """Main function to run the CAP experiments."""
    print("\n" + "="*70)
    print("CONSISTENT ADAPTIVE PURIFICATION (CAP) EXPERIMENT SUITE")
    print("="*70)
    print("A novel diffusion-purification algorithm for adversarial robustness")
    print("="*70 + "\n")
    
    print("Parsing command line arguments...")
    args = parse_args()
    print(f"Arguments: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    print(f"           device={args.device}, seed={args.seed}")
    print(f"           quick_test={args.quick_test}, skip_training={args.skip_training}")
    
    print("\nSetting random seed for reproducibility...")
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")
    
    print("\nSetting up directories...")
    setup_directories()
    print("Created directories: logs/, models/, data/, config/")
    
    device = args.device
    print(f"\nUsing device: {device}")
    if device.startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    
    if args.quick_test:
        print("\nRunning quick test mode...")
        quick_test(device)
        return
    
    print("\n" + "="*70)
    print("STEP 1: LOADING CIFAR-10 DATASET")
    print("="*70)
    start_time = time.perf_counter()
    train_loader, test_loader = load_cifar10(batch_size=args.batch_size)
    data_time = time.perf_counter() - start_time
    print(f"Dataset loaded in {data_time:.2f}s")
    print(f"Training set: {len(train_loader.dataset):,} images")
    print(f"Test set: {len(test_loader.dataset):,} images")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of training batches: {len(train_loader):,}")
    print(f"Number of test batches: {len(test_loader):,}")
    
    print("\n" + "="*70)
    print("STEP 2: PREPARING CLASSIFIER MODEL")
    print("="*70)
    model_path = 'models/classifier.pth'
    if args.skip_training and os.path.exists(model_path):
        print(f"Loading pretrained classifier from {model_path}...")
        start_time = time.perf_counter()
        model = load_model(pretrained=False)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        load_time = time.perf_counter() - start_time
        print(f"Model loaded in {load_time:.2f}s")
    else:
        print("Training classifier model from scratch...")
        start_time = time.perf_counter()
        model = load_model(pretrained=True)
        print(f"Model architecture: ResNet-18")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        model, history = train_model(
            model, train_loader, test_loader,
            epochs=args.epochs, lr=args.lr,
            device=device, save_path=model_path
        )
        train_time = time.perf_counter() - start_time
        print(f"Model training completed in {train_time:.2f}s")
        print(f"Final training accuracy: {history['train_acc'][-1]:.4f}")
        print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
    
    print("\n" + "="*70)
    print("STEP 3: PREPARING SCORE MODEL FOR TWEEDIE ESTIMATION")
    print("="*70)
    score_model_path = 'models/score_model.pth'
    if args.skip_training and os.path.exists(score_model_path):
        print(f"Loading pretrained score model from {score_model_path}...")
        start_time = time.perf_counter()
        score_model = SimpleScoreModel()
        score_model.load_state_dict(torch.load(score_model_path))
        score_model = score_model.to(device)
        load_time = time.perf_counter() - start_time
        print(f"Score model loaded in {load_time:.2f}s")
    else:
        print("Training score model for Tweedie estimation...")
        start_time = time.perf_counter()
        score_model = SimpleScoreModel()
        print(f"Score model parameters: {sum(p.numel() for p in score_model.parameters()):,}")
        score_model = train_score_model(
            score_model, train_loader,
            epochs=max(1, args.epochs // 2),
            lr=args.lr / 10,
            device=device,
            save_path=score_model_path
        )
        train_time = time.perf_counter() - start_time
        print(f"Score model training completed in {train_time:.2f}s")
    
    print("\n" + "="*70)
    print("STEP 4: GENERATING ADVERSARIAL EXAMPLES")
    print("="*70)
    print("Using PGD attack with epsilon=8/255, steps=10, alpha=2/255")
    start_time = time.perf_counter()
    adv_examples, clean_labels = generate_adversarial_examples(
        model, test_loader, attack_type='PGD', 
        eps=8/255, steps=10, alpha=2/255
    )
    adv_time = time.perf_counter() - start_time
    print(f"Generated {len(adv_examples):,} adversarial examples in {adv_time:.2f}s")
    
    adv_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(adv_examples, clean_labels),
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print("\n" + "="*70)
    print("STEP 5: RUNNING EXPERIMENT 1 - ADVERSARIAL ROBUSTNESS COMPARISON")
    print("="*70)
    start_time = time.perf_counter()
    exp1_results = experiment1(model, adv_loader, device, score_model)
    exp1_time = time.perf_counter() - start_time
    print(f"Experiment 1 completed in {exp1_time:.2f}s")
    print(f"CAP Accuracy: {exp1_results['CAP']:.4f}")
    print(f"Purify++ Accuracy: {exp1_results['Purify++']:.4f}")
    print(f"Baseline Accuracy: {exp1_results['Baseline']:.4f}")
    print(f"Clean Accuracy: {exp1_results['Clean']:.4f}")
    print(f"Adversarial Accuracy (no defense): {exp1_results['Adversarial']:.4f}")
    print(f"Robustness improvement: {(exp1_results['CAP'] - exp1_results['Adversarial']) * 100:.2f}%")
    
    print("\n" + "="*70)
    print("STEP 6: RUNNING EXPERIMENT 2 - ABLATION STUDY")
    print("="*70)
    start_time = time.perf_counter()
    exp2_results = experiment2(model, adv_loader, device, score_model)
    exp2_time = time.perf_counter() - start_time
    print(f"Experiment 2 completed in {exp2_time:.2f}s")
    print("Ablation study results:")
    for variant, results in exp2_results.items():
        print(f"  {variant}: Accuracy={results['Accuracy']:.4f}, Steps={results['Steps']:.2f}, Time={results['Time']:.4f}s")
    
    print("\n" + "="*70)
    print("STEP 7: RUNNING EXPERIMENT 3 - EFFICIENCY ANALYSIS")
    print("="*70)
    start_time = time.perf_counter()
    exp3_results = experiment3(model, adv_loader, device, score_model)
    exp3_time = time.perf_counter() - start_time
    print(f"Experiment 3 completed in {exp3_time:.2f}s")
    print("Efficiency analysis results:")
    for i, eps in enumerate(exp3_results["Perturbation"]):
        print(f"  ε={eps:.6f}: Steps={exp3_results['Avg_Steps'][i]:.2f}, " +
              f"Time={exp3_results['Avg_Duration'][i]:.4f}s, " +
              f"Accuracy={exp3_results['Accuracy'][i]:.4f}")
    
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print("1. Adversarial Robustness Performance:")
    print(f"   CAP Accuracy: {exp1_results['CAP']:.4f}")
    print(f"   Purify++ Accuracy: {exp1_results['Purify++']:.4f}")
    print(f"   Baseline Accuracy: {exp1_results['Baseline']:.4f}")
    print(f"   Clean Accuracy: {exp1_results['Clean']:.4f}")
    print(f"   Adversarial Accuracy (no defense): {exp1_results['Adversarial']:.4f}")
    
    print("\n2. Ablation Study Results:")
    print(f"   Full CAP: {exp2_results['Full_CAP']['Accuracy']:.4f}")
    print(f"   Without Adaptive Consistency: {exp2_results['No_Adaptive_Consistency']['Accuracy']:.4f}")
    print(f"   Without Double-Tweedie: {exp2_results['No_Double_Tweedie']['Accuracy']:.4f}")
    print(f"   Without Adaptive Steps: {exp2_results['No_Adaptive_Steps']['Accuracy']:.4f}")
    
    print("\n3. Efficiency Analysis:")
    print(f"   Average Steps at ε=8/255: {exp3_results['Avg_Steps'][-1]:.2f}")
    print(f"   Average Duration at ε=8/255: {exp3_results['Avg_Duration'][-1]:.4f}s")
    print(f"   Accuracy at ε=8/255: {exp3_results['Accuracy'][-1]:.4f}")
    
    print("\nFigures saved to:")
    print("- logs/adversarial_robustness_performance.pdf")
    print("- logs/ablation_study.pdf")
    print("- logs/inference_efficiency.pdf")
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == '__main__':
    main()
