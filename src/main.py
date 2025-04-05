"""
DALWGAN Main Experiment Script

This script implements three experiments:
  1. Synthetic Data Analysis for Intrinsic Dimension Recovery
  2. Quality Evaluation on Real-World Image Datasets
  3. Ablation Study on the Diffusion Purification Stage

Each experiment prints key results to stdout and saves plots as .pdf files.
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from preprocess import generate_synthetic_data, load_mnist
from train import DALWGAN, Encoder, DiffusionPurification, Generator, Discriminator
from evaluate import plot_samples, plot_latent_visualization, compute_intrinsic_dimension

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOG_DIR = './logs'
MODEL_DIR = './models'

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, 'figures'), exist_ok=True)

def run_experiment1(test_mode=False):
    """
    Experiment 1: Synthetic Data Analysis for Intrinsic Dimension Recovery
    """
    print("\n" + "="*80)
    print("Running Experiment 1: Synthetic Data Analysis for Intrinsic Dimension Recovery")
    print("="*80)
    
    config = {
        'dataset': 'swiss_roll',
        'n_samples': 1500,
        'noise': 0.1,
        'latent_dim': 2,
        'diffusion_steps': 10,
        'integration_method': 'heun',
        'epochs': 100,
        'batch_size': 64,
        'lr': 0.0002,
        'lambda_rank': 0.01,
        'lambda_diff': 0.1,
    }
    
    print(f"Generating {config['dataset']} dataset with {config['n_samples']} samples...")
    data = generate_synthetic_data(
        dataset=config['dataset'],
        n_samples=config['n_samples'],
        noise=config['noise']
    )
    print(f"Generated dataset with shape: {data.shape}")
    
    plt.figure()
    plt.scatter(data[:, 0], data[:, 2], c=np.linspace(0, 1, data.shape[0]), cmap='viridis')
    plt.title(f"{config['dataset'].capitalize()} Dataset")
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(os.path.join(LOG_DIR, "figures", "synthetic_data.pdf"), format='pdf', dpi=300)
    plt.close()
    print(f"Saved plot: synthetic_data.pdf")
    
    data_tensor = torch.from_numpy(data)
    dataset = TensorDataset(data_tensor, torch.zeros(data_tensor.size(0)))  # Dummy labels
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    print("Creating DALWGAN model...")
    model = DALWGAN(
        input_dim=data.shape[1],
        latent_dim=config['latent_dim'],
        diffusion_steps=config['diffusion_steps'],
        integration_method=config['integration_method'],
        lambda_rank=config['lambda_rank'],
        lambda_diff=config['lambda_diff'],
        device=DEVICE
    )
    
    epochs = 5 if test_mode else config['epochs']
    print(f"Training DALWGAN for {epochs} epochs...")
    
    losses = {'g_loss': [], 'd_loss': [], 'rank_penalty': [], 'diff_loss': []}
    
    for epoch in tqdm(range(epochs)):
        epoch_losses = {'g_loss': 0, 'd_loss': 0, 'rank_penalty': 0, 'diff_loss': 0}
        num_batches = 0
        
        for batch_data, _ in dataloader:
            step_losses = model.train_step(batch_data)
            
            for k, v in step_losses.items():
                epoch_losses[k] += v
            
            num_batches += 1
            
            if test_mode and num_batches >= 3:
                break
                
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            losses[k].append(epoch_losses[k])
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - " +
                  f"G Loss: {epoch_losses['g_loss']:.4f}, " +
                  f"D Loss: {epoch_losses['d_loss']:.4f}, " +
                  f"Rank: {epoch_losses['rank_penalty']:.4f}, " +
                  f"Diff: {epoch_losses['diff_loss']:.4f}")
    
    plt.figure(figsize=(12, 8))
    for i, (loss_name, loss_values) in enumerate(losses.items()):
        plt.subplot(2, 2, i+1)
        plt.plot(range(len(loss_values)), loss_values, marker='o')
        plt.title(f'{loss_name.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "figures", "training_curves.pdf"), format='pdf', dpi=300)
    plt.close()
    print(f"Saved plot: training_curves.pdf")
    
    print("Analyzing latent space...")
    with torch.no_grad():
        z = model.encoder(data_tensor.to(DEVICE))
        z_purified = model.diffusion(z)
        
        z_np = z.cpu().numpy()
        z_purified_np = z_purified.cpu().numpy()
        
        U, S, V = np.linalg.svd(z_np, full_matrices=False)
        print("Singular values of latent representations:", S)
        
        plt.figure()
        plt.plot(np.arange(len(S)), S, marker='o')
        plt.title('Singular Values of Latent Representations')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.grid(True)
        plt.savefig(os.path.join(LOG_DIR, "figures", "singular_values.pdf"), format='pdf', dpi=300)
        plt.close()
        print(f"Saved plot: singular_values.pdf")
        
        plot_latent_visualization(
            z_np, 
            method='pca', 
            filename=os.path.join(LOG_DIR, "figures", "latent_pca.pdf")
        )
        print(f"Saved plot: latent_pca.pdf")
        
        plot_latent_visualization(
            z_np, 
            method='tsne', 
            perplexity=30 if not test_mode else 5,
            filename=os.path.join(LOG_DIR, "figures", "latent_tsne.pdf")
        )
        print(f"Saved plot: latent_tsne.pdf")
        
        intrinsic_dim = compute_intrinsic_dimension(z_np)
        print(f"Estimated intrinsic dimension: {intrinsic_dim:.2f}")
    
    model.save_models(os.path.join(MODEL_DIR, "exp1"))
    print("Model saved to", os.path.join(MODEL_DIR, "exp1"))
    
    print("Experiment 1 finished.\n")

def run_experiment2(test_mode=False):
    """
    Experiment 2: Quality Evaluation on Real-World Image Datasets (MNIST)
    """
    print("\n" + "="*80)
    print("Running Experiment 2: Quality Evaluation on Real-World Image Datasets")
    print("="*80)
    
    config = {
        'dataset': 'mnist',
        'img_size': 64,
        'latent_dim': 100,
        'diffusion_steps': 10,
        'integration_method': 'heun',
        'epochs': 20,
        'batch_size': 64,
        'lr': 0.0002,
        'lambda_rank': 0.01,
        'lambda_diff': 0.1,
    }
    
    print(f"Loading {config['dataset']} dataset...")
    dataloader = load_mnist(batch_size=config['batch_size'], img_size=config['img_size'])
    print(f"Loaded dataset with {len(dataloader.dataset)} samples")
    
    print("Creating DALWGAN model for image data...")
    
    input_dim = 1 * config['img_size'] * config['img_size']
    
    model = DALWGAN(
        input_dim=input_dim,
        latent_dim=config['latent_dim'],
        diffusion_steps=config['diffusion_steps'],
        integration_method=config['integration_method'],
        lambda_rank=config['lambda_rank'],
        lambda_diff=config['lambda_diff'],
        device=DEVICE
    )
    
    epochs = 2 if test_mode else config['epochs']
    print(f"Training DALWGAN for {epochs} epochs...")
    
    losses = {'g_loss': [], 'd_loss': [], 'rank_penalty': [], 'diff_loss': []}
    
    for epoch in tqdm(range(epochs)):
        epoch_losses = {'g_loss': 0, 'd_loss': 0, 'rank_penalty': 0, 'diff_loss': 0}
        num_batches = 0
        
        for batch_data, _ in dataloader:
            batch_data = batch_data.view(batch_data.size(0), -1)
            
            step_losses = model.train_step(batch_data)
            
            for k, v in step_losses.items():
                epoch_losses[k] += v
            
            num_batches += 1
            
            if test_mode and num_batches >= 3:
                break
                
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            losses[k].append(epoch_losses[k])
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - " +
                  f"G Loss: {epoch_losses['g_loss']:.4f}, " +
                  f"D Loss: {epoch_losses['d_loss']:.4f}, " +
                  f"Rank: {epoch_losses['rank_penalty']:.4f}, " +
                  f"Diff: {epoch_losses['diff_loss']:.4f}")
    
    print("Generating samples...")
    with torch.no_grad():
        z = torch.randn(16, config['latent_dim']).to(DEVICE)
        
        z_purified = model.diffusion(z)
        
        samples = model.generator(z_purified)
        
        samples = samples.view(-1, 1, config['img_size'], config['img_size'])
        
        plt.figure(figsize=(10, 10))
        for i in range(samples.size(0)):
            plt.subplot(4, 4, i+1)
            plt.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, "figures", "mnist_generated_samples.pdf"), format='pdf', dpi=300)
        plt.close()
        print(f"Saved plot: mnist_generated_samples.pdf")
    
    model.save_models(os.path.join(MODEL_DIR, "exp2"))
    print("Model saved to", os.path.join(MODEL_DIR, "exp2"))
    
    print("Experiment 2 finished.\n")

def run_experiment3(test_mode=False):
    """
    Experiment 3: Ablation Study on Diffusion Purification Stage
    """
    print("\n" + "="*80)
    print("Running Experiment 3: Ablation Study on Diffusion Purification Stage")
    print("="*80)
    
    config = {
        'dataset': 'swiss_roll',
        'n_samples': 500,
        'noise': 0.1,
        'latent_dim': 2,
        'configurations': [
            {'num_steps': 5, 'method': 'euler'},
            {'num_steps': 5, 'method': 'heun'},
            {'num_steps': 15, 'method': 'euler'},
            {'num_steps': 15, 'method': 'heun'}
        ],
        'epochs': 10,
        'batch_size': 64,
    }
    
    print(f"Generating {config['dataset']} dataset with {config['n_samples']} samples...")
    data = generate_synthetic_data(
        dataset=config['dataset'],
        n_samples=config['n_samples'],
        noise=config['noise']
    )
    print(f"Generated dataset with shape: {data.shape}")
    
    data_tensor = torch.from_numpy(data)
    
    encoder = Encoder(input_dim=data.shape[1], latent_dim=config['latent_dim']).to(DEVICE)
    
    print("Running ablation study for different diffusion configurations...")
    
    iterations = 3 if test_mode else config['epochs']
    losses_record = {}
    
    for cfg in config['configurations']:
        key = f"steps{cfg['num_steps']}_{cfg['method']}"
        print(f"Running ablation config: {key}")
        
        purifier = DiffusionPurification(
            latent_dim=config['latent_dim'], 
            num_steps=cfg['num_steps'], 
            method=cfg['method']
        ).to(DEVICE)
        
        optimizer = optim.Adam(purifier.parameters(), lr=0.01)
        losses = []
        
        for it in range(iterations):
            optimizer.zero_grad()
            
            latent_codes = encoder(data_tensor.to(DEVICE))
            purified = purifier(latent_codes)
            
            loss = ((purified - latent_codes)**2).mean()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"  Iteration {it+1}/{iterations} - Loss: {loss.item():.4f}")
            
        losses_record[key] = losses
        print(f"Final loss for config {key}: {losses[-1]:.4f}")
    
    plt.figure(figsize=(10, 6))
    for key, losses in losses_record.items():
        plt.plot(range(iterations), losses, marker='o', label=key)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Ablation Study on Diffusion Purification Stage')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOG_DIR, "figures", "diffusion_ablation.pdf"), format='pdf', dpi=300)
    plt.close()
    print(f"Saved plot: diffusion_ablation.pdf")
    
    print("Experiment 3 finished.\n")

def main():
    """
    Main function to run all experiments
    """
    parser = argparse.ArgumentParser(description='DALWGAN Experiments')
    parser.add_argument('--test', action='store_true', help='Run in test mode with minimal iterations')
    parser.add_argument('--exp', type=int, default=0, help='Run specific experiment (1, 2, 3) or 0 for all')
    args = parser.parse_args()
    
    print("="*80)
    print("DALWGAN Experiment")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Test mode: {args.test}")
    print(f"Running experiment(s): {args.exp if args.exp > 0 else 'all'}")
    print("="*80)
    
    start_time = time.time()
    
    if args.exp == 1 or args.exp == 0:
        run_experiment1(test_mode=args.test)
    
    if args.exp == 2 or args.exp == 0:
        run_experiment2(test_mode=args.test)
    
    if args.exp == 3 or args.exp == 0:
        run_experiment3(test_mode=args.test)
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print("="*80)

if __name__ == '__main__':
    main()
