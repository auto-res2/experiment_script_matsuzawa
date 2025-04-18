"""
Evaluation functions for DALWGAN experiments
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import utils

sns.set(style="whitegrid")

def compute_svd_analysis(latent_representations, save_path="logs/singular_values.pdf"):
    """
    Compute SVD analysis on latent representations
    
    Args:
        latent_representations: Latent code tensor or numpy array
        save_path: Path to save the plot
        
    Returns:
        S: Singular values
    """
    if isinstance(latent_representations, torch.Tensor):
        latent_np = latent_representations.detach().cpu().numpy()
    else:
        latent_np = latent_representations
        
    U, S, V = np.linalg.svd(latent_np, full_matrices=False)
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(np.arange(len(S)), S, marker='o')
    plt.title('Singular Values of Latent Representations')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return S

def visualize_latent_space(latent_representations, method='pca', save_path="logs/latent_viz.pdf"):
    """
    Visualize latent space using dimensionality reduction
    
    Args:
        latent_representations: Latent code tensor or numpy array
        method: Method for visualization ('pca' or 'tsne')
        save_path: Path to save the plot
        
    Returns:
        transformed: Transformed data for visualization
    """
    if isinstance(latent_representations, torch.Tensor):
        latent_np = latent_representations.detach().cpu().numpy()
    else:
        latent_np = latent_representations
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA on Latent Representations'
        cmap = 'plasma'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=250, random_state=42)
        title = 't-SNE on Latent Representations'
        cmap = 'viridis'
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    transformed = reducer.fit_transform(latent_np)
    
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(
        transformed[:, 0], 
        transformed[:, 1], 
        c=np.linspace(0, 1, transformed.shape[0]), 
        cmap=cmap
    )
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return transformed

def visualize_synthetic_data(data, save_path="logs/synthetic_data.pdf"):
    """
    Visualize synthetic data (e.g., Swiss roll)
    
    Args:
        data: Data array or tensor
        save_path: Path to save the plot
    """
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data
    
    plt.figure(figsize=(10, 8), dpi=300)
    plt.scatter(
        data_np[:, 0], 
        data_np[:, 2], 
        c=np.linspace(0, 1, data_np.shape[0]), 
        cmap='viridis'
    )
    plt.title('Swiss Roll Projection')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_generated_samples(samples, nrow=4, save_path="logs/generated_samples.pdf"):
    """
    Visualize and save generated samples
    
    Args:
        samples: Generated image samples (tensor)
        nrow: Number of images per row in grid
        save_path: Path to save the visualization
    """
    grid_img = utils.make_grid(samples, nrow=nrow, normalize=True).permute(1,2,0).cpu().numpy()
    
    plt.figure(figsize=(10, 10), dpi=300)
    if grid_img.shape[-1] == 1 or len(grid_img.shape) == 2:
        plt.imshow(grid_img.squeeze(), cmap='gray')
    else:
        plt.imshow(grid_img)
    plt.title('Generated Samples')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

def plot_ablation_results(losses_record, save_path="logs/ablation_results.pdf"):
    """
    Plot ablation study results
    
    Args:
        losses_record: Dictionary of losses for different configurations
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6), dpi=300)
    
    for key, losses in losses_record.items():
        plt.plot(range(len(losses)), losses, marker='o', label=key)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title('Ablation Study on Diffusion Purification Stage')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def simulate_metrics():
    """
    Simulate evaluation metrics for demonstration (FID and Inception Score)
    
    Returns:
        metrics: Dictionary of simulated metrics
    """
    return {
        "fid": np.random.uniform(10, 20),
        "inception_score": np.random.uniform(8, 12)
    }
