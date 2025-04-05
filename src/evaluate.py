"""
DALWGAN Evaluation Module

This module implements evaluation metrics and visualization tools for DALWGAN:
1. FID score computation (simulated)
2. Inception Score computation (simulated)
3. Visualization of latent representations
4. Evaluation of generation quality
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import seaborn as sns

def plot_samples(samples, title, filename, n_rows=4, n_cols=4):
    """
    Plot generated samples in a grid
    
    Args:
        samples (torch.Tensor): Tensor of samples (N, C, H, W)
        title (str): Plot title
        filename (str): Output filename
        n_rows (int): Number of rows in grid
        n_cols (int): Number of columns in grid
    """
    plt.figure(figsize=(10, 10))
    
    if len(samples.shape) == 4:
        for i in range(min(n_rows * n_cols, samples.shape[0])):
            plt.subplot(n_rows, n_cols, i+1)
            
            if samples.shape[1] == 1:  # Grayscale
                plt.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
            else:  # RGB
                img = samples[i].permute(1, 2, 0).cpu().numpy()
                plt.imshow(np.clip(img * 0.5 + 0.5, 0, 1))  # Denormalize
                
            plt.axis('off')
    else:  # Non-image data (e.g., synthetic data)
        if samples.shape[1] >= 2:  # At least 2D for scatter plot
            plt.scatter(samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy())
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()
    
def plot_latent_visualization(latent_codes, method='pca', perplexity=30, filename='latent_viz.pdf'):
    """
    Visualize latent space using dimensionality reduction
    
    Args:
        latent_codes (torch.Tensor or numpy.ndarray): Latent codes
        method (str): Dimensionality reduction method ('pca' or 'tsne')
        perplexity (int): Perplexity parameter for t-SNE
        filename (str): Output filename
    """
    if isinstance(latent_codes, torch.Tensor):
        latent_np = latent_codes.detach().cpu().numpy()
    else:
        latent_np = latent_codes
        
    plt.figure(figsize=(10, 8))
    
    if method == 'pca':
        pca = PCA(n_components=2)
        latent_viz = pca.fit_transform(latent_np)
        plt.scatter(latent_viz[:, 0], latent_viz[:, 1], c=np.linspace(0, 1, latent_viz.shape[0]), cmap='plasma')
        plt.title('PCA Visualization of Latent Space')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        
    elif method == 'tsne':
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
        latent_viz = tsne.fit_transform(latent_np)
        plt.scatter(latent_viz[:, 0], latent_viz[:, 1], c=np.linspace(0, 1, latent_viz.shape[0]), cmap='viridis')
        plt.title('t-SNE Visualization of Latent Space')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
    plt.colorbar(label='Sample Index (normalized)')
    plt.tight_layout()
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()
    
def compute_intrinsic_dimension(data, method='mle'):
    """
    Estimate the intrinsic dimensionality of data
    
    Args:
        data (numpy.ndarray): Input data
        method (str): Method for estimation
        
    Returns:
        float: Estimated intrinsic dimension
    """
    if method == 'mle':
        X = np.asarray(data)
        dists = pairwise_distances(X, metric='euclidean')
        np.fill_diagonal(dists, np.inf)  # Exclude self-distances
        k = 10  # Use k nearest neighbors
        
        nn_dists = np.partition(dists, k, axis=1)[:, :k]
        r1 = nn_dists[:, 0]
        r2 = nn_dists[:, 1:].mean(axis=1)
        
        d_mle = 1.0 / np.mean(np.log(r2 / r1))
        return d_mle
    else:
        raise ValueError(f"Method {method} not implemented")
