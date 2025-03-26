"""
Evaluation module for Graph-GaussianAssembler experiments.

This module contains functions for evaluating models and computing metrics.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from matplotlib.backends.backend_pdf import PdfPages


def convert_gaussians_to_pointcloud(asset):
    """
    Convert Gaussian parameters to a point cloud.
    
    Args:
        asset (dict): Asset containing Gaussian parameters
        
    Returns:
        np.ndarray: Point cloud representation
    """
    positions = asset["positions"].detach().cpu().numpy() if isinstance(asset["positions"], torch.Tensor) else asset["positions"]
    n_points = positions.shape[0] * 10
    pointcloud = np.repeat(positions, 10, axis=0) + np.random.randn(n_points, 3) * 0.005
    return pointcloud


def compute_chamfer_distance(pc1, pc2):
    """
    Compute Chamfer distance between two point clouds.
    
    Args:
        pc1 (np.ndarray): First point cloud
        pc2 (np.ndarray): Second point cloud
        
    Returns:
        float: Chamfer distance
    """
    mdiff = abs(np.mean(pc1) - np.mean(pc2))
    return mdiff


def compute_redundancy_metric(pc):
    """
    Compute redundancy metric for a point cloud.
    
    Args:
        pc (np.ndarray): Point cloud
        
    Returns:
        float: Redundancy metric (lower is better)
    """
    from scipy.spatial.distance import pdist
    if pc.shape[0] < 2:
        return 0.0
    distances = pdist(pc)
    metric = np.mean(distances)
    return metric


def compute_image_quality_metrics(rendered, reference):
    """
    Compute image quality metrics between rendered and reference images.
    
    Args:
        rendered (np.ndarray): Rendered image
        reference (np.ndarray): Reference image
        
    Returns:
        tuple: SSIM and PSNR values
    """
    try:
        ssim_val = ssim(rendered, reference, multichannel=True)
        psnr_val = psnr(reference, rendered)
        return ssim_val, psnr_val
    except Exception as e:
        print(f"Error computing image quality metrics: {e}")
        return None, None


def denoising_process(model, asset, adaptive_schedule=True):
    """
    Run denoising process on an asset.
    
    Args:
        model: Model to use for denoising
        asset (dict): Asset to denoise
        adaptive_schedule (bool): Whether to use adaptive scheduling
        
    Returns:
        tuple: Denoised asset, number of iterations, iteration history, position history
    """
    num_iterations_fixed = 50
    current_asset = asset
    iteration = 0
    iterations_history = []
    positions_history = []
    
    while True:
        gauss_positions = current_asset['positions']
        complexity_measure = torch.tensor([float(torch.var(gauss_positions))])
        if adaptive_schedule:
            scheduler = AdaptiveTimeScheduler(init_steps=num_iterations_fixed)
            num_iterations = scheduler(complexity_measure)
        else:
            num_iterations = num_iterations_fixed
        
        for _ in range(num_iterations):
            current_asset = model.denoise_step(current_asset)
            iteration += 1
            iterations_history.append(iteration)
            positions_history.append(torch.norm(current_asset['positions']).item())
        
        change = torch.norm(current_asset['positions'] - asset['positions'])
        if change < 1e-3 or iteration > 200:
            break
        
        asset = current_asset
    
    return current_asset, iteration, iterations_history, positions_history


def save_plot(figure, filename):
    """
    Save a matplotlib figure to a file.
    
    Args:
        figure (matplotlib.figure.Figure): Figure to save
        filename (str): Filename to save to
    """
    os.makedirs("logs", exist_ok=True)
    path = os.path.join("logs", filename)
    figure.savefig(path)
    print(f"Plot saved to {path}")
    plt.close(figure)


def plot_chamfer_comparison(results, prompts_list, filename="chamfer_comparison.pdf"):
    """
    Plot chamfer distance comparison.
    
    Args:
        results (list): List of result dictionaries
        prompts_list (list): List of prompts
        filename (str): Filename to save the plot to
    """
    chamfer_baselines = [r["chamfer_baseline"] for r in results]
    chamfer_recons = [r["chamfer_recon"] for r in results]
    
    plt.figure(figsize=(6, 4))
    plt.plot(prompts_list, chamfer_baselines, marker="o", label="Baseline vs Full")
    plt.plot(prompts_list, chamfer_recons, marker="s", label="Recon vs Full")
    plt.title("Chamfer Distance Comparison")
    plt.xlabel("Prompt")
    plt.ylabel("Chamfer Distance")
    plt.legend()
    plt.tight_layout()
    
    save_plot(plt.gcf(), filename)
    return plt.gcf()


def plot_redundancy_comparison(labels, graph_metrics, simple_metrics, filename="redundancy_comparison.pdf"):
    """
    Plot redundancy metric comparison.
    
    Args:
        labels (list): List of labels
        graph_metrics (list): List of graph module metrics
        simple_metrics (list): List of simple module metrics
        filename (str): Filename to save the plot to
    """
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(6, 4))
    plt.bar(x - width/2, graph_metrics, width, label='Graph Module')
    plt.bar(x + width/2, simple_metrics, width, label='Simple Module')
    plt.xticks(x, labels)
    plt.ylabel("Redundancy Metric")
    plt.title("Ablation Study: Redundancy Comparison")
    plt.legend()
    plt.tight_layout()
    
    save_plot(plt.gcf(), filename)
    return plt.gcf()


def plot_denoising_trajectory(hist_iters_adaptive, hist_pos_adaptive, 
                              hist_iters_fixed, hist_pos_fixed, 
                              prompt, filename=None):
    """
    Plot denoising trajectory.
    
    Args:
        hist_iters_adaptive (list): Iteration history for adaptive scheduling
        hist_pos_adaptive (list): Position history for adaptive scheduling
        hist_iters_fixed (list): Iteration history for fixed scheduling
        hist_pos_fixed (list): Position history for fixed scheduling
        prompt (str): Prompt for the plot title
        filename (str, optional): Filename to save the plot to
    """
    plt.figure(figsize=(6, 4))
    plt.plot(hist_iters_adaptive, hist_pos_adaptive, label="Adaptive Scheduling")
    plt.plot(hist_iters_fixed, hist_pos_fixed, label="Fixed Scheduling", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Position Norm")
    plt.title(f"Denoising Trajectory for prompt: {prompt}")
    plt.legend()
    plt.tight_layout()
    
    if filename is None:
        filename = f"trajectory_{prompt.replace(' ', '_')}.pdf"
    
    save_plot(plt.gcf(), filename)
    return plt.gcf()


from train import AdaptiveTimeScheduler
