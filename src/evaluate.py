import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from train import DiffusionModel, harmonic_ansatz_update, neural_correction_update

def run_experiment(correction_module, initial_data, guidance_scale, use_hncg=True, max_steps=100, tol=1e-3):
    """
    Run iterative diffusion updates either with baseline (only harmonic ansatz)
    or the HNCG method (harmonic ansatz with pretrained neural correction).
    Returns final output, residual history, number of iterations, and runtime.
    """
    x = initial_data.clone()
    residuals = []
    start_time = time.time()
    for step in range(max_steps):
        if use_hncg:
            x_next = neural_correction_update(x, correction_module, guidance_scale)
        else:
            x_next = harmonic_ansatz_update(x, guidance_scale)
        residual = torch.norm(x_next - x).item()
        residuals.append(residual)
        x = x_next
        if residual < tol:
            print(f"Convergence reached at step {step+1} with residual {residual:.4e}")
            break
    elapsed = time.time() - start_time
    return x, residuals, step+1, elapsed

def adaptive_update(x, correction_module, guidance_scale, weight_func):
    """
    Adaptive update where weight_func computes the blending weight for the neural correction.
    """
    analytic = harmonic_ansatz_update(x, guidance_scale)
    neural_corr = correction_module(x)
    weight = weight_func(x)  # weight_func must return a tensor of appropriate shape
    return analytic + weight * neural_corr

def fixed_weight(x, weight_value=0.5):
    return torch.full((x.shape[0], 1), weight_value)

def confidence_weight(x):
    conf = torch.sigmoid(torch.norm(x, dim=1, keepdim=True))
    return conf

def gradient_variance_weight(x):
    variance = torch.var(x, dim=1, keepdim=True)
    norm_variance = (variance - variance.min()) / (variance.max() - variance.min() + 1e-6)
    return norm_variance

def run_adaptive_experiment(correction_module, initial_data, weight_func, guidance_scale=0.1, max_steps=50):
    """
    Runs an iterative update using an adaptive weighting strategy.
    Returns the residual history.
    """
    x = initial_data.clone()
    residuals = []
    for step in range(max_steps):
        x_next = adaptive_update(x, correction_module, guidance_scale, weight_func)
        residual = torch.norm(x_next - x).item()
        residuals.append(residual)
        x = x_next
    return residuals

def evaluate_model_outputs(model_guided, model_joint):
    """
    Generate evaluation grid to visualize outputs from both models.
    """
    with torch.no_grad():
        grid_x, grid_y = torch.meshgrid(torch.linspace(-5, 5, 50), torch.linspace(-5, 5, 50), indexing='ij')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        pred_guided = model_guided(grid).reshape(50, 50, -1)[:, :, 0].cpu().numpy()
        pred_joint = model_joint(grid).reshape(50, 50, -1)[:, :, 0].cpu().numpy()
    
    return pred_guided, pred_joint

def save_figure(fig, filename):
    """
    Save figure in high-quality PDF format.
    """
    os.makedirs('../logs', exist_ok=True)
    save_path = f"../logs/{filename}"
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved as {save_path}")
    plt.close(fig)

def test_experiments():
    """
    Quick tests to ensure that the code executes properly.
    Each experiment is run with reduced iterations to finish quickly.
    """
    print("\n========== Running quick tests ==========")
    torch.manual_seed(0)
    test_data = torch.randn(8, 2)
    print("Testing baseline update...")
    _, res, steps, _ = run_experiment(None, test_data, guidance_scale=0.1, use_hncg=False, max_steps=5)
    print("Baseline test completed in", steps, "steps.")

    print("Testing adaptive update...")
    dummy_module = DiffusionModel(in_features=2)
    res_adaptive = run_adaptive_experiment(dummy_module, test_data, lambda x: fixed_weight(x, 0.3), guidance_scale=0.1, max_steps=3)
    print("Adaptive update test residuals:", res_adaptive)
    
    print("Quick tests passed.")
    return True
