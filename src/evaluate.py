"""
Scripts for evaluation.

This module implements evaluation procedures for D-DAME experiments.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.model_utils import plot_metrics

def evaluate_model(model, dataloader, T, device=None):
    """
    Evaluate a model on the provided dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader providing evaluation data
        T: Number of diffusion steps
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    losses = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            t = torch.randint(0, T, (data.size(0),), device=device).float()
            
            if isinstance(model, nn.Module) and hasattr(model, 'dmre'):
                output, risk, damping = model(data, t)
                loss = ((output - data)**2).mean()
                print(f"[Eval] Batch {batch_idx}: Loss={loss.item():.4f}, "
                      f"Risk={risk.mean().item():.4f}, Damping={damping.mean().item():.4f}")
            else:
                output = model(data, t)
                loss = ((output - data)**2).mean()
                print(f"[Eval] Batch {batch_idx}: Loss={loss.item():.4f}")
            
            losses.append(loss.item())
    
    avg_loss = sum(losses) / len(losses)
    print(f"Evaluation complete. Average loss: {avg_loss:.4f}")
    
    return {"loss": avg_loss, "all_losses": losses}

def compare_models(baseline_metrics, ddame_metrics, filename_prefix="model_comparison"):
    """
    Plot comparison metrics between baseline and D-DAME models.
    
    Args:
        baseline_metrics: Metrics from baseline model
        ddame_metrics: Metrics from D-DAME model
        filename_prefix: Prefix for output files
        
    Returns:
        None
    """
    plot_metrics({"Baseline Loss": baseline_metrics["loss"],
                  "D-DAME Loss": ddame_metrics["loss"]}, 
                 filename_prefix=f"{filename_prefix}_loss")
    
    if "grad_norm" in baseline_metrics and "grad_norm" in ddame_metrics:
        plot_metrics({"Baseline Grad Norm": baseline_metrics["grad_norm"],
                      "D-DAME Grad Norm": ddame_metrics["grad_norm"]}, 
                     filename_prefix=f"{filename_prefix}_gradnorm")
    
    if "risk" in ddame_metrics and "damping" in ddame_metrics:
        plot_metrics({"D-DAME Risk": ddame_metrics["risk"],
                      "D-DAME Damping": ddame_metrics["damping"]}, 
                     filename_prefix=f"{filename_prefix}_risk_damping")
