"""
Main script for running video super-resolution experiments.
"""

import os
import torch
import yaml
import numpy as np
from preprocess import preprocess_data
from train import train_model
from evaluate import experiment1, experiment2, experiment3

def main():
    """
    Main function for running video super-resolution experiments.
    """
    print("Starting ATRD Video Super-Resolution Experiments")
    
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    config_path = 'config/atrd_config.yaml'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n=== Data Preprocessing ===")
    print(f"Loading configuration from {config_path}")
    print(f"Data parameters: {config['data']}")
    dataloader = preprocess_data(config_path)
    print("Data preprocessing complete")
    
    print("\n=== Model Initialization ===")
    print("Initializing StableVSR (baseline model)...")
    baseline_model = train_model('StableVSR', dataloader, config_path)
    print(f"Model parameters: {sum(p.numel() for p in baseline_model.parameters())} total parameters")
    
    print("\nInitializing ATRD model with OTAR module...")
    atrd_model = train_model('ATRD', dataloader, config_path)
    print(f"Diffusion steps: {atrd_model.diffusion_steps}")
    print(f"OTAR enabled: {atrd_model.otar_enabled}")
    print(f"Model parameters: {sum(p.numel() for p in atrd_model.parameters())} total parameters")
    
    print("\nInitializing ATRD model without OTAR module...")
    atrd_no_otar_model = train_model('ATRD_NoOTAR', dataloader, config_path)
    print(f"Diffusion steps: {atrd_no_otar_model.diffusion_steps}")
    print(f"OTAR enabled: {atrd_no_otar_model.otar_enabled}")
    print(f"Model parameters: {sum(p.numel() for p in atrd_no_otar_model.parameters())} total parameters")
    
    print("\n=== Moving Models to Device ===")
    print(f"Using device: {device}")
    baseline_model = baseline_model.to(device)
    atrd_model = atrd_model.to(device)
    atrd_no_otar_model = atrd_no_otar_model.to(device)
    print(f"All models successfully moved to {device}")
    
    print("\n=== Running Experiments ===")
    print("Experiment 1: Performance and Quality Benchmarking Against Baselines")
    print("This experiment compares StableVSR (baseline) with ATRD on metrics including PSNR, SSIM, LPIPS, and temporal consistency")
    print("Starting experiment 1...")
    exp1_results = experiment1(dataloader, baseline_model, atrd_model, device)
    
    print("\nExperiment 2: Ablation Study on OTAR Module")
    print("This experiment compares ATRD with and without the OTAR module to evaluate its contribution")
    print("Starting experiment 2...")
    exp2_results = experiment2(dataloader, atrd_model, atrd_no_otar_model, device)
    
    print("\nExperiment 3: Diffusion Efficiency Evaluation")
    print("This experiment evaluates how different diffusion step counts affect quality and computation time")
    diffusion_steps_list = config['experiment']['diffusion_steps_list']
    print(f"Testing diffusion steps: {diffusion_steps_list}")
    print("Starting experiment 3...")
    exp3_results = experiment3(dataloader, atrd_model, diffusion_steps_list, device)
    print(f"Experiment 3 results: {exp3_results}")
    
    print("\n=== Summary of Results ===")
    
    print("\n--- Experiment 1: Performance Comparison ---")
    avg_psnr_bl = np.mean([res['psnr_bl'] for res in exp1_results])
    avg_psnr_atrd = np.mean([res['psnr_atrd'] for res in exp1_results])
    avg_ssim_bl = np.mean([res['ssim_bl'] for res in exp1_results])
    avg_ssim_atrd = np.mean([res['ssim_atrd'] for res in exp1_results])
    avg_lpips_bl = np.mean([res['lpips_bl'] for res in exp1_results])
    avg_lpips_atrd = np.mean([res['lpips_atrd'] for res in exp1_results])
    avg_time_bl = np.mean([res['time_bl'] for res in exp1_results])
    avg_time_atrd = np.mean([res['time_atrd'] for res in exp1_results])
    
    print("Quality Metrics:")
    print(f"  Baseline PSNR: {avg_psnr_bl:.2f} dB | ATRD PSNR: {avg_psnr_atrd:.2f} dB | Improvement: {avg_psnr_atrd - avg_psnr_bl:.2f} dB")
    print(f"  Baseline SSIM: {avg_ssim_bl:.4f} | ATRD SSIM: {avg_ssim_atrd:.4f} | Improvement: {avg_ssim_atrd - avg_ssim_bl:.4f}")
    print(f"  Baseline LPIPS: {avg_lpips_bl:.4f} | ATRD LPIPS: {avg_lpips_atrd:.4f} | Improvement: {avg_lpips_bl - avg_lpips_atrd:.4f} (lower is better)")
    
    print("\nPerformance Metrics:")
    print(f"  Baseline Time: {avg_time_bl:.4f}s | ATRD Time: {avg_time_atrd:.4f}s")
    print(f"  Time Overhead: {(avg_time_atrd / avg_time_bl - 1) * 100:.2f}%")
    if torch.cuda.is_available():
        print(f"  GPU Memory Usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    
    print("\n--- Experiment 2: Ablation Study on OTAR Module ---")
    avg_psnr_full = np.mean([res['metrics_full']['psnr'] for res in exp2_results])
    avg_psnr_no_otar = np.mean([res['metrics_no_otar']['psnr'] for res in exp2_results])
    avg_ssim_full = np.mean([res['metrics_full']['ssim'] for res in exp2_results])
    avg_ssim_no_otar = np.mean([res['metrics_no_otar']['ssim'] for res in exp2_results])
    
    avg_iter_full = np.mean([res['iterations_full'] for res in exp2_results])
    avg_iter_no_otar = np.mean([res['iterations_no_otar'] for res in exp2_results])
    
    print("Quality Metrics:")
    print(f"  Full ATRD PSNR: {avg_psnr_full:.2f} dB | No OTAR PSNR: {avg_psnr_no_otar:.2f} dB | Difference: {avg_psnr_full - avg_psnr_no_otar:.2f} dB")
    print(f"  Full ATRD SSIM: {avg_ssim_full:.4f} | No OTAR SSIM: {avg_ssim_no_otar:.4f} | Difference: {avg_ssim_full - avg_ssim_no_otar:.4f}")
    
    print("\nEfficiency Metrics:")
    print(f"  Full ATRD Iterations: {avg_iter_full:.1f} | No OTAR Iterations: {avg_iter_no_otar:.1f}")
    print(f"  Iteration Reduction: {(avg_iter_no_otar - avg_iter_full) / avg_iter_no_otar * 100:.2f}%")
    
    print("\n--- Experiment 3: Diffusion Efficiency Evaluation ---")
    print("  Diffusion steps tested: " + ", ".join([str(step) for step in exp3_results['diffusion_steps_list']]))
    print("  PSNR results: " + ", ".join([f"{psnr:.2f} dB" for psnr in exp3_results['psnr_results']]))
    print("  Computation times: " + ", ".join([f"{time:.4f}s" for time in exp3_results['computation_times']]))
    print("\n  See generated plots in logs/diffusion_psnr.pdf and logs/diffusion_time.pdf")
    print("\nExperiment complete! ATRD model successfully evaluated.")

if __name__ == "__main__":
    main()
