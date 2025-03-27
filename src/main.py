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
    
    print("Preprocessing data...")
    dataloader = preprocess_data(config_path)
    
    print("Training models...")
    baseline_model = train_model('StableVSR', dataloader, config_path)
    atrd_model = train_model('ATRD', dataloader, config_path)
    atrd_no_otar_model = train_model('ATRD_NoOTAR', dataloader, config_path)
    
    baseline_model = baseline_model.to(device)
    atrd_model = atrd_model.to(device)
    atrd_no_otar_model = atrd_no_otar_model.to(device)
    
    print("Running experiments...")
    
    exp1_results = experiment1(dataloader, baseline_model, atrd_model, device)
    
    exp2_results = experiment2(dataloader, atrd_model, atrd_no_otar_model, device)
    
    diffusion_steps_list = config['experiment']['diffusion_steps_list']
    _ = experiment3(dataloader, atrd_model, diffusion_steps_list, device)
    
    print("\nSummary of Results:")
    
    print("\nExperiment 1 - Performance Comparison:")
    avg_psnr_bl = np.mean([res['psnr_bl'] for res in exp1_results])
    avg_psnr_atrd = np.mean([res['psnr_atrd'] for res in exp1_results])
    avg_ssim_bl = np.mean([res['ssim_bl'] for res in exp1_results])
    avg_ssim_atrd = np.mean([res['ssim_atrd'] for res in exp1_results])
    avg_lpips_bl = np.mean([res['lpips_bl'] for res in exp1_results])
    avg_lpips_atrd = np.mean([res['lpips_atrd'] for res in exp1_results])
    avg_time_bl = np.mean([res['time_bl'] for res in exp1_results])
    avg_time_atrd = np.mean([res['time_atrd'] for res in exp1_results])
    
    print(f"  Baseline PSNR: {avg_psnr_bl:.2f}, ATRD PSNR: {avg_psnr_atrd:.2f}")
    print(f"  Baseline SSIM: {avg_ssim_bl:.4f}, ATRD SSIM: {avg_ssim_atrd:.4f}")
    print(f"  Baseline LPIPS: {avg_lpips_bl:.4f}, ATRD LPIPS: {avg_lpips_atrd:.4f}")
    print(f"  Baseline Time: {avg_time_bl:.4f}s, ATRD Time: {avg_time_atrd:.4f}s")
    
    print("\nExperiment 2 - Ablation Study:")
    avg_psnr_full = np.mean([res['metrics_full']['psnr'] for res in exp2_results])
    avg_psnr_no_otar = np.mean([res['metrics_no_otar']['psnr'] for res in exp2_results])
    avg_ssim_full = np.mean([res['metrics_full']['ssim'] for res in exp2_results])
    avg_ssim_no_otar = np.mean([res['metrics_no_otar']['ssim'] for res in exp2_results])
    
    print(f"  Full ATRD PSNR: {avg_psnr_full:.2f}, No OTAR PSNR: {avg_psnr_no_otar:.2f}")
    print(f"  Full ATRD SSIM: {avg_ssim_full:.4f}, No OTAR SSIM: {avg_ssim_no_otar:.4f}")
    
    print("\nExperiment 3 - Diffusion Efficiency:")
    print("  See generated plots in logs/diffusion_psnr.pdf and logs/diffusion_time.pdf")

if __name__ == "__main__":
    main()
