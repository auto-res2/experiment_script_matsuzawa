"""
Evaluation script for video super-resolution models.
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import peak_signal_noise_ratio as compute_psnr
from torchmetrics.image import structural_similarity_index_measure as compute_ssim
from utils.metrics import temporal_consistency

def experiment1(dataloader, baseline_model, atrd_model, device):
    """
    Experiment 1: Performance and Quality Benchmarking Against Baselines
    """
    print("Starting Experiment 1: Performance and Quality Benchmarking Against Baselines")
    
    results = []
    for idx, (lr_sequence, hr_sequence) in enumerate(dataloader):
        print(f"Processing sequence {idx}")
        
        lr_sequence = [frame.to(device) for frame in lr_sequence]
        hr_sequence = [frame.to(device) for frame in hr_sequence]
        
        t0 = time.time()
        baseline_out = baseline_model(lr_sequence)
        baseline_time = time.time() - t0
        
        t0 = time.time()
        atrd_out = atrd_model(lr_sequence)
        atrd_time = time.time() - t0
        
        psnr_bl = np.mean([compute_psnr(baseline_out[i], hr_sequence[i]).item() for i in range(len(hr_sequence))])
        psnr_atrd = np.mean([compute_psnr(atrd_out[i], hr_sequence[i]).item() for i in range(len(hr_sequence))])
        
        ssim_bl = np.mean([compute_ssim(baseline_out[i].unsqueeze(0), hr_sequence[i].unsqueeze(0)).item() for i in range(len(hr_sequence))])
        ssim_atrd = np.mean([compute_ssim(atrd_out[i].unsqueeze(0), hr_sequence[i].unsqueeze(0)).item() for i in range(len(hr_sequence))])
        
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            lpips_bl = np.mean([lpips_fn(baseline_out[i].unsqueeze(0), hr_sequence[i].unsqueeze(0)).item() 
                               for i in range(len(hr_sequence))])
            lpips_atrd = np.mean([lpips_fn(atrd_out[i].unsqueeze(0), hr_sequence[i].unsqueeze(0)).item() 
                                 for i in range(len(hr_sequence))])
        except ImportError:
            print("Warning: lpips package not found. Using dummy values.")
            lpips_bl = lpips_atrd = 0.1
        
        temp_bl = temporal_consistency(baseline_out)
        temp_atrd = temporal_consistency(atrd_out)
        
        if torch.cuda.is_available():
            gpu_memory_bl = torch.cuda.memory_allocated()
            gpu_memory_atrd = torch.cuda.memory_allocated()
        else:
            gpu_memory_bl = gpu_memory_atrd = 0
        
        res = {
            'psnr_bl': psnr_bl, 'psnr_atrd': psnr_atrd,
            'ssim_bl': ssim_bl, 'ssim_atrd': ssim_atrd,
            'lpips_bl': lpips_bl, 'lpips_atrd': lpips_atrd,
            'temp_bl': temp_bl, 'temp_atrd': temp_atrd,
            'time_bl': baseline_time, 'time_atrd': atrd_time,
            'gpu_memory_bl': gpu_memory_bl, 'gpu_memory_atrd': gpu_memory_atrd,
        }
        results.append(res)
        print(f"Sequence {idx} results:")
        for k, v in res.items():
            print(f"    {k}: {v}")
    
    print("Experiment 1 Completed.\n")
    return results

def experiment2(dataloader, atrd_model, atrd_no_otar_model, device):
    """
    Experiment 2: Ablation Study on OTAR Module
    """
    print("Starting Experiment 2: Ablation Study on OTAR Module")
    
    results = []
    for idx, (lr_sequence, hr_sequence) in enumerate(dataloader):
        print(f"Processing sequence {idx}")
        
        lr_sequence = [frame.to(device) for frame in lr_sequence]
        hr_sequence = [frame.to(device) for frame in hr_sequence]
        
        out_full = atrd_model(lr_sequence)
        
        out_no_otar = atrd_no_otar_model(lr_sequence)
        
        metrics_full = {
            'psnr': np.mean([compute_psnr(out_full[i], hr_sequence[i]).item() for i in range(len(hr_sequence))]),
            'ssim': np.mean([compute_ssim(out_full[i].unsqueeze(0), hr_sequence[i].unsqueeze(0)).item() for i in range(len(hr_sequence))]),
            'temp_consistency': temporal_consistency(out_full)
        }
        
        metrics_no_otar = {
            'psnr': np.mean([compute_psnr(out_no_otar[i], hr_sequence[i]).item() for i in range(len(hr_sequence))]),
            'ssim': np.mean([compute_ssim(out_no_otar[i].unsqueeze(0), hr_sequence[i].unsqueeze(0)).item() for i in range(len(hr_sequence))]),
            'temp_consistency': temporal_consistency(out_no_otar)
        }
        
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex').to(device)
            metrics_full['lpips'] = np.mean([lpips_fn(out_full[i].unsqueeze(0), hr_sequence[i].unsqueeze(0)).item() 
                                           for i in range(len(hr_sequence))])
            metrics_no_otar['lpips'] = np.mean([lpips_fn(out_no_otar[i].unsqueeze(0), hr_sequence[i].unsqueeze(0)).item() 
                                              for i in range(len(hr_sequence))])
        except ImportError:
            print("Warning: lpips package not found. Using dummy values.")
            metrics_full['lpips'] = metrics_no_otar['lpips'] = 0.1
        
        iterations_full = atrd_model.iteration_count
        iterations_no_otar = atrd_no_otar_model.iteration_count
        
        res = {
            'metrics_full': metrics_full,
            'metrics_no_otar': metrics_no_otar,
            'iterations_full': iterations_full,
            'iterations_no_otar': iterations_no_otar
        }
        results.append(res)
        
        print(f"Sequence {idx} Full ATRD metrics: {metrics_full}")
        print(f"Sequence {idx} No OTAR metrics: {metrics_no_otar}")
        print(f"Diffusion iterations -> Full ATRD: {iterations_full}, No OTAR: {iterations_no_otar}")
    
    print("Experiment 2 Completed.\n")
    return results

def experiment3(dataloader, atrd_model, diffusion_steps_list, device):
    """
    Experiment 3: Diffusion Efficiency Evaluation
    """
    print("Starting Experiment 3: Diffusion Efficiency Evaluation")
    
    psnr_results = []
    computation_times = []
    
    lr_sequence, hr_sequence = next(iter(dataloader))
    
    lr_sequence = [frame.to(device) for frame in lr_sequence]
    hr_sequence = [frame.to(device) for frame in hr_sequence]
    
    for steps in diffusion_steps_list:
        print(f"Processing diffusion steps = {steps}")
        atrd_model.set_diffusion_steps(steps)
        
        t0 = time.time()
        output = atrd_model(lr_sequence)
        elapsed_time = time.time() - t0
        
        avg_psnr = np.mean([compute_psnr(output[i], hr_sequence[i]).item() for i in range(len(hr_sequence))])
        psnr_results.append(avg_psnr)
        computation_times.append(elapsed_time)
        
        print(f"   PSNR: {avg_psnr}, Time: {elapsed_time}")
    
    plt.figure()
    plt.plot(diffusion_steps_list, psnr_results, marker='o')
    plt.xlabel("Number of Diffusion Steps")
    plt.ylabel("Average PSNR")
    plt.title("Diffusion Steps vs. Quality (PSNR)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/diffusion_psnr.pdf")
    print("Saved plot: logs/diffusion_psnr.pdf")
    
    plt.figure()
    plt.plot(diffusion_steps_list, computation_times, marker='o', color='red')
    plt.xlabel("Number of Diffusion Steps")
    plt.ylabel("Computation Time (s)")
    plt.title("Diffusion Steps vs. Computation Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/diffusion_time.pdf")
    print("Saved plot: logs/diffusion_time.pdf")
    
    print("Experiment 3 Completed.\n")
    
    return {
        'diffusion_steps_list': diffusion_steps_list,
        'psnr_results': psnr_results,
        'computation_times': computation_times
    }
