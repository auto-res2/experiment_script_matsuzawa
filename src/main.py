"""
Main script for running UPR Defense experiments.
This script implements three experiments:
1. Robustness under various trigger intensities
2. Computational efficiency and convergence speed
3. Effect of controlled randomness and adaptive noise scheduling
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings from matplotlib or PyTorch

from config.experiment_config import (
    DEVICE, BATCH_SIZE, TRIGGER_INTENSITIES, 
    MAX_STEPS, LOSS_THRESHOLD, MIX_COEFFICIENTS, 
    NOISE_SCHEDULES, FIGURES_DIR
)
from src.preprocess import get_cifar10_loader, add_trigger
from src.train import (
    DiffusionPurifier, dual_consistency_loss, 
    terp_purification, upr_purification, 
    terp_multiplesampler_purification
)
from src.evaluate import (
    compute_psnr_ssim, upr_heun_purification, 
    upr_adaptive_purification, save_plot
)

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

def run_smoke_tests():
    """
    A test function that quickly verifies the implementation.
    """
    print("\n[Smoke Test] Running quick tests...")
    
    dummy_image = torch.rand(3, 32, 32).to(DEVICE)
    
    triggered = add_trigger(dummy_image.unsqueeze(0), intensity=0.5)
    print(f"add_trigger test passed. Shape: {triggered.shape}")
    
    purified_terp = terp_purification(dummy_image.unsqueeze(0), trigger_intensity=0.8)
    purified_upr = upr_purification(dummy_image.unsqueeze(0), trigger_intensity=0.8, iterations=3)
    print(f"Purification functions test passed. terp output mean: {purified_terp.mean().item():.4f}, "
          f"upr output mean: {purified_upr.mean().item():.4f}")
    
    psnr, ssim = compute_psnr_ssim(dummy_image, dummy_image)
    print(f"compute_psnr_ssim test passed. PSNR: {psnr:.2f}, SSIM: {ssim:.2f}")
    print("[Smoke Test] All basic tests passed.\n")

def experiment_robustness():
    """
    Experiment 1: Test robustness under various trigger intensities.
    """
    print("\n[Experiment 1] Robustness Under Various Trigger Intensities")
    testloader = get_cifar10_loader(batch_size=BATCH_SIZE, train=False)
    metrics = {'PSNR': [], 'SSIM': [], 'DetectionAccuracy': []}

    for intensity in TRIGGER_INTENSITIES:
        print(f"Testing trigger intensity: {intensity:.1f}")
        psnr_list = []
        ssim_list = []
        correct_removals = 0
        total = 0
        
        for i, (images, labels) in enumerate(testloader):
            if i >= 5:  # Limit to 5 batches for a quick run
                break
                
            images = images.to(DEVICE)
            triggered_images = add_trigger(images, intensity=intensity)
            
            purified_images = upr_purification(triggered_images, intensity)
            
            for j in range(images.size(0)):
                score_psnr, score_ssim = compute_psnr_ssim(images[j], purified_images[j])
                psnr_list.append(score_psnr)
                ssim_list.append(score_ssim)
                
                if score_psnr > 30:
                    correct_removals += 1
                total += 1
                
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        detection_accuracy = correct_removals / total if total > 0 else 0
        metrics['PSNR'].append(avg_psnr)
        metrics['SSIM'].append(avg_ssim)
        metrics['DetectionAccuracy'].append(detection_accuracy)

        print(f"Intensity {intensity:.1f}: Avg PSNR: {avg_psnr:.2f}, "
              f"Avg SSIM: {avg_ssim:.2f}, Detection Accuracy: {detection_accuracy:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(TRIGGER_INTENSITIES, metrics['PSNR'], marker='o', label='PSNR')
    plt.plot(TRIGGER_INTENSITIES, metrics['SSIM'], marker='s', label='SSIM')
    plt.xlabel('Trigger Intensity')
    plt.ylabel('Quality Metrics')
    plt.title('Reconstruction Quality vs. Trigger Intensity')
    plt.legend()
    plt.grid(True)
    
    save_plot(plt.gcf(), "reconstruction_quality_robustness")

def experiment_efficiency():
    """
    Experiment 2: Test computational efficiency and convergence speed.
    """
    print("\n[Experiment 2] Computational Efficiency and Convergence Speed")
    testloader = get_cifar10_loader(batch_size=1, train=False)
    times_terp = []
    steps_terp = []
    times_upr = []
    steps_upr = []
    
    count = 0
    for images, _ in testloader:
        image = images[0].to(DEVICE)  # Single image
        
        start_time = time.time()
        _, steps = terp_multiplesampler_purification(
            image, trigger_intensity=0.8, 
            max_steps=MAX_STEPS, loss_threshold=LOSS_THRESHOLD
        )
        elapsed_time = time.time() - start_time
        times_terp.append(elapsed_time)
        steps_terp.append(steps)
        
        start_time = time.time()
        _, steps_h = upr_heun_purification(
            image, trigger_intensity=0.8, 
            max_steps=MAX_STEPS, loss_threshold=LOSS_THRESHOLD
        )
        elapsed_time = time.time() - start_time
        times_upr.append(elapsed_time)
        steps_upr.append(steps_h)
        
        count += 1
        if count >= 10:  # Limit to 10 images for a quick run
            break

    avg_time_terp = np.mean(times_terp)
    avg_time_upr = np.mean(times_upr)
    avg_steps_terp = np.mean(steps_terp)
    avg_steps_upr = np.mean(steps_upr)

    print(f"TERD: Average time per sample: {avg_time_terp:.4f}s, Average steps: {avg_steps_terp:.1f}")
    print(f"UPR-Heun: Average time per sample: {avg_time_upr:.4f}s, Average steps: {avg_steps_upr:.1f}")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    labels = ['TERD', 'UPR-Heun']
    avg_times = [avg_time_terp, avg_time_upr]
    avg_steps = [avg_steps_terp, avg_steps_upr]

    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, avg_times, width, color='skyblue', label='Avg Time (s)')
    ax1.set_ylabel('Average Time per Sample (s)')
    ax1.set_title('Computational Efficiency Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, avg_steps, width, color='salmon', label='Avg Steps')
    ax2.set_ylabel('Average Iterations')
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    save_plot(plt.gcf(), "inference_latency_efficiency")

def experiment_adaptive_noise():
    """
    Experiment 3: Test effect of controlled randomness and adaptive noise scheduling.
    """
    print("\n[Experiment 3] Effect of Controlled Randomness and Adaptive Noise Scheduling")
    testloader = get_cifar10_loader(batch_size=1, train=False)
    results = {}
    
    for mix in MIX_COEFFICIENTS:
        for schedule in NOISE_SCHEDULES:
            loss_curves_agg = []
            count = 0
            
            for images, _ in testloader:
                image = images[0].to(DEVICE)
                _, loss_history = upr_adaptive_purification(
                    image, trigger_intensity=0.8, 
                    mix_coef=mix, noise_schedule=schedule, 
                    max_steps=20  # Reduced for a quick run
                )
                loss_curves_agg.append(loss_history)
                count += 1
                if count >= 5:  # Limit to 5 images per config for a quick run
                    break
                    
            avg_loss = np.mean(loss_curves_agg, axis=0)
            results[(mix, schedule)] = avg_loss
            
            print(f"Mix coef: {mix:.1f}, Noise schedule: {schedule} --> "
                  f"Final Consistency Loss: {avg_loss[-1]:.4f}")

    plt.figure(figsize=(12, 8))
    
    for key, losses in results.items():
        label_str = f"Mix={key[0]:.1f}/{key[1]}"
        plt.plot(losses, label=label_str)
        
    plt.xlabel("Iteration")
    plt.ylabel("Consistency Loss")
    plt.title("Consistency Loss vs. Iteration for Different Settings")
    plt.legend()
    plt.grid(True)
    
    save_plot(plt.gcf(), "consistency_loss_adaptive")

def main():
    """
    Main function to run all experiments.
    """
    print("Starting UPR Defense Experiments")
    print("=================================")
    print(f"Using device: {DEVICE}")
    
    create_directories()
    
    run_smoke_tests()
    
    experiment_robustness()
    
    experiment_efficiency()
    
    experiment_adaptive_noise()
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    main()
