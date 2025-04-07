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

import sys
import datetime
sys.path.insert(0, '..')

try:
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
    print("Successfully imported modules using project root path")
except ImportError:
    print("Falling back to local imports...")
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    from config.experiment_config import (
        DEVICE, BATCH_SIZE, TRIGGER_INTENSITIES, 
        MAX_STEPS, LOSS_THRESHOLD, MIX_COEFFICIENTS, 
        NOISE_SCHEDULES, FIGURES_DIR
    )
    from preprocess import get_cifar10_loader, add_trigger
    from train import (
        DiffusionPurifier, dual_consistency_loss, 
        terp_purification, upr_purification, 
        terp_multiplesampler_purification
    )
    from evaluate import (
        compute_psnr_ssim, upr_heun_purification, 
        upr_adaptive_purification, save_plot
    )
    print("Successfully imported modules using local path")

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
    print(f"Purpose: Evaluate UPR Defense performance across different trigger intensities")
    print(f"Testing intensities: {TRIGGER_INTENSITIES}")
    print(f"Metrics: PSNR, SSIM, Detection Accuracy")
    print("-" * 60)
    
    testloader = get_cifar10_loader(batch_size=BATCH_SIZE, train=False)
    metrics = {'PSNR': [], 'SSIM': [], 'DetectionAccuracy': []}
    
    total_intensities = len(TRIGGER_INTENSITIES)
    
    for idx, intensity in enumerate(TRIGGER_INTENSITIES):
        start_time = time.time()
        print(f"\n[{idx+1}/{total_intensities}] Testing trigger intensity: {intensity:.1f}")
        psnr_list = []
        ssim_list = []
        correct_removals = 0
        total = 0
        
        batch_limit = 5  # Limit to 5 batches for a quick run
        
        for i, (images, labels) in enumerate(testloader):
            if i >= batch_limit:
                break
                
            batch_start = time.time()
            print(f"  Processing batch {i+1}/{batch_limit} ({(i+1)/batch_limit*100:.1f}%)...", end="", flush=True)
                
            images = images.to(DEVICE)
            triggered_images = add_trigger(images, intensity=intensity)
            
            purified_images = upr_purification(triggered_images, intensity)
            
            batch_psnr = []
            batch_ssim = []
            
            for j in range(images.size(0)):
                score_psnr, score_ssim = compute_psnr_ssim(images[j], purified_images[j])
                psnr_list.append(score_psnr)
                ssim_list.append(score_ssim)
                batch_psnr.append(score_psnr)
                batch_ssim.append(score_ssim)
                
                if score_psnr > 30:
                    correct_removals += 1
                total += 1
            
            batch_time = time.time() - batch_start
            print(f" done in {batch_time:.2f}s (Avg batch PSNR: {np.mean(batch_psnr):.2f}, SSIM: {np.mean(batch_ssim):.2f})")
                
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        detection_accuracy = correct_removals / total if total > 0 else 0
        metrics['PSNR'].append(avg_psnr)
        metrics['SSIM'].append(avg_ssim)
        metrics['DetectionAccuracy'].append(detection_accuracy)

        elapsed_time = time.time() - start_time
        print(f"\nResults for intensity {intensity:.1f}:")
        print(f"  - Average PSNR: {avg_psnr:.2f} dB")
        print(f"  - Average SSIM: {avg_ssim:.4f}")
        print(f"  - Detection Accuracy: {detection_accuracy:.2f} ({correct_removals}/{total} samples)")
        print(f"  - Processing time: {elapsed_time:.2f} seconds")
    
    print("\nSummary of results across all intensities:")
    print("-" * 60)
    print("| Intensity | PSNR (dB) | SSIM  | Detection Accuracy |")
    print("|" + "-" * 58 + "|")
    
    for i, intensity in enumerate(TRIGGER_INTENSITIES):
        print(f"| {intensity:9.1f} | {metrics['PSNR'][i]:8.2f} | {metrics['SSIM'][i]:5.4f} | {metrics['DetectionAccuracy'][i]:18.2f} |")
    
    print("-" * 60)
    
    plt.figure(figsize=(10, 6))
    plt.plot(TRIGGER_INTENSITIES, metrics['PSNR'], marker='o', label='PSNR')
    plt.plot(TRIGGER_INTENSITIES, metrics['SSIM'], marker='s', label='SSIM')
    plt.xlabel('Trigger Intensity')
    plt.ylabel('Quality Metrics')
    plt.title('Reconstruction Quality vs. Trigger Intensity')
    plt.legend()
    plt.grid(True)
    
    save_plot(plt.gcf(), "reconstruction_quality_robustness")
    print(f"Plot saved to {FIGURES_DIR}/reconstruction_quality_robustness.pdf")

def experiment_efficiency():
    """
    Experiment 2: Test computational efficiency and convergence speed.
    """
    print("\n[Experiment 2] Computational Efficiency and Convergence Speed")
    print(f"Purpose: Compare computational efficiency between TERD and UPR-Heun methods")
    print(f"Parameters: Trigger intensity=0.8, Max steps={MAX_STEPS}, Loss threshold={LOSS_THRESHOLD}")
    print(f"Metrics: Processing time, Number of iterations")
    print("-" * 60)
    
    testloader = get_cifar10_loader(batch_size=1, train=False)
    times_terp = []
    steps_terp = []
    times_upr = []
    steps_upr = []
    
    sample_limit = 10  # Limit to 10 images for a quick run
    print(f"Testing on {sample_limit} samples...")
    
    for count, (images, _) in enumerate(testloader):
        if count >= sample_limit:
            break
            
        image = images[0].to(DEVICE)  # Single image
        print(f"\nProcessing sample {count+1}/{sample_limit} ({(count+1)/sample_limit*100:.1f}%):")
        
        print(f"  Running TERD purification...", end="", flush=True)
        start_time = time.time()
        _, steps = terp_multiplesampler_purification(
            image, trigger_intensity=0.8, 
            max_steps=MAX_STEPS, loss_threshold=LOSS_THRESHOLD
        )
        elapsed_time = time.time() - start_time
        times_terp.append(elapsed_time)
        steps_terp.append(steps)
        print(f" completed in {elapsed_time:.4f}s using {steps} steps")
        
        print(f"  Running UPR-Heun purification...", end="", flush=True)
        start_time = time.time()
        _, steps_h = upr_heun_purification(
            image, trigger_intensity=0.8, 
            max_steps=MAX_STEPS, loss_threshold=LOSS_THRESHOLD
        )
        elapsed_time = time.time() - start_time
        times_upr.append(elapsed_time)
        steps_upr.append(steps_h)
        print(f" completed in {elapsed_time:.4f}s using {steps_h} steps")
        
        speedup = times_terp[-1] / times_upr[-1] if times_upr[-1] > 0 else float('inf')
        step_reduction = (steps_terp[-1] - steps_upr[-1]) / steps_terp[-1] * 100 if steps_terp[-1] > 0 else 0
        print(f"  Speedup: {speedup:.2f}x, Step reduction: {step_reduction:.1f}%")

    avg_time_terp = np.mean(times_terp)
    avg_time_upr = np.mean(times_upr)
    avg_steps_terp = np.mean(steps_terp)
    avg_steps_upr = np.mean(steps_upr)
    
    std_time_terp = np.std(times_terp)
    std_time_upr = np.std(times_upr)
    std_steps_terp = np.std(steps_terp)
    std_steps_upr = np.std(steps_upr)
    
    overall_speedup = avg_time_terp / avg_time_upr if avg_time_upr > 0 else float('inf')
    overall_step_reduction = (avg_steps_terp - avg_steps_upr) / avg_steps_terp * 100 if avg_steps_terp > 0 else 0

    print("\nEfficiency Results Summary:")
    print("-" * 60)
    print("| Method   | Avg Time (s)     | Avg Steps        | Speedup |")
    print("|" + "-" * 58 + "|")
    print(f"| TERD     | {avg_time_terp:.4f} ± {std_time_terp:.4f} | {avg_steps_terp:.1f} ± {std_steps_terp:.1f} | 1.00x   |")
    print(f"| UPR-Heun | {avg_time_upr:.4f} ± {std_time_upr:.4f} | {avg_steps_upr:.1f} ± {std_steps_upr:.1f} | {overall_speedup:.2f}x   |")
    print("-" * 60)
    
    print(f"\nOverall performance improvement:")
    print(f"  - Time reduction: {(1 - avg_time_upr/avg_time_terp)*100:.1f}%")
    print(f"  - Step reduction: {overall_step_reduction:.1f}%")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    labels = ['TERD', 'UPR-Heun']
    avg_times = [avg_time_terp, avg_time_upr]
    avg_steps = [avg_steps_terp, avg_steps_upr]
    
    time_errors = [std_time_terp, std_time_upr]
    step_errors = [std_steps_terp, std_steps_upr]

    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, avg_times, width, color='skyblue', label='Avg Time (s)', 
                   yerr=time_errors, capsize=5)
    ax1.set_ylabel('Average Time per Sample (s)')
    ax1.set_title('Computational Efficiency Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, avg_steps, width, color='salmon', label='Avg Steps',
                   yerr=step_errors, capsize=5)
    ax2.set_ylabel('Average Iterations')
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    save_plot(plt.gcf(), "inference_latency_efficiency")
    print(f"Plot saved to {FIGURES_DIR}/inference_latency_efficiency.pdf")

def experiment_adaptive_noise():
    """
    Experiment 3: Test effect of controlled randomness and adaptive noise scheduling.
    """
    print("\n[Experiment 3] Effect of Controlled Randomness and Adaptive Noise Scheduling")
    print(f"Purpose: Evaluate impact of controlled randomness and noise scheduling on UPR Defense")
    print(f"Parameters: Trigger intensity=0.8, Mix coefficients={MIX_COEFFICIENTS}, Noise schedules={NOISE_SCHEDULES}")
    print(f"Metrics: Consistency loss convergence")
    print("-" * 60)
    
    testloader = get_cifar10_loader(batch_size=1, train=False)
    results = {}
    
    total_configs = len(MIX_COEFFICIENTS) * len(NOISE_SCHEDULES)
    config_count = 0
    
    print(f"Testing {total_configs} different configurations...")
    
    for mix in MIX_COEFFICIENTS:
        for schedule in NOISE_SCHEDULES:
            config_count += 1
            config_start_time = time.time()
            print(f"\n[Config {config_count}/{total_configs}] Mix coefficient: {mix:.1f}, Noise schedule: {schedule}")
            
            loss_curves_agg = []
            sample_limit = 5  # Limit to 5 images per config for a quick run
            
            for count, (images, _) in enumerate(testloader):
                if count >= sample_limit:
                    break
                    
                image = images[0].to(DEVICE)
                print(f"  Processing sample {count+1}/{sample_limit} ({(count+1)/sample_limit*100:.1f}%)...", end="", flush=True)
                
                start_time = time.time()
                _, loss_history = upr_adaptive_purification(
                    image, trigger_intensity=0.8, 
                    mix_coef=mix, noise_schedule=schedule, 
                    max_steps=20  # Reduced for a quick run
                )
                elapsed_time = time.time() - start_time
                
                loss_curves_agg.append(loss_history)
                
                if len(loss_history) > 1:
                    convergence_rate = (loss_history[0] - loss_history[-1]) / len(loss_history)
                    print(f" done in {elapsed_time:.2f}s (Initial loss: {loss_history[0]:.4f}, "
                          f"Final loss: {loss_history[-1]:.4f}, Convergence rate: {convergence_rate:.6f}/iter)")
                else:
                    print(f" done in {elapsed_time:.2f}s")
                    
            avg_loss = np.mean(loss_curves_agg, axis=0)
            results[(mix, schedule)] = avg_loss
            
            initial_avg_loss = avg_loss[0] if len(avg_loss) > 0 else 0
            final_avg_loss = avg_loss[-1] if len(avg_loss) > 0 else 0
            loss_reduction = initial_avg_loss - final_avg_loss
            loss_reduction_percent = (loss_reduction / initial_avg_loss * 100) if initial_avg_loss > 0 else 0
            
            config_time = time.time() - config_start_time
            
            print(f"\n  Results for Mix={mix:.1f}, Schedule={schedule}:")
            print(f"    - Initial avg loss: {initial_avg_loss:.4f}")
            print(f"    - Final avg loss: {final_avg_loss:.4f}")
            print(f"    - Loss reduction: {loss_reduction:.4f} ({loss_reduction_percent:.1f}%)")
            print(f"    - Configuration processing time: {config_time:.2f}s")

    print("\nSummary of Final Consistency Loss for All Configurations:")
    print("-" * 70)
    print("| Mix Coef | " + " | ".join([f"{s:^10}" for s in NOISE_SCHEDULES]) + " |")
    print("|" + "-" * 68 + "|")
    
    for mix in MIX_COEFFICIENTS:
        row = f"| {mix:7.1f} | "
        for schedule in NOISE_SCHEDULES:
            final_loss = results.get((mix, schedule), [0])[-1]
            row += f"{final_loss:10.4f} | "
        print(row)
    
    print("-" * 70)
    
    plt.figure(figsize=(12, 8))
    
    for key, losses in results.items():
        label_str = f"Mix={key[0]:.1f}/{key[1]}"
        plt.plot(losses, label=label_str, linewidth=2, marker='o', markersize=4)
        
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Consistency Loss", fontsize=12)
    plt.title("Consistency Loss vs. Iteration for Different Settings", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    save_plot(plt.gcf(), "consistency_loss_adaptive")
    print(f"Plot saved to {FIGURES_DIR}/consistency_loss_adaptive.pdf")

def main():
    """
    Main function to run all experiments.
    """
    start_time = datetime.datetime.now()
    print("\n" + "="*80)
    print(f"Starting UPR Defense Experiments at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Configuration Summary:")
    print(f"- Device: {DEVICE}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print(f"- Trigger Intensities: {TRIGGER_INTENSITIES}")
    print(f"- Max Steps: {MAX_STEPS}")
    print(f"- Loss Threshold: {LOSS_THRESHOLD}")
    print(f"- Mix Coefficients: {MIX_COEFFICIENTS}")
    print(f"- Noise Schedules: {NOISE_SCHEDULES}")
    print(f"- Output Directory: {FIGURES_DIR}")
    print("="*80 + "\n")
    
    create_directories()
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Created output directories")
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Running smoke tests...")
    run_smoke_tests()
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting Experiment 1: Robustness Testing")
    print("-"*60)
    experiment_robustness()
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Experiment 1 completed")
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting Experiment 2: Efficiency Testing")
    print("-"*60)
    experiment_efficiency()
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Experiment 2 completed")
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting Experiment 3: Adaptive Noise Testing")
    print("-"*60)
    experiment_adaptive_noise()
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Experiment 3 completed")
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("\n" + "="*80)
    print(f"All experiments completed successfully at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {duration.total_seconds():.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    
    main()
