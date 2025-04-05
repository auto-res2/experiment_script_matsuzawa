import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchattacks import PGD
import matplotlib.pyplot as plt
import numpy as np
import time

from src.models.tcdp_models import PurifyPlusPlus, TCDP, TCDP_NoConsistency, TCDP_FixedNoise, TCDP_Adaptive
from src.models.classifier import load_classifier
from src.preprocess import get_testloader
from config.experiment_config import EXPERIMENT_CONFIG

def experiment1(num_batches=2):
    """
    Experiment 1: Robustness Under Adaptive Adversarial Attacks
    
    Tests the robustness of Purify++ and TCDP methods against PGD attacks
    with different epsilon values.
    """
    print("Running Experiment 1: Robustness Under Adaptive Adversarial Attacks")
    
    os.makedirs('./logs', exist_ok=True)
    
    testloader = get_testloader(batch_size=EXPERIMENT_CONFIG['test_batch_size'])
    classifier = load_classifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    purify_pp = PurifyPlusPlus().to(device)
    tc_dp = TCDP().to(device)
    
    try:
        purify_pp.load_state_dict(torch.load('./models/purify_pp.pth', map_location=device))
        tc_dp.load_state_dict(torch.load('./models/tcdp.pth', map_location=device))
        print("Loaded pretrained models.")
    except Exception as e:
        print(f"Warning: Pretrained models not found ({e}). Using randomly initialized models for demonstration.")
    
    eps_list = EXPERIMENT_CONFIG['eps_list']
    
    criterion = nn.MSELoss()
    
    results = {}
    for eps in eps_list:
        results[eps] = {'purify_pp': {'acc': [], 'mse': []},
                        'tcdp': {'acc': [], 'mse': [], 'consistency': []}}
    
    for eps in eps_list:
        print(f"\nUsing PGD attack with epsilon = {eps:.4f}")
        attack = PGD(classifier, eps=eps, alpha=2/255, steps=10)
    
        batch_count = 0
        for batch_idx, (images, labels) in enumerate(testloader):
            if batch_count >= num_batches:
                break
            images, labels = images.to(device), labels.to(device)
            
            adv_images = attack(images, labels)
            
            purified_pp = purify_pp(adv_images)
            output_pp = classifier(purified_pp)
            _, pred_pp = torch.max(output_pp, 1)
            acc_pp = (pred_pp == labels).float().mean().item()
            mse_pp = criterion(purified_pp, images).item()
            
            purified_tc, consistency_loss = tc_dp(adv_images)
            output_tc = classifier(purified_tc)
            _, pred_tc = torch.max(output_tc, 1)
            acc_tc = (pred_tc == labels).float().mean().item()
            mse_tc = criterion(purified_tc, images).item()
            
            results[eps]['purify_pp']['acc'].append(acc_pp)
            results[eps]['purify_pp']['mse'].append(mse_pp)
            results[eps]['tcdp']['acc'].append(acc_tc)
            results[eps]['tcdp']['mse'].append(mse_tc)
            results[eps]['tcdp']['consistency'].append(consistency_loss.item())
            
            print(f"  Batch {batch_idx}: Purify++ Acc: {acc_pp:.3f}, TCDP Acc: {acc_tc:.3f}")
            
            batch_count += 1
    
    epsilons = []
    acc_pp_means = []
    acc_tc_means = []
    mse_pp_means = []
    mse_tc_means = []
    
    for eps in eps_list:
        epsilons.append(eps)
        acc_pp_mean = np.mean(results[eps]['purify_pp']['acc'])
        acc_tc_mean = np.mean(results[eps]['tcdp']['acc'])
        mse_pp_mean = np.mean(results[eps]['purify_pp']['mse'])
        mse_tc_mean = np.mean(results[eps]['tcdp']['mse'])
        acc_pp_means.append(acc_pp_mean)
        acc_tc_means.append(acc_tc_mean)
        mse_pp_means.append(mse_pp_mean)
        mse_tc_means.append(mse_tc_mean)
        print(f"Epsilon {eps:.4f} -> Purify++: Acc={acc_pp_mean:.3f}, MSE={mse_pp_mean:.4f} | TCDP: Acc={acc_tc_mean:.3f}, MSE={mse_tc_mean:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, acc_pp_means, marker='o', label='Purify++')
    plt.plot(epsilons, acc_tc_means, marker='s', label='TCDP')
    plt.xlabel("PGD Epsilon")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy vs. PGD Epsilon (Exp 1)")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/accuracy_adaptiverobust_pair1.pdf", bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, mse_pp_means, marker='o', label='Purify++ MSE')
    plt.plot(epsilons, mse_tc_means, marker='s', label='TCDP MSE')
    plt.xlabel("PGD Epsilon")
    plt.ylabel("MSE (Reconstruction Error)")
    plt.title("Reconstruction Error vs. PGD Epsilon (Exp 1)")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/reconstruction_error_adaptiverobust_pair1.pdf", bbox_inches='tight')
    plt.close()
    
    return results

def experiment2(num_batches=2):
    """
    Experiment 2: Ablation Study – Evaluating Component Impact
    
    Tests the performance of different TCDP variants against synthetic noise.
    """
    print("\nRunning Experiment 2: Ablation Study – Evaluating Component Impact")
    
    os.makedirs('./logs', exist_ok=True)
    
    testloader = get_testloader(batch_size=EXPERIMENT_CONFIG['test_batch_size'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tcdp_full = TCDP().to(device)
    tcdp_no_cons = TCDP_NoConsistency().to(device)
    tcdp_fixed = TCDP_FixedNoise(fixed_noise_level=0.1).to(device)
    
    try:
        tcdp_full.load_state_dict(torch.load('./models/tcdp.pth', map_location=device))
        tcdp_no_cons.load_state_dict(torch.load('./models/tcdp_no_consistency.pth', map_location=device))
        tcdp_fixed.load_state_dict(torch.load('./models/tcdp_fixed.pth', map_location=device))
        print("Loaded pretrained models.")
    except Exception as e:
        print(f"Warning: Pretrained models not found ({e}). Using randomly initialized models for demonstration.")
    
    criterion = nn.MSELoss()
    
    noise_levels = EXPERIMENT_CONFIG['noise_levels']
    
    results = {}
    for noise in noise_levels:
        results[noise] = {'TCDP_full': [], 'TCDP_no_cons': [], 'TCDP_fixed': []}
    
    for noise in noise_levels:
        print(f"\nProcessing noise level: {noise}")
        batch_count = 0
        for batch_idx, (images, _) in enumerate(testloader):
            if batch_count >= num_batches:
                break
            images = images.to(device)
            noisy_images = images + noise * torch.randn_like(images)
            
            purified_full, _ = tcdp_full(noisy_images)
            purified_no_cons = tcdp_no_cons(noisy_images)
            purified_fixed = tcdp_fixed(noisy_images)
            
            mse_full = criterion(purified_full, images).item()
            mse_no_cons = criterion(purified_no_cons, images).item()
            mse_fixed = criterion(purified_fixed, images).item()
            
            results[noise]['TCDP_full'].append(mse_full)
            results[noise]['TCDP_no_cons'].append(mse_no_cons)
            results[noise]['TCDP_fixed'].append(mse_fixed)
            
            print(f"  Batch {batch_idx}: MSE Full: {mse_full:.4f}, NoCons: {mse_no_cons:.4f}, Fixed: {mse_fixed:.4f}")
            
            batch_count += 1
    
    mse_full_means = []
    mse_no_cons_means = []
    mse_fixed_means = []
    noise_list = []
    for noise in noise_levels:
        noise_list.append(noise)
        mse_full_means.append(np.mean(results[noise]['TCDP_full']))
        mse_no_cons_means.append(np.mean(results[noise]['TCDP_no_cons']))
        mse_fixed_means.append(np.mean(results[noise]['TCDP_fixed']))
        print(f"Noise {noise:.2f} -> Full: {np.mean(results[noise]['TCDP_full']):.4f}, NoCons: {np.mean(results[noise]['TCDP_no_cons']):.4f}, Fixed: {np.mean(results[noise]['TCDP_fixed']):.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_list, mse_full_means, marker='o', label='TCDP Full')
    plt.plot(noise_list, mse_no_cons_means, marker='s', label='TCDP NoConsistency')
    plt.plot(noise_list, mse_fixed_means, marker='^', label='TCDP Fixed Noise')
    plt.xlabel("Synthetic Noise Level")
    plt.ylabel("Mean Squared Error")
    plt.title("Ablation Study: MSE vs. Noise Level (Exp 2)")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/ablation_mse_pair1.pdf", bbox_inches='tight')
    plt.close()
    
    return results

def experiment3(num_batches=2):
    """
    Experiment 3: Adaptive Noise Control Efficiency
    
    Tests the efficiency of adaptive noise control compared to fixed noise schedule.
    """
    print("\nRunning Experiment 3: Adaptive Noise Control Efficiency")
    
    os.makedirs('./logs', exist_ok=True)
    
    testloader = get_testloader(batch_size=EXPERIMENT_CONFIG['test_batch_size'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tcdp_adaptive = TCDP_Adaptive(max_steps=10, early_stop_thresh=0.01).to(device)
    tcdp_fixed = TCDP_FixedNoise(fixed_noise_level=0.1).to(device)
    
    try:
        tcdp_adaptive.load_state_dict(torch.load('./models/tcdp_adaptive.pth', map_location=device))
        tcdp_fixed.load_state_dict(torch.load('./models/tcdp_fixed.pth', map_location=device))
        print("Loaded pretrained models.")
    except Exception as e:
        print(f"Warning: Pretrained models not found ({e}). Using randomly initialized models for demonstration.")
    
    adaptive_timings = []
    adaptive_steps = []
    fixed_timings = []
    
    for batch_idx, (images, _) in enumerate(testloader):
        if batch_idx >= num_batches:
            break
        images = images.to(device)
        noisy_images = images + 0.1 * torch.randn_like(images)
        
        start_time = time.time()
        purified_adaptive, steps, cons_history = tcdp_adaptive(noisy_images)
        adaptive_time = time.time() - start_time
        
        start_time = time.time()
        purified_fixed = tcdp_fixed(noisy_images)
        fixed_time = time.time() - start_time
        
        adaptive_timings.append(adaptive_time)
        adaptive_steps.append(steps)
        fixed_timings.append(fixed_time)
        
        print(f"Batch {batch_idx}: Adaptive -> Steps: {steps}, Time: {adaptive_time:.4f}s; Fixed -> Time: {fixed_time:.4f}s; Consistency history: {cons_history}")
    
    avg_adaptive_time = np.mean(adaptive_timings)
    avg_fixed_time = np.mean(fixed_timings)
    avg_steps = np.mean(adaptive_steps)
    
    print(f"\nAverage Adaptive Time per Batch: {avg_adaptive_time:.4f}s, Average Fixed Time per Batch: {avg_fixed_time:.4f}s, Average Adaptive Steps: {avg_steps:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(adaptive_steps)), adaptive_steps)
    plt.xlabel("Batch Index")
    plt.ylabel("Number of Iterations")
    plt.title("Adaptive Method Step Count Per Batch (Exp 3)")
    plt.grid(True)
    plt.savefig("logs/inference_iteration_adaptive_pair1.pdf", bbox_inches='tight')
    plt.close()
    
    indices = np.arange(len(adaptive_timings))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(indices, adaptive_timings, width, label='Adaptive')
    plt.bar(indices + width, fixed_timings, width, label='Fixed')
    plt.xlabel("Batch Index")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime per Batch: Adaptive vs. Fixed (Exp 3)")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/inference_latency_comparison_pair1.pdf", bbox_inches='tight')
    plt.close()
    
    efficiency_results = {'adaptive_time': adaptive_timings,
                          'adaptive_steps': adaptive_steps,
                          'fixed_time': fixed_timings}
    return efficiency_results
