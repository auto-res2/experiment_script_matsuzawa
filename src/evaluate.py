import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from src.utils.diffusion import diffusion_step, double_tweedie_transform, calculate_adaptive_lambda, check_consistency

def cap_defense(inputs, score_model, max_steps=10, adaptive_consistency=True, 
               use_double_tweedie=True, adaptive_steps=True):
    """Apply Consistent Adaptive Purification (CAP) defense.
    
    Args:
        inputs: Input tensor (adversarial examples)
        score_model: Model for score estimation
        max_steps: Maximum number of diffusion steps
        adaptive_consistency: Whether to use adaptive consistency loss
        use_double_tweedie: Whether to use double-Tweedie transformation
        adaptive_steps: Whether to use adaptive step scheduling
        
    Returns:
        purified: Purified tensor
        steps_executed: Number of steps executed
        duration: Time taken for purification
    """
    start_time = time.time()
    device = inputs.device
    purified = inputs.clone()
    previous = None
    steps_executed = 0
    
    noise_level = 0.1
    
    if adaptive_consistency:
        lambda_val = calculate_adaptive_lambda(inputs)
    else:
        lambda_val = torch.ones_like(inputs[:, 0:1, 0:1, 0:1])
    
    for step in range(max_steps):
        if use_double_tweedie and step > 0:
            purified = double_tweedie_transform(purified, score_model, noise_level)
        else:
            purified = diffusion_step(purified, noise_level)
        
        noise_level = noise_level * 0.9
        
        if previous is None:
            previous = purified.clone()
            
        steps_executed += 1
        
        if adaptive_steps and step > 0 and check_consistency(purified, previous):
            break
            
        previous = purified.clone()
    
    duration = time.time() - start_time
    return purified, steps_executed, duration

def purifypp_defense(inputs, steps=10):
    """Apply Purify++ defense (simplified).
    
    Args:
        inputs: Input tensor (adversarial examples)
        steps: Number of diffusion steps
        
    Returns:
        purified: Purified tensor
    """
    purified = inputs.clone()
    noise_level = 0.1
    
    for _ in range(steps):
        purified = diffusion_step(purified, noise_level)
        noise_level = noise_level * 0.9
        
    return purified

def baseline_defense(inputs):
    """Apply baseline defense (simplified).
    
    Args:
        inputs: Input tensor (adversarial examples)
        
    Returns:
        purified: Purified tensor
    """
    return inputs + 0.001 * torch.randn_like(inputs)

def evaluate_model(model, test_loader, defense_fn=None, device='cuda:0'):
    """Evaluate a model's accuracy.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        defense_fn: Defense function to apply (None for clean evaluation)
        device: Device to evaluate on
        
    Returns:
        accuracy: Classification accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if defense_fn is not None:
                inputs = defense_fn(inputs)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def save_figure(fig, filename, dpi=300):
    """Save a figure in high-quality PDF format.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution (dots per inch)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, format='pdf', dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {filename}")

def experiment1(model, test_loader, device='cuda:0', score_model=None):
    """Run Experiment 1: Comparison of Adversarial Robustness Performance.
    
    Args:
        model: Classifier model
        test_loader: DataLoader with adversarial examples
        device: Device to run on
        score_model: Score model for CAP defense
        
    Returns:
        results: Dictionary with accuracy results
    """
    print("Running Experiment 1: Comparison of Adversarial Robustness Performance")
    
    def cap_defense_fn(inputs):
        purified, _, _ = cap_defense(inputs, score_model)
        return purified
        
    cap_accuracy = evaluate_model(model, test_loader, cap_defense_fn, device)
    purifypp_accuracy = evaluate_model(model, test_loader, purifypp_defense, device)
    baseline_accuracy = evaluate_model(model, test_loader, baseline_defense, device)
    
    print(f"CAP Accuracy: {cap_accuracy:.4f}")
    print(f"Purify++ Accuracy: {purifypp_accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    
    labels = ['CAP', 'Purify++', 'Baseline']
    accuracies = [cap_accuracy, purifypp_accuracy, baseline_accuracy]
    
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=labels, y=accuracies, palette="viridis")
    plt.ylabel("Classification Accuracy")
    plt.title("Adversarial Robustness Performance")
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
    plt.tight_layout()
    save_figure(plt.gcf(), "logs/adversarial_robustness_performance.pdf")
    plt.close()
    
    return {'CAP': cap_accuracy, 'Purify++': purifypp_accuracy, 'Baseline': baseline_accuracy}

def experiment2(model, test_loader, device='cuda:0', score_model=None):
    """Run Experiment 2: Ablation Study of CAP Components.
    
    Args:
        model: Classifier model
        test_loader: DataLoader with adversarial examples
        device: Device to run on
        score_model: Score model for CAP defense
        
    Returns:
        results: Dictionary with ablation study results
    """
    print("Running Experiment 2: Ablation Study of CAP Components")
    
    variants = {
        "Full_CAP": {"adaptive_consistency": True, "use_double_tweedie": True, "adaptive_steps": True},
        "No_Adaptive_Consistency": {"adaptive_consistency": False, "use_double_tweedie": True, "adaptive_steps": True},
        "No_Double_Tweedie": {"adaptive_consistency": True, "use_double_tweedie": False, "adaptive_steps": True},
        "No_Adaptive_Steps": {"adaptive_consistency": True, "use_double_tweedie": True, "adaptive_steps": False}
    }
    
    results = {}
    
    for name, options in variants.items():
        accuracies = []
        steps_list = []
        durations = []
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            purified, steps, duration = cap_defense(inputs, score_model, **options)
            
            model.eval()
            with torch.no_grad():
                outputs = model(purified)
                _, predicted = outputs.max(1)
                accuracy = predicted.eq(labels).float().mean().item()
                
            accuracies.append(accuracy)
            steps_list.append(steps)
            durations.append(duration)
            
        avg_accuracy = np.mean(accuracies)
        avg_steps = np.mean(steps_list)
        avg_duration = np.mean(durations)
        
        results[name] = {
            "Accuracy": avg_accuracy,
            "Avg_Steps": avg_steps,
            "Avg_Duration": avg_duration
        }
        
        print(f"{name}: Accuracy={avg_accuracy:.4f}, Avg_Steps={avg_steps:.2f}, Avg_Duration={avg_duration:.4f}s")
    
    variant_names = list(results.keys())
    accuracies = [results[name]["Accuracy"] for name in variant_names]
    avg_steps = [results[name]["Avg_Steps"] for name in variant_names]
    avg_durations = [results[name]["Avg_Duration"] for name in variant_names]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    ax1 = sns.barplot(x=variant_names, y=accuracies, palette="magma")
    plt.ylabel("Classification Accuracy")
    plt.title("Ablation Study: Accuracy")
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.subplot(1, 2, 2)
    bar_width = 0.35
    indices = np.arange(len(variant_names))
    
    ax2 = plt.gca()
    bar1 = ax2.bar(indices, avg_steps, bar_width, label="Avg Steps")
    bar2 = ax2.bar(indices + bar_width, avg_durations, bar_width, label="Avg Duration (s)")
    
    plt.xticks(indices + bar_width/2, variant_names)
    plt.ylabel("Resource Usage")
    plt.title("Ablation Study: Resource Usage")
    plt.legend()
    
    plt.tight_layout()
    save_figure(plt.gcf(), "logs/ablation_study.pdf")
    plt.close()
    
    return results

def experiment3(model, test_loader, device='cuda:0', score_model=None):
    """Run Experiment 3: Efficiency and Resource Usage Analysis.
    
    Args:
        model: Classifier model
        test_loader: DataLoader for test data
        device: Device to run on
        score_model: Score model for CAP defense
        
    Returns:
        data: Dictionary with efficiency analysis results
    """
    print("Running Experiment 3: Efficiency and Resource Usage Analysis")
    
    perturbation_levels = [2/255, 4/255, 6/255, 8/255]
    
    efficiency_data = {
        "Perturbation": [],
        "Avg_Steps": [],
        "Avg_Duration": [],
        "Accuracy": []
    }
    
    from advertorch.attacks import PGDAttack
    
    criterion = nn.CrossEntropyLoss()
    
    for eps in perturbation_levels:
        adversary = PGDAttack(
            model, loss_fn=criterion, eps=eps,
            nb_iter=10, eps_iter=eps/4,
            rand_init=True, clip_min=0.0, clip_max=1.0
        )
        
        steps_list = []
        durations = []
        accuracies = []
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            adv_inputs = adversary.perturb(inputs, labels)
            
            start_time = time.time()
            purified, steps, _ = cap_defense(adv_inputs, score_model)
            duration = time.time() - start_time
            
            model.eval()
            with torch.no_grad():
                outputs = model(purified)
                _, predicted = outputs.max(1)
                accuracy = predicted.eq(labels).float().mean().item()
            
            steps_list.append(steps)
            durations.append(duration)
            accuracies.append(accuracy)
        
        avg_steps = np.mean(steps_list)
        avg_duration = np.mean(durations)
        avg_accuracy = np.mean(accuracies)
        
        efficiency_data["Perturbation"].append(eps)
        efficiency_data["Avg_Steps"].append(avg_steps)
        efficiency_data["Avg_Duration"].append(avg_duration)
        efficiency_data["Accuracy"].append(avg_accuracy)
        
        print(f"Epsilon={eps:.6f}: Accuracy={avg_accuracy:.4f}, Avg_Steps={avg_steps:.2f}, Avg_Duration={avg_duration:.4f}s")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(efficiency_data["Perturbation"], efficiency_data["Avg_Steps"], marker='o', linestyle='-')
    plt.title("Avg Diffusion Steps vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Avg Diffusion Steps")
    
    plt.subplot(1, 3, 2)
    plt.plot(efficiency_data["Perturbation"], efficiency_data["Avg_Duration"], marker='o', linestyle='-')
    plt.title("Avg Runtime vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Avg Duration (s)")
    
    plt.subplot(1, 3, 3)
    plt.plot(efficiency_data["Perturbation"], efficiency_data["Accuracy"], marker='o', linestyle='-')
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    save_figure(plt.gcf(), "logs/inference_efficiency.pdf")
    plt.close()
    
    return efficiency_data
