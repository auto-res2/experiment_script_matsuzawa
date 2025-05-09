"""
G‑DS3 Transformer Experiments:
  1. Comparative Performance on Long‐Sequence Tasks
  2. Efficiency and Hardware Scalability Benchmark
  3. Ablation Study: Impact of the Gating Mechanism
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from preprocess import CopyMemoryDataset
from train import (TransformerModel, AblationTransformerModel, train_model, 
                  GatingModule, LightweightGatingModule)
from evaluate import benchmark_model, run_profiler, calculate_flops

os.makedirs('logs', exist_ok=True)

def experiment_comparative_performance():
    print("\nRunning Experiment 1: Comparative Performance on Long‐Sequence Tasks")
    dataset = CopyMemoryDataset(seq_len=50, num_samples=400)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model_gds3 = TransformerModel(d_model=10, d_state=16, d_conv=8, num_layers=2, use_gate=True)
    model_baseline = TransformerModel(d_model=10, d_state=16, d_conv=8, num_layers=2, use_gate=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses_gds3 = train_model(model_gds3, dataloader, num_epochs=3, device=device)
    losses_baseline = train_model(model_baseline, dataloader, num_epochs=3, device=device)
    
    plt.figure(figsize=(10, 6), dpi=300)  # Higher resolution for academic papers
    plt.plot(losses_gds3, marker='o', label="G‑DS3")
    plt.plot(losses_baseline, marker='s', label="Baseline SSD")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss: Comparative Performance")
    plt.legend()
    filename = "logs/training_loss_comparative_pair1.pdf"
    plt.savefig(filename, format='pdf')
    print("Saved training loss plot as", filename)
    plt.close()

def experiment_efficiency_scalability():
    print("\nRunning Experiment 2: Efficiency and Hardware Scalability Benchmark")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_gds3 = TransformerModel(d_model=10, d_state=16, d_conv=8, num_layers=2, use_gate=True)
    model_gds3.to(device)
    seq_lengths = [50, 100, 200]
    times = []
    
    for seq_len in seq_lengths:
        dataset = CopyMemoryDataset(seq_len=seq_len, num_samples=100)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        t = benchmark_model(model_gds3, dataloader, device=device)
        print("Sequence length:", seq_len, "Time taken:", t)
        times.append(t)
        run_profiler(model_gds3, dataloader, device=device)
    
    plt.figure(figsize=(10, 6), dpi=300)  # Higher resolution for academic papers
    plt.plot(seq_lengths, times, marker='x')
    plt.xlabel("Sequence Length")
    plt.ylabel("Inference Time (s)")
    plt.title("Inference Latency vs. Sequence Length")
    filename = "logs/inference_latency_efficiency_pair1.pdf"
    plt.savefig(filename, format='pdf')
    print("Saved efficiency benchmark plot as", filename)
    plt.close()
    
    flops, params = calculate_flops(model_gds3, (50, 10))
    print("G‑DS3 FLOPs (dummy input):", flops, "Params:", params)

def experiment_ablation_study():
    print("\nRunning Experiment 3: Ablation Study on Gating Mechanism")
    dataset = CopyMemoryDataset(seq_len=50, num_samples=400)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    full_gate_factory = lambda d_state: GatingModule(d_state)
    no_gate_factory = lambda d_state: None
    light_gate_factory = lambda d_state: LightweightGatingModule(d_state)
    
    model_full = AblationTransformerModel(d_model=10, d_state=16, d_conv=8, num_layers=2, gating_module_factory=full_gate_factory)
    model_no = AblationTransformerModel(d_model=10, d_state=16, d_conv=8, num_layers=2, gating_module_factory=no_gate_factory)
    model_light = AblationTransformerModel(d_model=10, d_state=16, d_conv=8, num_layers=2, gating_module_factory=light_gate_factory)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    losses_full = train_model(model_full, dataloader, num_epochs=3, device=device)
    losses_none = train_model(model_no, dataloader, num_epochs=3, device=device)
    losses_light = train_model(model_light, dataloader, num_epochs=3, device=device)
    
    plt.figure(figsize=(10, 6), dpi=300)  # Higher resolution for academic papers
    plt.plot(losses_full, marker='o', label="Full Gate")
    plt.plot(losses_none, marker='s', label="No Gate")
    plt.plot(losses_light, marker='^', label="Lightweight Gate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Ablation Study: Impact of Gate Mechanism")
    plt.legend()
    filename = "logs/training_loss_ablation_pair1.pdf"
    plt.savefig(filename, format='pdf')
    print("Saved ablation study loss plot as", filename)
    plt.close()
    
    def hook_fn(module, input, output):
        print("Gate mean activation:", output.mean().item())
    for name, module in model_full.named_modules():
        if isinstance(module, GatingModule):
            module.register_forward_hook(hook_fn)
    
    model_full.eval()
    with torch.no_grad():
        sample = F.one_hot(torch.randint(0, 10, (1, 50)), num_classes=10).float().to(device)
        _ = model_full(sample)

def quick_test():
    print("\nRunning quick test to verify code execution.")
    test_dataset = CopyMemoryDataset(seq_len=20, num_samples=50)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model = TransformerModel(d_model=10, d_state=8, d_conv=4, num_layers=1, use_gate=True)
    _ = train_model(test_model, test_dataloader, num_epochs=1, device=device)
    
    t = benchmark_model(test_model, test_dataloader, device=device)
    print("Quick benchmark time:", t)
    
    test_model_ablation = AblationTransformerModel(d_model=10, d_state=8, d_conv=4, num_layers=1, gating_module_factory=lambda d_state: LightweightGatingModule(d_state))
    _ = train_model(test_model_ablation, test_dataloader, num_epochs=1, device=device)
    print("Quick test finished.")

if __name__ == "__main__":
    status_enum = "running"
    print("Status: ", status_enum)
    
    quick_test()
    
    experiment_comparative_performance()
    experiment_efficiency_scalability()
    experiment_ablation_study()
    
    status_enum = "stopped"
    print("Status: ", status_enum)
    print("\nAll experiments completed successfully.")
