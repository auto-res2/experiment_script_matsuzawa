import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import memory_usage

def experiment_multimodal_instruction(tokenizer, sample_texts, sample_image_paths, load_image_func):
    """
    Uses a small benchmark of image-text pairs to compare AMICT vs Base Method.
    Demonstrates the effect of added visual cues.
    """
    print("\n[Experiment 1] Starting Multimodal Instruction-Following Evaluation...")

    results_amict = []
    results_base = []

    for i, (text, path) in enumerate(zip(sample_texts, sample_image_paths)):
        sample_image = load_image_func(path)
        encoded_text = tokenizer(text, return_tensors="pt") if hasattr(tokenizer, '__call__') else text
        
        from train import run_amict, run_base_method
        
        response_amict = run_amict(sample_image, encoded_text)
        response_base = run_base_method(encoded_text)
        
        print(f"[Experiment 1] Text: '{text}'")
        print(f"[Experiment 1] AMICT response: {response_amict}")
        print(f"[Experiment 1] Base Method response: {response_base}")
        results_amict.append(len(response_amict))
        results_base.append(len(response_base))
    
    plt.figure(figsize=(8, 6))
    indices = np.arange(len(results_amict))
    width = 0.35
    plt.bar(indices - width/2, results_amict, width=width, label="AMICT", color='blue', alpha=0.7)
    plt.bar(indices + width/2, results_base, width=width, label="Base Method", color='green', alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("Response String Length")
    plt.title("Response Length Comparison (Multimodal Evaluation)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("logs/experiment1_multimodal_comparison.pdf", dpi=300, bbox_inches='tight')
    print("[Experiment 1] Plot saved as 'logs/experiment1_multimodal_comparison.pdf'.")

def evaluate_model_on_long_text(model_func, token_length, tokenizer=None, text_generator=None):
    """
    Runs a long text through the given model function and measures latency.
    """
    text_input = text_generator(token_length) if text_generator else f"Dummy text with {token_length} tokens."
    encoded = tokenizer(text_input, return_tensors="pt") if tokenizer else text_input
    
    start_time = time.time()
    response = model_func(encoded)
    latency = time.time() - start_time
    
    return response, latency

def experiment_long_context(tokenizer=None, text_generator=None):
    """
    Evaluates performance on inputs of varied token lengths.
    Records generation latencies and prints response snippets.
    Also plots latency vs. token length.
    """
    print("\n[Experiment 2] Starting Long-Context Handling Evaluation...")
    token_lengths = [2000, 8000, 10000]
    latencies_amict = []
    latencies_base = []
    
    from train import run_amict_text, run_base_text

    for length in token_lengths:
        response_amict, latency_amict = evaluate_model_on_long_text(
            run_amict_text, length, tokenizer, text_generator
        )
        response_base, latency_base = evaluate_model_on_long_text(
            run_base_text, length, tokenizer, text_generator
        )
        
        print(f"[Experiment 2] Token length: {length}")
        print(f"[Experiment 2] AMICT latency: {latency_amict:.4f} sec; Response snippet: {response_amict[:80]}")
        print(f"[Experiment 2] Base Method latency: {latency_base:.4f} sec; Response snippet: {response_base[:80]}")
        latencies_amict.append(latency_amict)
        latencies_base.append(latency_base)
    
    plt.figure(figsize=(10, 6))
    plt.plot(token_lengths, latencies_amict, marker='o', linestyle='-', label="AMICT", color='blue')
    plt.plot(token_lengths, latencies_base, marker='s', linestyle='--', label="Base Method", color='green')
    plt.xlabel("Input Token Length")
    plt.ylabel("Latency (seconds)")
    plt.title("Long-Context Inference Latency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("logs/experiment2_long_context_latency.pdf", dpi=300, bbox_inches='tight')
    print("[Experiment 2] Plot saved as 'logs/experiment2_long_context_latency.pdf'.")

def benchmark_inference(model, input_tensor, iterations=20):
    """
    Benchmarks a model by measuring average inference latency and memory usage change.
    Uses a few warm-up iterations first.
    """
    for _ in range(3):
        _ = model(input_tensor)
        
    mem_before = memory_usage()[0]
    
    start = time.time()
    for _ in range(iterations):
        _ = model(input_tensor)
    elapsed_time = time.time() - start
    
    mem_after = memory_usage()[0]
    
    avg_latency = elapsed_time / iterations
    memory_used = mem_after - mem_before
    return avg_latency, memory_used

def experiment_on_device_inference(device):
    """
    Benchmarks dummy AMICT and Base Method models for inference latency and memory efficiency.
    Saves a plot comparing both metrics.
    """
    print(f"\n[Experiment 3] Starting On-Device Inference Resource Efficiency Benchmark on {device}...")
    
    from train import DummyAMICTModel, DummyBaseModel
    
    amict_model = DummyAMICTModel().to(device).eval()
    base_model = DummyBaseModel().to(device).eval()
    
    input_tensor = torch.rand(1, 768, device=device)
    
    latency_amict, mem_amict = benchmark_inference(amict_model, input_tensor)
    latency_base, mem_base = benchmark_inference(base_model, input_tensor)
    
    print("[Experiment 3] AMICT Inference Latency (avg): {:.4f} sec; Memory Usage Change: {:.4f} MB"
          .format(latency_amict, mem_amict))
    print("[Experiment 3] Base Method Inference Latency (avg): {:.4f} sec; Memory Usage Change: {:.4f} MB"
          .format(latency_base, mem_base))
    
    labels = ["AMICT", "Base Method"]
    latencies = [latency_amict, latency_base]
    mem_changes = [mem_amict, mem_base]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.bar(labels, latencies, color=['blue', 'green'], alpha=0.7)
    ax1.set_ylabel("Latency (seconds)")
    ax1.set_title("Inference Latency")
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(labels, mem_changes, color=['blue', 'green'], alpha=0.7)
    ax2.set_ylabel("Memory Change (MB)")
    ax2.set_title("Memory Usage")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("logs/experiment3_inference_benchmark.pdf", dpi=300, bbox_inches='tight')
    print("[Experiment 3] Plot saved as 'logs/experiment3_inference_benchmark.pdf'.")
