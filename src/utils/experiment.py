import time
import os
import torch
import torch.quantization as quant
import matplotlib.pyplot as plt
import numpy as np

def benchmark_inference(model, input_ids, num_runs=20):
    """
    Benchmark inference latency for a given model.
    
    Args:
        model: PyTorch model to benchmark
        input_ids: Input tensor for the model
        num_runs: Number of runs to average over
        
    Returns:
        Average inference time in milliseconds
    """
    timings = []
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(input_ids)
            end_time = time.time()
            timings.append((end_time - start_time) * 1000)  # in milliseconds
    avg_time = sum(timings) / len(timings)
    return avg_time

def quantize_model(model):
    """
    Apply dynamic quantization to a PyTorch model.
    
    Args:
        model: PyTorch model to quantize
        
    Returns:
        Quantized model
    """
    quantized_model = quant.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model

def save_barplot(x_labels, y_values, title, xlabel, ylabel, filename):
    """
    Create and save a bar plot as a PDF file.
    
    Args:
        x_labels: Labels for x-axis
        y_values: Values for y-axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.bar(x_labels, y_values, color=['blue', 'green'])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close()

def save_lineplot(x_values, y_series, labels, title, xlabel, ylabel, filename):
    """
    Create and save a line plot with multiple series as a PDF file.
    
    Args:
        x_values: Values for x-axis
        y_series: List of y-value series
        labels: Labels for each series
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Output filename
    """
    plt.figure(figsize=(10, 6), dpi=300)
    for i, y_values in enumerate(y_series):
        plt.plot(x_values, y_values, label=labels[i], marker='o' if i == 0 else 'x',
                linestyle='-' if i == 0 else '--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close()

def setup_device():
    """
    Set up the device for training/inference.
    
    Returns:
        torch.device: Device to use
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available. Using CPU.")
    return device
