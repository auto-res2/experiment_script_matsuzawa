"""
Evaluation functions for G-DS3 Transformer experiments.
"""

import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

def benchmark_model(model, dataloader, device="cpu"):
    """
    Benchmark the inference time of a model on the given dataloader.
    
    Args:
        model: The model to benchmark
        dataloader: The dataloader containing the test data
        device: Device to run inference on (cpu or cuda)
        
    Returns:
        Total inference time in seconds
    """
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for seq, _ in dataloader:
            embed = F.one_hot(seq, num_classes=10).float().to(device)
            model(embed)
    total_time = time.time() - start_time
    return total_time

def run_profiler(model, dataloader, device="cpu"):
    """
    Run PyTorch profiler on the model with the given dataloader.
    
    Args:
        model: The model to profile
        dataloader: The dataloader containing the test data
        device: Device to run inference on (cpu or cuda)
    """
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True
         ) as prof:
        with torch.no_grad():
            for seq, _ in dataloader:
                embed = F.one_hot(seq, num_classes=10).float().to(device)
                model(embed)
    print("Profiler results:\n", prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))

def calculate_flops(model, input_shape=(50, 10)):
    """
    Calculate FLOPs for the given model.
    
    Args:
        model: The model to calculate FLOPs for
        input_shape: Input shape (seq_len, d_model)
        
    Returns:
        FLOPs and number of parameters as strings
    """
    flops, params = get_model_complexity_info(model, input_shape, as_strings=True, 
                                              print_per_layer_stat=False)
    return flops, params
