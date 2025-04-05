"""
Script for evaluating models.

This module implements the evaluation procedures for the ASID-M experiments.
"""

import time
import torch
import torch.nn as nn
import timeit
import numpy as np
from train import TeacherModel, OneStepStudent, SimpleOneStepStudent, set_seed

def measure_inference_time(model, sample_batch, device, num_runs=10):
    """
    Measures the inference time of a model.
    
    Args:
        model (nn.Module): Model to measure.
        sample_batch (torch.Tensor): Sample batch to use for measurement.
        device (torch.device): Device to run the model on.
        num_runs (int): Number of runs to average over.
        
    Returns:
        float: Average inference time in seconds.
    """
    model.eval()
    sample_batch = sample_batch.to(device)
    
    with torch.no_grad():
        _ = model(sample_batch)
    
    def run_inference():
        with torch.no_grad():
            _ = model(sample_batch)
    
    time_taken = timeit.timeit(run_inference, number=num_runs)
    avg_time = time_taken / num_runs
    
    return avg_time

def evaluate_model(model, dataloader, teacher_model, criterion, device, 
                  noise_std=0.1, max_batches=None):
    """
    Evaluates a model on a dataset.
    
    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader containing evaluation data.
        teacher_model (nn.Module): Teacher model for comparison.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.
        noise_std (float): Standard deviation of noise to add to images.
        max_batches (int): Maximum number of batches to use.
        
    Returns:
        float: Average loss.
    """
    model.eval()
    running_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            images = images.to(device)
            noisy_images = images + torch.randn_like(images) * noise_std
            
            teacher_target = teacher_model.generate_trajectory(images, num_steps=20)
            
            student_output = model(noisy_images)
            loss = criterion(student_output, teacher_target)
            
            running_loss += loss.item()
            batch_count += 1
    
    avg_loss = running_loss / batch_count if batch_count > 0 else 0
    return avg_loss

def evaluate_noise_sensitivity(model, dataloader, teacher_model, criterion, device, 
                              noise_levels, max_batches=5):
    """
    Evaluates a model's sensitivity to different noise levels.
    
    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader containing evaluation data.
        teacher_model (nn.Module): Teacher model for comparison.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to evaluate on.
        noise_levels (list): List of noise standard deviations to test.
        max_batches (int): Maximum number of batches to use.
        
    Returns:
        tuple: Dictionaries mapping noise levels to losses and inference times.
    """
    loss_vs_noise = {}
    inference_time_vs_noise = {}
    
    for noise_std in noise_levels:
        print(f"Testing with noise_std = {noise_std}")
        
        avg_loss = evaluate_model(
            model, dataloader, teacher_model, criterion, device, 
            noise_std=noise_std, max_batches=max_batches
        )
        loss_vs_noise[noise_std] = avg_loss
        
        sample_batch = torch.randn(64, 3, 32, 32).to(device)
        avg_time = measure_inference_time(model, sample_batch, device)
        inference_time_vs_noise[noise_std] = avg_time
        
        print(f"Average loss at noise level {noise_std:.2f}: {avg_loss:.4f}")
        print(f"Inference time per batch at noise level {noise_std:.2f}: {avg_time:.6f} sec")
    
    return loss_vs_noise, inference_time_vs_noise

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_data = torch.randn(16, 3, 32, 32).to(device)
    dummy_dataloader = [(input_data, torch.zeros(16)) for _ in range(2)]
    
    teacher_model = TeacherModel().to(device)
    student_model = OneStepStudent().to(device)
    
    criterion = nn.MSELoss()
    
    avg_loss = evaluate_model(
        student_model, dummy_dataloader, teacher_model, criterion, device,
        noise_std=0.1, max_batches=2
    )
    
    print(f"Test evaluation completed with average loss: {avg_loss:.4f}")
    
    sample_batch = torch.randn(64, 3, 32, 32).to(device)
    avg_time = measure_inference_time(student_model, sample_batch, device, num_runs=5)
    print(f"Average inference time: {avg_time:.6f} seconds")
