"""
Main script for running ASID-M experiments.

This module coordinates the entire experiment process, from data preprocessing
to model training and evaluation.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from preprocess import get_dataloader
from train import TeacherModel, OneStepStudent, SimpleOneStepStudent, train_model, set_seed
from evaluate import evaluate_model, measure_inference_time, evaluate_noise_sensitivity
from utils.plotting import save_loss_curve, save_image_grid, save_scatter_plot

def ensure_dirs_exist():
    """Ensures that necessary directories exist."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("config", exist_ok=True)

def experiment_controlled_image_generation(epochs=2, noise_std=0.1, batch_size=128, max_batches=5):
    """
    Experiment 1: Controlled Image Generation Comparison.
    
    Args:
        epochs (int): Number of epochs to train for.
        noise_std (float): Standard deviation of noise to add to images.
        batch_size (int): Batch size for training.
        max_batches (int): Maximum number of batches to use per epoch.
    """
    print("\nStarting Experiment 1: Controlled Image Generation Comparison")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader = get_dataloader(batch_size=batch_size, train=True)
    
    teacher_model = TeacherModel().to(device)
    teacher_model.eval()
    teacher_trajectories = []
    
    print("Precomputing teacher trajectories...")
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            traj = teacher_model.generate_trajectory(images, num_steps=20)
            teacher_trajectories.append(traj)
    print(f"Precomputed trajectories for {len(teacher_trajectories)} batches")
    
    student_model = OneStepStudent().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    
    loss_history = []
    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= len(teacher_trajectories):
                break
            
            images = images.to(device)
            noisy_images = images + torch.randn_like(images) * noise_std
            teacher_target = teacher_trajectories[batch_idx]
            
            student_output = student_model(noisy_images)
            loss = criterion(student_output, teacher_target)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            running_loss += loss.item()
            if (batch_idx+1) % 2 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(teacher_trajectories)}], Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / len(teacher_trajectories)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} finished with average loss {avg_loss:.4f}")
    
    torch.save(student_model.state_dict(), "models/asidm_student.pth")
    
    save_loss_curve(
        list(range(1, epochs+1)), loss_history, 
        "training_loss", "Training Loss Curve for ASID-M (Exp. 1)"
    )
    print("Saved training loss plot as 'training_loss.pdf'")
    
    student_model.eval()
    with torch.no_grad():
        sample_noise = torch.randn(64, 3, 32, 32).to(device)
        generated_images = student_model(sample_noise)
        save_image_grid(generated_images, "generated_samples", nrow=8)
    print("Saved generated samples as 'generated_samples.pdf'")
    
    inference_time = measure_inference_time(student_model, sample_noise, device, num_runs=10)
    print(f"Average inference time (10 runs) per batch: {inference_time:.6f} sec")

def experiment_ablation_study(epochs=2, noise_std=0.1, batch_size=128, max_batches=5):
    """
    Experiment 2: Ablation Study on Momentum Memory Integration.
    
    Args:
        epochs (int): Number of epochs to train for.
        noise_std (float): Standard deviation of noise to add to images.
        batch_size (int): Batch size for training.
        max_batches (int): Maximum number of batches to use per epoch.
    """
    print("\nStarting Experiment 2: Ablation Study on Momentum Memory Integration")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader = get_dataloader(batch_size=batch_size, train=True)
    
    teacher_model = TeacherModel().to(device)
    teacher_model.eval()
    teacher_trajectories = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            images = images.to(device)
            traj = teacher_model.generate_trajectory(images, num_steps=20)
            teacher_trajectories.append(traj)
    
    student_with_momentum = OneStepStudent().to(device)
    student_without_momentum = SimpleOneStepStudent().to(device)
    
    criterion = nn.MSELoss()
    optimizer_mom = optim.Adam(student_with_momentum.parameters(), lr=1e-4)
    optimizer_no_mom = optim.Adam(student_without_momentum.parameters(), lr=1e-4)
    
    loss_history_mom = []
    loss_history_no_mom = []
    
    for epoch in range(epochs):
        student_with_momentum.train()
        student_without_momentum.train()
        running_loss_mom = 0.0
        running_loss_no_mom = 0.0
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx >= len(teacher_trajectories):
                break
            images = images.to(device)
            noisy_images = images + torch.randn_like(images) * noise_std
            
            out_mom = student_with_momentum(noisy_images)
            out_no_mom = student_without_momentum(noisy_images)
            teacher_target = teacher_trajectories[batch_idx]
            
            loss_mom = criterion(out_mom, teacher_target)
            loss_no_mom = criterion(out_no_mom, teacher_target)
            
            optimizer_mom.zero_grad()
            loss_mom.backward(retain_graph=True)
            optimizer_mom.step()
            
            optimizer_no_mom.zero_grad()
            loss_no_mom.backward(retain_graph=True)
            optimizer_no_mom.step()
            
            running_loss_mom += loss_mom.item()
            running_loss_no_mom += loss_no_mom.item()
        avg_loss_mom = running_loss_mom / len(teacher_trajectories)
        avg_loss_no_mom = running_loss_no_mom / len(teacher_trajectories)
        loss_history_mom.append(avg_loss_mom)
        loss_history_no_mom.append(avg_loss_no_mom)
        print(f"Epoch {epoch+1}: With Momentum Loss {avg_loss_mom:.4f}, Without Momentum Loss {avg_loss_no_mom:.4f}")
    
    torch.save(student_with_momentum.state_dict(), "models/asidm_with_momentum.pth")
    torch.save(student_without_momentum.state_dict(), "models/asidm_without_momentum.pth")
    
    save_loss_curve(
        list(range(1, epochs+1)), 
        [loss_history_mom, loss_history_no_mom], 
        "training_loss_ablation_pair1", 
        "Training Loss Comparison (Ablation Study)",
        multiple_curves=True,
        labels=["ASID-M", "Baseline (No Momentum)"]
    )
    print("Saved ablation study loss plot as 'training_loss_ablation_pair1.pdf'")
    
    student_with_momentum.eval()
    student_without_momentum.eval()
    with torch.no_grad():
        sample_noise = torch.randn(64, 3, 32, 32).to(device)
        generated_images_mom = student_with_momentum(sample_noise)
        generated_images_no_mom = student_without_momentum(sample_noise)
        
        save_image_grid(generated_images_mom, "generated_samples_with_momentum", nrow=8)
        save_image_grid(generated_images_no_mom, "generated_samples_without_momentum", nrow=8)
    
    print("Saved generated samples from both models")

def experiment_sensitivity_efficiency_analysis(epochs=2, batch_size=128, max_batches=5):
    """
    Experiment 3: Sensitivity and Efficiency Analysis under Varying Noise Levels.
    
    Args:
        epochs (int): Number of epochs to train for.
        batch_size (int): Batch size for training.
        max_batches (int): Maximum number of batches to use per epoch.
    """
    print("\nStarting Experiment 3: Sensitivity and Efficiency Analysis under Varying Noise Levels")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataloader = get_dataloader(batch_size=batch_size, train=True)
    
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    loss_vs_noise = {}
    inference_time_vs_noise = {}
    
    for noise_std in noise_levels:
        print(f"\nTesting with noise_std = {noise_std}")
        student_model = OneStepStudent().to(device)
        teacher_model = TeacherModel().to(device)
        teacher_model.eval()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
        
        loss_history = train_model(
            student_model, dataloader, teacher_model, criterion, optimizer,
            device, epochs=epochs, noise_std=noise_std, max_batches=max_batches
        )
        
        avg_loss = evaluate_model(
            student_model, dataloader, teacher_model, criterion, device,
            noise_std=noise_std, max_batches=max_batches
        )
        loss_vs_noise[noise_std] = avg_loss
        print(f"Average loss at noise level {noise_std:.2f}: {avg_loss:.4f}")
        
        student_model.eval()
        sample_batch = torch.randn(64, 3, 32, 32).to(device)
        time_taken = measure_inference_time(student_model, sample_batch, device)
        inference_time_vs_noise[noise_std] = time_taken
        print(f"Inference time per batch at noise level {noise_std:.2f}: {time_taken:.6f} sec")
    
    save_scatter_plot(
        list(loss_vs_noise.keys()), 
        list(loss_vs_noise.values()), 
        "loss_vs_noise", 
        "Loss vs Noise Level",
        "Noise Std",
        "Average Loss"
    )
    print("Saved Loss vs Noise plot as 'loss_vs_noise.pdf'")
    
    save_scatter_plot(
        list(inference_time_vs_noise.keys()), 
        list(inference_time_vs_noise.values()), 
        "inference_latency_pair1", 
        "Inference Time vs Noise Level",
        "Noise Std",
        "Inference Time (sec)"
    )
    print("Saved Inference Time vs Noise plot as 'inference_latency_pair1.pdf'")

def test_run():
    """
    A quick test to verify that the code can run end-to-end.
    This test uses only one epoch (or a few batches) in each experiment for a rapid check.
    """
    print("\n=== Starting Test Run ===")
    ensure_dirs_exist()
    
    experiment_controlled_image_generation(epochs=1, noise_std=0.1, max_batches=2)
    experiment_ablation_study(epochs=1, noise_std=0.1, max_batches=2)
    experiment_sensitivity_efficiency_analysis(epochs=1, max_batches=2)
    
    print("=== Test Run Finished ===")

if __name__ == "__main__":
    set_seed(42)
    
    ensure_dirs_exist()
    
    test_run()
    
    
    print("\nAll experiments completed.")
