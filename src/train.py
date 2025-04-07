"""
Training module for ANCD experiments.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.experiment_config import LEARNING_RATE, DEVICE, NUM_EPOCHS, TEST_MODE_EPOCHS, TEST_MODE_MAX_ITER
from utils.experiment_utils import save_plot
from utils.models import ANCDNetworkVariants

def run_experiment1(model, train_loader, test_mode=False, use_ancd=True):
    """
    Run Experiment 1: Efficiency and Memory Overhead Comparison.
    
    Args:
        model: Model to train
        train_loader: DataLoader with training data
        test_mode: If True, use test settings (fewer epochs/iterations)
        use_ancd: If True, use ANCD network else baseline
        
    Returns:
        results: Dictionary with experiment results
    """
    print("\n=== Running Experiment 1: Efficiency and Memory Overhead Comparison ===")
    print(f"Using {'ANCD' if use_ancd else 'Baseline triple'} network.")
    
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    num_epochs = TEST_MODE_EPOCHS if test_mode else NUM_EPOCHS
    epoch_losses = []
    epoch_times = []
    peak_memory = []  # in MB
    
    from config.experiment_config import MAX_BATCH_SIZE
    current_batch_size = train_loader.batch_size
    gradient_variance_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        epoch_gradient_variance = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        iter_count = 0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            
            if use_ancd and hasattr(model, 'shared'):
                with torch.no_grad():
                    shared_out = model.shared(inputs)
                    batch_gradient_variance = torch.mean(torch.abs(shared_out)).item()
                    epoch_gradient_variance += batch_gradient_variance
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # using reconstruction loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iter_count += 1
            
            if test_mode and iter_count >= TEST_MODE_MAX_ITER:
                break
                
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / iter_count
        
        mem = torch.cuda.max_memory_allocated(device)/1e6 if torch.cuda.is_available() else 0.0
        
    if use_ancd and hasattr(model, 'shared') and 'epoch_gradient_variance' in locals():
        avg_gradient_variance = epoch_gradient_variance / iter_count if iter_count > 0 else 0
        
        if 'gradient_variance_history' not in locals():
            gradient_variance_history = []
        gradient_variance_history.append(avg_gradient_variance)
        
        if len(gradient_variance_history) >= 2:
            variance_change = gradient_variance_history[-1] - gradient_variance_history[-2]
            
            if variance_change < -0.01 and current_batch_size < MAX_BATCH_SIZE:
                new_batch_size = min(current_batch_size * 2, MAX_BATCH_SIZE)
                print(f"Increasing batch size from {current_batch_size} to {new_batch_size}")
                current_batch_size = new_batch_size
                
                train_loader = torch.utils.data.DataLoader(
                    train_loader.dataset, 
                    batch_size=current_batch_size,
                    shuffle=True, 
                    num_workers=2
                )
        
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        peak_memory.append(mem)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Time: {epoch_time:.2f}s, Peak Mem: {mem:.2f}MB")
    
    epochs = np.arange(1, num_epochs+1)
    
    save_plot(
        epochs, epoch_losses, 
        "Epoch", "Training Loss", 
        "Training Loss", 
        f"logs/training_loss_exp1{'_ancd' if use_ancd else '_baseline'}.pdf"
    )
    
    save_plot(
        epochs, epoch_times, 
        "Epoch", "Time per Epoch (s)", 
        "Epoch Time", 
        f"logs/time_epoch_exp1{'_ancd' if use_ancd else '_baseline'}.pdf",
        color='green'
    )
    
    save_plot(
        epochs, peak_memory, 
        "Epoch", "Peak GPU Memory (MB)", 
        "Peak Memory Usage", 
        f"logs/memory_peak_exp1{'_ancd' if use_ancd else '_baseline'}.pdf",
        color='red'
    )
    
    print(f"Experiment 1 plots saved in logs/ directory")
    return {"loss": epoch_losses, "time": epoch_times, "memory": peak_memory}

def train_model_for_experiment2(model, train_loader, test_mode=False):
    """
    Train model for Experiment 2.
    
    Args:
        model: Model to train
        train_loader: DataLoader with training data
        test_mode: If True, use test settings (fewer epochs/iterations)
        
    Returns:
        model: Trained model
    """
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    num_epochs = TEST_MODE_EPOCHS if test_mode else NUM_EPOCHS
    
    from config.experiment_config import MAX_BATCH_SIZE
    current_batch_size = train_loader.batch_size
    gradient_variance_history = []
    
    for epoch in range(num_epochs):
        model.train()
        iter_count = 0
        epoch_gradient_variance = 0.0
        
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            
            if hasattr(model, 'shared'):
                with torch.no_grad():
                    shared_out = model.shared(inputs)
                    batch_gradient_variance = torch.mean(torch.abs(shared_out)).item()
                    epoch_gradient_variance += batch_gradient_variance
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            iter_count += 1
            
            if test_mode and iter_count >= TEST_MODE_MAX_ITER:
                break
        
        if hasattr(model, 'shared') and iter_count > 0:
            avg_gradient_variance = epoch_gradient_variance / iter_count
            gradient_variance_history.append(avg_gradient_variance)
            
            if len(gradient_variance_history) >= 2:
                variance_change = gradient_variance_history[-1] - gradient_variance_history[-2]
                
                if variance_change < -0.01 and current_batch_size < MAX_BATCH_SIZE:
                    new_batch_size = min(current_batch_size * 2, MAX_BATCH_SIZE)
                    print(f"Increasing batch size from {current_batch_size} to {new_batch_size}")
                    current_batch_size = new_batch_size
                    
                    train_loader = torch.utils.data.DataLoader(
                        train_loader.dataset, 
                        batch_size=current_batch_size,
                        shuffle=True, 
                        num_workers=2
                    )
                
        print(f"Epoch {epoch+1}/{num_epochs} completed.")
        
    return model

def train_variants_for_experiment3(variant_configs, train_loader, test_mode=False):
    """
    Train model variants for Experiment 3 (Ablation Study).
    
    Args:
        variant_configs: Dictionary of variant configurations
        train_loader: DataLoader with training data
        test_mode: If True, use test settings (fewer epochs/iterations)
        
    Returns:
        variant_losses: Dictionary of loss values for each variant
    """
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    
    num_epochs = TEST_MODE_EPOCHS if test_mode else NUM_EPOCHS
    variant_losses = {}
    
    for variant_name, model in variant_configs.items():
        print(f"\nTraining variant: {variant_name}")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        losses = []
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            iter_count = 0
            
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs, inputs)
                
                if isinstance(model, ANCDNetworkVariants) and model.use_consistency_loss:
                    from utils.experiment_utils import measure_consistency
                    cons_loss = measure_consistency(model, inputs)
                    loss = loss + 0.1 * cons_loss
                    
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                iter_count += 1
                
                if test_mode and iter_count >= TEST_MODE_MAX_ITER:
                    break
                    
            epoch_loss = running_loss / iter_count
            losses.append(epoch_loss)
            print(f"Variant {variant_name}, Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            
        variant_losses[variant_name] = losses
        
    return variant_losses
