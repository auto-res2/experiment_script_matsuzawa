"""
Training script for the MEAB-DG experiments.
Trains the models for the different experiments.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, AutoTokenizer, AutoModel
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import DualModalEncoder, DynamicContextModel
from src.preprocess import get_dataloaders, prepare_data
from src.utils.model_utils import get_device, count_parameters, set_seed
from src.utils.visualization_utils import plot_training_curve

from config.experiment_config import (
    RANDOM_SEED, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM, FUSION_DIM,
    NUM_CLASSES, USE_AMP, T4_CONFIG
)

def train_multimodal_gating(save_dir="models", logs_dir="logs"):
    """
    Train the Dual Modal Encoder with Dynamic Gating.
    This implements Experiment 1: Dynamic Gating Mechanism for Multimodal Fusion.
    
    Args:
        save_dir: Directory to save the trained model
        logs_dir: Directory to save logs and plots
    
    Returns:
        Dict with training results and model path
    """
    print("\n" + "="*80)
    print("Experiment 1: Training Dual Modal Encoder with Dynamic Gating")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    dataloaders = get_dataloaders(tokenizer)
    multimodal_loader = dataloaders["multimodal"]
    
    model = DualModalEncoder(
        num_labels=NUM_CLASSES,
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        fusion_dim=FUSION_DIM
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    train_losses = []
    val_accuracies = []
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (text_inputs, images, labels) in enumerate(multimodal_loader):
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(text_inputs, images, use_dynamic_gate=True)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}/{NUM_EPOCHS} | Batch {batch_idx}/{len(multimodal_loader)} | Loss: {loss.item():.4f}")
                
        avg_epoch_loss = epoch_loss / len(multimodal_loader)
        train_losses.append(avg_epoch_loss)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for text_inputs, images, labels in multimodal_loader:  # Using same data for demo
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(text_inputs, images, use_dynamic_gate=True)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        val_accuracies.append(accuracy)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{NUM_EPOCHS} completed in {epoch_time:.2f}s | "
              f"Loss: {avg_epoch_loss:.4f} | Accuracy: {accuracy:.4f}")
              
    model_path = f"{save_dir}/dual_modal_encoder.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    epochs_range = list(range(1, NUM_EPOCHS + 1))
    plot_training_curve(
        epochs_range, 
        train_losses, 
        val_accuracies,
        title="Experiment 1: Dynamic Gating Training Curve",
        filename="experiment1_training_curve"
    )
    
    return {
        "model_path": model_path,
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "num_epochs": NUM_EPOCHS
    }


def train_context_splitting(save_dir="models", logs_dir="logs"):
    """
    Train the Dynamic Context Splitting Model.
    This implements Experiment 2: Dynamic Context Splitting for Long-Context Tasks.
    
    Args:
        save_dir: Directory to save the trained model
        logs_dir: Directory to save logs and plots
    
    Returns:
        Dict with training results and model path
    """
    print("\n" + "="*80)
    print("Experiment 2: Training Dynamic Context Splitting Model")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    transformer = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    
    dataloaders = get_dataloaders(tokenizer)
    longtext_loader = dataloaders["longtext"]
    
    model = DynamicContextModel(transformer).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    train_losses = []
    val_mse_errors = []
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (texts, targets) in enumerate(longtext_loader):
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            batch_outputs = []
            for text in texts:
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    output = model(tokenizer, text)
                    batch_outputs.append(output)
            
            outputs = torch.cat(batch_outputs, dim=0)
            
            loss = criterion(outputs.squeeze(), targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}/{NUM_EPOCHS} | Batch {batch_idx}/{len(longtext_loader)} | Loss: {loss.item():.4f}")
                
        avg_epoch_loss = epoch_loss / len(longtext_loader)
        train_losses.append(avg_epoch_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for texts, targets in longtext_loader:  # Using same data for demo
                targets = targets.to(device)
                
                batch_outputs = []
                for text in texts:
                    output = model(tokenizer, text)
                    batch_outputs.append(output)
                
                outputs = torch.cat(batch_outputs, dim=0)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(longtext_loader)
        val_mse_errors.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{NUM_EPOCHS} completed in {epoch_time:.2f}s | "
              f"Train Loss: {avg_epoch_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
              
    model_path = f"{save_dir}/dynamic_context_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    epochs_range = list(range(1, NUM_EPOCHS + 1))
    plot_training_curve(
        epochs_range, 
        train_losses, 
        val_mse_errors,
        metric_name="MSE Error",
        title="Experiment 2: Dynamic Context Splitting Training Curve",
        filename="experiment2_training_curve"
    )
    
    return {
        "model_path": model_path,
        "train_losses": train_losses,
        "val_mse_errors": val_mse_errors,
        "num_epochs": NUM_EPOCHS
    }


if __name__ == "__main__":
    print("Testing training module...")
    
    train_results = train_multimodal_gating()
    print(f"Training completed. Results: {train_results}")
    
    context_results = train_context_splitting()
    print(f"Training completed. Results: {context_results}")
