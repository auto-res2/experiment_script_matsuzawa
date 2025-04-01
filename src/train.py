import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model import VisionEncoder, CrossAttentionFusion, AdaptiveGate, MM_BTLM, MM_BTLM_Adaptive
from utils.experiment import setup_device, save_lineplot

def train_model(model, data, config, device, test_mode=False):
    """
    Train the MM-BTLM model.
    
    Args:
        model: Model to train
        data: Dictionary containing training and validation data
        config: Configuration dictionary
        device: Device to train on
        test_mode: If True, run a minimal training loop for testing
        
    Returns:
        dict: Dictionary containing training history
    """
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate']
    )
    
    num_epochs = config['training']['num_epochs']
    if test_mode:
        num_epochs = 2
        print("Running in test mode with reduced epochs:", num_epochs)
    
    batch_size = config['training']['batch_size']
    train_data, val_data = data['train'], data['val']
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        num_samples = len(train_data['images'])
        num_batches = num_samples // batch_size + (1 if num_samples % batch_size != 0 else 0)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_images = train_data['images'][start_idx:end_idx].to(device)
            batch_text_ids = train_data['text_ids'][start_idx:end_idx].to(device)
            
            outputs = model(batch_text_ids, batch_images)
            
            loss = criterion(outputs[:, 0, :], train_data['labels'][start_idx:end_idx].to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs[:, 0, :], 1)
            train_total += end_idx - start_idx
            train_correct += (predicted == train_data['labels'][start_idx:end_idx].to(device)).sum().item()
        
        train_loss /= num_batches
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            num_val_samples = len(val_data['images'])
            num_val_batches = num_val_samples // batch_size + (1 if num_val_samples % batch_size != 0 else 0)
            
            for i in range(num_val_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_val_samples)
                
                batch_images = val_data['images'][start_idx:end_idx].to(device)
                batch_text_ids = val_data['text_ids'][start_idx:end_idx].to(device)
                
                outputs = model(batch_text_ids, batch_images)
                
                loss = criterion(outputs[:, 0, :], val_data['labels'][start_idx:end_idx].to(device))
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs[:, 0, :], 1)
                val_total += end_idx - start_idx
                val_correct += (predicted == val_data['labels'][start_idx:end_idx].to(device)).sum().item()
        
        val_loss /= num_val_batches
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    save_training_curves(history, config['paths']['logs_dir'])
    
    return history

def save_training_curves(history, log_dir):
    """
    Save training curves as PDF.
    
    Args:
        history: Dictionary containing training history
        log_dir: Directory to save the plots
    """
    epochs = list(range(1, len(history['train_loss']) + 1))
    save_lineplot(
        x_values=epochs,
        y_series=[history['train_loss'], history['val_loss']],
        labels=['Training Loss', 'Validation Loss'],
        title='Training and Validation Loss',
        xlabel='Epoch',
        ylabel='Loss',
        filename=os.path.join(log_dir, 'loss_curve.pdf')
    )
    
    save_lineplot(
        x_values=epochs,
        y_series=[history['train_acc'], history['val_acc']],
        labels=['Training Accuracy', 'Validation Accuracy'],
        title='Training and Validation Accuracy',
        xlabel='Epoch',
        ylabel='Accuracy (%)',
        filename=os.path.join(log_dir, 'accuracy_curve.pdf')
    )
    
    print(f"Training curves saved to {log_dir}")

def create_model(config, device):
    """
    Create the MM-BTLM model based on the configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to create the model on
        
    Returns:
        nn.Module: Created model
    """
    text_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    vision_encoder = VisionEncoder()
    cross_attn = CrossAttentionFusion(embed_dim=text_model.config.n_embd)
    
    vision_dim = 512  # ResNet18 output dimension
    text_dim = text_model.config.n_embd  # GPT2 embedding dimension
    
    if config['model']['use_adaptive_gate']:
        gate = AdaptiveGate(input_dim=vision_dim, output_dim=text_dim)
        model = MM_BTLM_Adaptive(text_model, vision_encoder, cross_attn, gate)
    else:
        model = MM_BTLM(text_model, vision_encoder, cross_attn)
    
    return model

if __name__ == "__main__":
    device = torch.device("cpu")
    
    config = {
        'model': {'use_adaptive_gate': True},
        'training': {'learning_rate': 1e-4, 'batch_size': 2, 'num_epochs': 1},
        'paths': {'logs_dir': 'logs'},
        'data': {'num_samples': 10, 'text_length': 5}
    }
    
    train_data = {
        'images': torch.rand(6, 3, 224, 224),
        'text_ids': torch.randint(0, 1000, (6, 5)),
        'labels': torch.randint(0, 5, (6,))
    }
    
    val_data = {
        'images': torch.rand(4, 3, 224, 224),
        'text_ids': torch.randint(0, 1000, (4, 5)),
        'labels': torch.randint(0, 5, (4,))
    }
    
    data = {'train': train_data, 'val': val_data}
    
    model = create_model(config, device)
    
    outputs = model(data['train']['text_ids'], data['train']['images'])
    print(f"Model output shape: {outputs.shape}")
    print("Training module test successful!")
