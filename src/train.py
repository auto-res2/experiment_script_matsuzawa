# Scripts for training models.
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from utils.models import HFIDModel, BaseMethodModel

def train_model(model, train_loader, val_loader, config, model_name="model"):
    """
    Train a model with the given configuration.
    
    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Configuration dictionary containing training parameters.
        model_name: Name for saving the model.
        
    Returns:
        trained_model: The trained model.
        history: Dictionary of training metrics.
    """
    print(f"Training {model_name}...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Set up tensorboard
    writer = SummaryWriter(log_dir=f"./logs/{model_name}")
    
    # Initialize tracking variables
    history = {'train_loss': [], 'val_loss': []}
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            _, loss_dict = model(data)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Log batch level metrics
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{config['num_epochs']}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Log epoch level metrics
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config['num_epochs']} completed in {epoch_time:.2f}s, Avg Loss: {avg_train_loss:.4f}")
        
        writer.add_scalar(f"{model_name}/train_loss", avg_train_loss, epoch)
        
        # Save model checkpoint
        if (epoch+1) % config['save_freq'] == 0:
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), f"./models/{model_name}_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), f"./models/{model_name}_final.pth")
    writer.close()
    
    return model, history
