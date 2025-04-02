import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

class DummyAMICTModel(nn.Module):
    """
    Dummy model simulating a 3B parameter model (in a simplified manner).
    In practice, this would be replaced with the actual BTLMModel.
    """
    def __init__(self):
        super(DummyAMICTModel, self).__init__()
        self.linear = nn.Linear(768, 768)
    
    def forward(self, x):
        return self.linear(x)

class DummyBaseModel(nn.Module):
    """
    Dummy model simulating the base method model.
    """
    def __init__(self):
        super(DummyBaseModel, self).__init__()
        self.linear = nn.Linear(768, 768)
    
    def forward(self, x):
        return self.linear(x)

def train_model(model, config, device):
    """Simulate training a model based on the given configuration."""
    print(f"Training model on {device}...")
    
    for epoch in range(config['epochs']):
        time.sleep(0.2)  # Small delay to simulate computation
        train_loss = 1.0 / (epoch + 1)
        print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {train_loss:.4f}")
        
        if (epoch + 1) % config['log_interval'] == 0 or epoch == config['epochs'] - 1:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, epoch + 2), [1.0 / (e + 1) for e in range(epoch + 1)], marker='o', linestyle='-')
            plt.xlabel('Epoch')
            plt.ylabel('Training Loss')
            plt.title('Training Progress')
            plt.grid(True)
            plt.savefig(f"logs/training_progress_epoch_{epoch+1}.pdf", bbox_inches='tight')
            plt.close()
    
    return model

def run_amict(image_tensor, encoded_text):
    """
    Dummy AMICT model inference integrating image and text modalities.
    In practice, replace with an actual model call.
    """
    img_stat = float(torch.mean(image_tensor).item())
    txt_len = encoded_text.input_ids.shape[-1] if hasattr(encoded_text, 'input_ids') else len(encoded_text)
    response = f"AMICT response with image stat {img_stat:.3f} and text length {txt_len}."
    return response

def run_base_method(encoded_text):
    """
    Dummy Base Method that handles text only.
    """
    txt_len = encoded_text.input_ids.shape[-1] if hasattr(encoded_text, 'input_ids') else len(encoded_text)
    response = f"Base method response with text length {txt_len}."
    return response

def run_amict_text(encoded_text):
    """
    Dummy text generation for AMICT on long-context input.
    """
    token_count = encoded_text.input_ids.shape[-1] if hasattr(encoded_text, 'input_ids') else len(encoded_text)
    return f"AMICT generated text with context of {token_count} tokens."

def run_base_text(encoded_text):
    """
    Dummy text generation for Base Method on long-context input.
    """
    token_count = encoded_text.input_ids.shape[-1] if hasattr(encoded_text, 'input_ids') else len(encoded_text)
    return f"Base method generated text with context of {token_count} tokens."
