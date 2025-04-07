"""
Preprocessing module for ABACR experiments.
Handles data loading, tokenization, and preparation for training and evaluation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextDataset(Dataset):
    """Dataset class for text data processing and tokenization."""
    def __init__(self, texts, tokenizer, max_length=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx],
                                  truncation=True,
                                  max_length=self.max_length,
                                  padding="max_length",
                                  return_tensors="pt")
        return {
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0)
        }

def prepare_dataset(texts, tokenizer_name, max_length=64, batch_size=8):
    """Prepare dataset for training and evaluation.
    
    Args:
        texts: List of text samples
        tokenizer_name: Name of the HuggingFace tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size for dataloaders
        
    Returns:
        train_loader: DataLoader for training
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    dataset = TextDataset(texts, tokenizer, max_length)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, tokenizer

def generate_long_context(prompt, extra_text, tokenizer, total_length=150):
    """Create long context by concatenating extra text until desired token count.
    
    Args:
        prompt: Base prompt text
        extra_text: Text to append repeatedly
        tokenizer: Tokenizer to use for token counting
        total_length: Target token length
        
    Returns:
        long_context: Generated long context text
    """
    long_context = prompt
    while len(tokenizer.tokenize(long_context)) < total_length:
        long_context += " " + extra_text
    return long_context

def get_sample_data():
    """Get sample text data for quick testing."""
    return [
        "Sample text data for training the model.",
        "Another example of training text.",
        "Further data for analysis.",
        "More texts to simulate a mini dataset."
    ]
