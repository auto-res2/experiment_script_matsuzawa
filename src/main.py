import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from preprocess import load_image, generate_long_text, create_dummy_inputs
from train import DummyAMICTModel, DummyBaseModel, train_model
from evaluate import (
    experiment_multimodal_instruction,
    experiment_long_context,
    experiment_on_device_inference
)

def setup_environment():
    """Set up environment for the experiments."""
    os.makedirs("logs", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: GPU not available. Experiments may run slower on CPU.")
    
    return device

def load_config():
    """Load configuration for the experiments."""
    config = {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "log_interval": 1,
        "test_run": True,  # For quick test run
    }
    return config

def test_code_execution(device):
    """
    Test function to verify that core functions execute without errors.
    """
    print("\nRunning quick test of core functions...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Using simplified tokenization for tests.")
        tokenizer = None
    
    dummy_text = "Test text."
    dummy_image_tensor = load_image("non_existent_image.jpg")
    
    try:
        amict_model = DummyAMICTModel().to(device)
        base_model = DummyBaseModel().to(device)
        input_tensor = create_dummy_inputs().to(device)
        
        output_amict = amict_model(input_tensor)
        output_base = base_model(input_tensor)
        
        print(f"AMICT model output shape: {output_amict.shape}")
        print(f"Base model output shape: {output_base.shape}")
        print("Model tests passed.")
    except Exception as e:
        print(f"Error in model testing: {e}")
        return False
    
    print("All tests passed. Continuing with full experiment execution.\n")
    return True

def run_all_experiments(device, config):
    """
    Run all AMICT experiments sequentially.
    """
    print("Starting AMICT experiments simulation.")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        print("Using simplified tokenization for experiments.")
        tokenizer = lambda text, return_tensors=None: {"input_ids": torch.tensor([[1, 2, 3] * (len(text)//3 + 1)])}
    
    sample_texts = [
        "Describe the content of the image.",
        "What is the dominant color used?"
    ]
    sample_image_paths = [
        "sample_image1.jpg",  # Will generate dummy image
        "sample_image2.jpg"   # Will generate dummy image
    ]
    
    print("\nTraining AMICT and Base models...")
    amict_model = DummyAMICTModel().to(device)
    base_model = DummyBaseModel().to(device)
    
    if config['test_run']:
        config['epochs'] = 2
        print("Running in test mode with reduced epochs.")
    
    amict_model = train_model(amict_model, config, device)
    base_model = train_model(base_model, config, device)
    
    experiment_multimodal_instruction(tokenizer, sample_texts, sample_image_paths, load_image)
    experiment_long_context(tokenizer, generate_long_text)
    experiment_on_device_inference(device)
    
    print("\nAll AMICT experiments have been executed successfully.")

if __name__ == '__main__':
    start_time = time.time()
    
    print("=" * 80)
    print("AMICT (Adaptive Multimodal Instruction and Co-Training) Research Experiment")
    print("=" * 80)
    
    device = setup_environment()
    config = load_config()
    
    if test_code_execution(device):
        run_all_experiments(device, config)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print("=" * 80)
