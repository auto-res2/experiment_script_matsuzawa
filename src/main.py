import os
import torch
import argparse
import time

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_tcdp_models
from src.evaluate import experiment1, experiment2, experiment3
from config.experiment_config import TRAINING_CONFIG, EXPERIMENT_CONFIG

def verify_gpu_compatibility():
    """
    Verify that the code can run on NVIDIA Tesla T4 with 16GB VRAM
    """
    print("\n" + "="*50)
    print("GPU Compatibility Check")
    print("="*50)
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        print(f"CUDA is available. Device: {device_name}")
        
        if "T4" in device_name:
            print("Confirmed: Running on Tesla T4 GPU")
        else:
            print(f"Note: Running on {device_name}, not Tesla T4")
        
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"Created test tensor of shape {test_tensor.shape} on {test_tensor.device}")
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
    else:
        print("CUDA is not available. Running on CPU.")
        return False

def main(args):
    """
    Main function to run the experiments
    """
    start_time = time.time()
    
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    gpu_available = verify_gpu_compatibility()
    
    print("\n" + "="*50)
    print("TCDP Experiment Suite")
    print("="*50)
    print(f"Running on: {'GPU' if gpu_available else 'CPU'}")
    
    if args.train:
        print("\n" + "="*50)
        print("Training TCDP Models")
        print("="*50)
        train_tcdp_models(
            epochs=TRAINING_CONFIG['epochs'],
            batch_size=TRAINING_CONFIG['batch_size'],
            learning_rate=TRAINING_CONFIG['learning_rate']
        )
    
    if args.experiment1 or args.all:
        print("\n" + "="*50)
        experiment1(num_batches=args.num_batches)
    
    if args.experiment2 or args.all:
        print("\n" + "="*50)
        experiment2(num_batches=args.num_batches)
    
    if args.experiment3 or args.all:
        print("\n" + "="*50)
        experiment3(num_batches=args.num_batches)
    
    execution_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"Total execution time: {execution_time:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCDP Experiment Suite")
    parser.add_argument('--train', action='store_true', help='Train the models before running experiments')
    parser.add_argument('--experiment1', action='store_true', help='Run experiment 1: Robustness Under Adaptive Adversarial Attacks')
    parser.add_argument('--experiment2', action='store_true', help='Run experiment 2: Ablation Study – Evaluating Component Impact')
    parser.add_argument('--experiment3', action='store_true', help='Run experiment 3: Adaptive Noise Control Efficiency')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--num_batches', type=int, default=EXPERIMENT_CONFIG['test_batches'], help='Number of batches to process in each experiment')
    
    args = parser.parse_args()
    
    if not (args.experiment1 or args.experiment2 or args.experiment3 or args.all or args.train):
        args.all = True
    
    main(args)
