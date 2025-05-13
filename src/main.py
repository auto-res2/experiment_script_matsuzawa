"""
Main script for running IBGT experiments.

This script orchestrates the entire experimental workflow, from data preprocessing
to model training and evaluation. It runs three experiments:
1. Performance and Computational Efficiency Comparison
2. Ablation Study on the IB-Guided Triplet Filtering Module
3. Interpretability and Qualitative Analysis of Selected Triplets
"""
import os
import time
import torch
from utils.experiment_utils import set_random_seed
from config.ibgt_config import TRAIN_CONFIG, STATUS_CONFIG


def update_status(status):
    """Update the status_enum in the config file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "config", "ibgt_config.py")
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'status_enum' in line:
            lines[i] = f'    "status_enum": "{status}",  # Will be set to "stopped" after completion\n'
    
    with open(config_path, 'w') as f:
        f.writelines(lines)


def run_experiments():
    """Run all experiments."""
    print("=" * 80)
    print("INFORMATION BOTTLENECK GUIDED TRIPLET GRAPH TRANSFORMER (IBGT) EXPERIMENTS")
    print("=" * 80)
    
    os.makedirs("logs", exist_ok=True)
    
    set_random_seed(TRAIN_CONFIG["random_seed"])
    
    start_time = time.time()
    
    from evaluate import evaluate
    baseline_loss, ibgt_loss, variant_losses, important_triplets = evaluate()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total execution time: {duration:.2f} seconds")
    print(f"Device used: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Number of epochs: {TRAIN_CONFIG['num_epochs']}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Learning rate: {TRAIN_CONFIG['learning_rate']}")
    
    print("\nModel Performance:")
    print(f"- Baseline TGT Loss: {baseline_loss:.4f}")
    print(f"- IBGT Loss: {ibgt_loss:.4f}")
    for name, loss in variant_losses.items():
        print(f"- {name} Loss: {loss:.4f}")
    
    print(f"\nNumber of important triplets extracted: {len(important_triplets)}")
    print("\nAll plots saved in the logs directory.")
    
    update_status("stopped")
    print("\nStatus updated to 'stopped'.")
    
    print("\nExperiment completed successfully!")


def test_code():
    """Test function to quickly verify that the code executes."""
    print("Starting test_code() execution...")
    run_experiments()
    print("test_code() finished successfully.")


if __name__ == "__main__":
    test_code()
