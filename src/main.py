"""
Main script for running IBGT experiments.

This script orchestrates the entire experimental workflow, from data preprocessing
to model training and evaluation. It runs three experiments:
1. Performance and Computational Efficiency Comparison
2. Ablation Study on the IB-Guided Triplet Filtering Module
3. Interpretability and Qualitative Analysis of Selected Triplets
"""
import os
import sys
import time
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

print(f"Python path: {sys.path}")
print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")

from utils.experiment_utils import set_random_seed
from config.ibgt_config import TRAIN_CONFIG, STATUS_CONFIG, MODEL_CONFIG

print("\nSuccessfully imported configuration:")
print(f"- Training epochs: {TRAIN_CONFIG['num_epochs']}")
print(f"- Batch size: {TRAIN_CONFIG['batch_size']}")
print(f"- Learning rate: {TRAIN_CONFIG['learning_rate']}")
print(f"- Current status: {STATUS_CONFIG['status_enum']}")


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
    
    import platform
    
    print("\n--- SYSTEM INFORMATION ---")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    
    try:
        import psutil
        print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
    except ImportError:
        print("Available memory: [psutil not installed]")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    else:
        print("GPU: Not available")
    
    print("\n--- EXPERIMENT CONFIGURATION ---")
    print(f"Random seed: {TRAIN_CONFIG['random_seed']}")
    print(f"Number of epochs: {TRAIN_CONFIG['num_epochs']}")
    print(f"Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"Model height: {MODEL_CONFIG['model_height']}")
    print(f"Layer multiplier: {MODEL_CONFIG['layer_multiplier']}")
    print(f"Node width: {MODEL_CONFIG['node_width']}")
    print(f"Edge width: {MODEL_CONFIG['edge_width']}")
    
    os.makedirs("logs", exist_ok=True)
    print("\nCreated logs directory for saving experiment results.")
    
    set_random_seed(TRAIN_CONFIG["random_seed"])
    print("Set random seed for reproducibility.")
    
    start_time = time.time()
    print("\n--- STARTING EXPERIMENTS ---")
    
    print("Importing evaluate module...")
    from evaluate import evaluate
    
    print("\n--- EXPERIMENT 1: Performance and Computational Efficiency Comparison ---")
    print("Training and evaluating baseline TGT model and IBGT model...")
    
    print("\n--- EXPERIMENT 2: Ablation Study on IB-Guided Triplet Filtering Module ---")
    print("Training and evaluating model variants: Full IBGT, No Filter, Fixed Threshold...")
    
    print("\n--- EXPERIMENT 3: Interpretability and Qualitative Analysis ---")
    print("Extracting important triplets and generating visualizations...")
    
    print("\nRunning full evaluation pipeline...")
    baseline_loss, ibgt_loss, variant_losses, important_triplets = evaluate()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total execution time: {duration:.2f} seconds")
    print(f"Device used: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Baseline TGT Loss: {baseline_loss:.4f}")
    print(f"IBGT Loss: {ibgt_loss:.4f} ({((baseline_loss - ibgt_loss) / baseline_loss * 100):.2f}% improvement)")
    
    print("\n--- ABLATION STUDY RESULTS ---")
    for name, loss in variant_losses.items():
        improvement = (baseline_loss - loss) / baseline_loss * 100
        print(f"{name} Loss: {loss:.4f} ({improvement:.2f}% improvement over baseline)")
    
    print("\n--- INTERPRETABILITY ANALYSIS ---")
    print(f"Number of important triplets extracted: {len(important_triplets)}")
    print("Triplet importance distribution:")
    
    if important_triplets:
        scores = [score for _, _, score in important_triplets]
        print(f"  Min score: {min(scores):.4f}")
        print(f"  Max score: {max(scores):.4f}")
        print(f"  Mean score: {sum(scores) / len(scores):.4f}")
    
    print("\n--- VISUALIZATION OUTPUTS ---")
    print("Generated visualizations:")
    print("  - Model comparison plot: logs/model_comparison.pdf")
    print("  - Triplet interpretability plots:")
    print("    - logs/triplet_interpretability_pair1.pdf")
    print("    - logs/triplet_interpretability_pair2.pdf")
    
    update_status("stopped")
    print("\nStatus updated to 'stopped'.")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


def test_code():
    """Test function to quickly verify that the code executes."""
    print("Starting test_code() execution...")
    run_experiments()
    print("test_code() finished successfully.")


if __name__ == "__main__":
    test_code()
