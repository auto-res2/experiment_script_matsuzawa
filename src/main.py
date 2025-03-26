"""
Scripts for running the experiment.
Implements experiments for evaluating DiffuSynerMix and its variants.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

from preprocess import get_dataloaders
from train import get_model, train_model
from evaluate import evaluate, visualize_features, compare_experiments
from utils.diffusion_utils import (
    DiffuSynerMixModule, 
    get_diffusynermix_module
)
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.experiment_config import ExperimentConfig

os.makedirs('./logs', exist_ok=True)

def set_seed(seed):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def experiment_1(config):
    """
    Experiment 1: Benchmark Classification Comparison.
    Compare standard mixup, SynerMix, and DiffuSynerMix on CIFAR-100.
    
    Args:
        config: ExperimentConfig object with experiment parameters
    
    Returns:
        results: Dictionary with experiment results
    """
    print("\n========== Experiment 1: Benchmark Classification Comparison ==========")
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(config.SEED)
    
    train_loader, test_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR
    )
    
    results = {}
    
    for method in config.EXPERIMENT_1_MODES:
        print(f"\n--- Training with mixup mode: {method} ---")
        
        net = get_model(
            model_name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.NUM_EPOCHS
        )
        
        diffu_module = None
        if method == 'diffusynermix':
            diffu_module = DiffuSynerMixModule(
                in_channels=3,
                hidden_dim=config.DIFFU_HIDDEN_DIM,
                num_steps=config.DIFFU_NUM_STEPS,
                noise_std=config.DIFFU_NOISE_STD
            ).to(device)
        
        _, history = train_model(
            net=net,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.NUM_EPOCHS,
            device=device,
            mixup_mode=method,
            diffu_module=diffu_module,
            mixup_alpha=config.MIXUP_ALPHA,
            save_path=config.SAVE_PATH
        )
        
        results[method] = history
        
        print(f"\nVisualizing features for {method}...")
        visualize_features(
            net=net,
            test_loader=test_loader,
            device=device,
            output_dir='./logs',
            filename_prefix=f'experiment1_{method}'
        )
    
    compare_experiments(results, output_dir='./logs')
    
    return results

def experiment_2(config):
    """
    Experiment 2: Ablation Study on Diffusion Components.
    Vary the use of the direction predictor and the number of diffusion steps.
    
    Args:
        config: ExperimentConfig object with experiment parameters
    
    Returns:
        results: Dictionary with experiment results
    """
    print("\n========== Experiment 2: Ablation Study on Diffusion Components ==========")
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(config.SEED)
    
    train_loader, test_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR
    )
    
    results = {}
    
    for config_name, params in config.ABLATION_CONFIGS.items():
        print(f"\n--- Ablation configuration: {config_name} ---")
        
        net = get_model(
            model_name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.NUM_EPOCHS
        )
        
        diffu_module = get_diffusynermix_module(
            use_direction_predictor=params["use_direction_predictor"],
            num_steps=params["num_steps"],
            noise_std=config.DIFFU_NOISE_STD
        ).to(device)
        
        _, history = train_model(
            net=net,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.NUM_EPOCHS,
            device=device,
            mixup_mode='diffusynermix',
            diffu_module=diffu_module,
            mixup_alpha=config.MIXUP_ALPHA,
            save_path=config.SAVE_PATH
        )
        
        results[config_name] = history
        
        print(f"\nVisualizing features for ablation {config_name}...")
        visualize_features(
            net=net,
            test_loader=test_loader,
            device=device,
            output_dir='./logs',
            filename_prefix=f'experiment2_{config_name}'
        )
    
    compare_experiments(results, output_dir='./logs')
    
    return results

def experiment_3(config):
    """
    Experiment 3: Feature Space Visualization and Statistical Analysis.
    Extract features from the penultimate layer and apply t-SNE for visualization.
    
    Args:
        config: ExperimentConfig object with experiment parameters
    
    Returns:
        silhouette_scores: Dictionary with silhouette scores for each method
    """
    print("\n========== Experiment 3: Feature Space Visualization and Statistical Analysis ==========")
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(config.SEED)
    
    _, test_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        data_dir=config.DATA_DIR
    )
    
    silhouette_scores = {}
    
    for method in config.EXPERIMENT_1_MODES:
        model_path = os.path.join(config.SAVE_PATH, f'model_{method}_final.pth')
        
        if os.path.exists(model_path):
            print(f"\n--- Analyzing features for model trained with {method} ---")
            
            net = get_model(
                model_name=config.MODEL_NAME,
                num_classes=config.NUM_CLASSES,
                pretrained=False
            ).to(device)
            net.load_state_dict(torch.load(model_path))
            
            silhouette = visualize_features(
                net=net,
                test_loader=test_loader,
                device=device,
                output_dir='./logs',
                filename_prefix=f'experiment3_{method}'
            )
            
            silhouette_scores[method] = silhouette
        else:
            print(f"Model for {method} not found at {model_path}. Skipping...")
    
    with open(os.path.join('./logs', "experiment3_silhouette_scores.txt"), 'w') as f:
        f.write("Method\tSilhouette Score\n")
        for method, score in silhouette_scores.items():
            f.write(f"{method}\t{score:.4f}\n")
    
    return silhouette_scores

def run_test():
    """
    Run a quick test of all experiments with minimal settings.
    """
    print("\n********** Running quick test for all experiments **********")
    
    config = ExperimentConfig.get_test_config()
    
    start_time = time.time()
    
    experiment_1(config)
    
    experiment_2(config)
    
    experiment_3(config)
    
    end_time = time.time()
    print(f"\nQuick test completed in {end_time - start_time:.2f} seconds.")

def main():
    """
    Main function to run the experiments.
    """
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    sys.stdout = open('./logs/output.txt', 'w')
    sys.stderr = open('./logs/error.txt', 'w')
    
    try:
        config = ExperimentConfig()
        
        if config.TEST_MODE:
            run_test()
        else:
            print("\n========== Running full experiments ==========")
            
            experiment_1(config)
            
            experiment_2(config)
            
            experiment_3(config)
            
            print("\nAll experiments completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.close()
        sys.stderr.close()
        
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()
