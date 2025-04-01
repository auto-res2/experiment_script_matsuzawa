import os
import argparse
import torch
import json
import sys
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocess import prepare_data
from train import create_model, train_model
from evaluate import evaluate_model, experiment_1_multimodal_fusion, experiment_2_adaptive_gating, experiment_3_quantization
from utils.experiment import setup_device

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def setup_directories(config):
    """
    Create necessary directories if they don't exist.
    
    Args:
        config: Configuration dictionary
    """
    for dir_path in [
        config['paths']['logs_dir'],
        config['paths']['models_dir'],
        config['paths']['data_dir']
    ]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory {dir_path} is ready")

def run_experiment(config, test_mode=False):
    """
    Run the MM-BTLM experiment.
    
    Args:
        config: Configuration dictionary
        test_mode: If True, run a minimal version of the experiment for testing
    """
    print("=" * 50)
    print("Starting MM-BTLM Experiment")
    print("=" * 50)
    
    device = setup_device()
    
    setup_directories(config)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    print("\nPreparing data...")
    data = prepare_data(config, test_mode=test_mode)
    
    print("\nCreating models...")
    config_adaptive = config.copy()
    config_adaptive['model']['use_adaptive_gate'] = True
    model_adaptive = create_model(config_adaptive, device)
    
    config_fixed = config.copy()
    config_fixed['model']['use_adaptive_gate'] = False
    model_fixed = create_model(config_fixed, device)
    
    if not test_mode or config.get('force_training', False):
        print("\nTraining adaptive model...")
        train_model(model_adaptive, data, config_adaptive, device, test_mode=test_mode)
        
        print("\nTraining fixed model...")
        train_model(model_fixed, data, config_fixed, device, test_mode=test_mode)
    else:
        print("\nSkipping training in test mode")
    
    print("\nEvaluating models...")
    metrics_adaptive = evaluate_model(model_adaptive, data, device)
    metrics_fixed = evaluate_model(model_fixed, data, device)
    
    print("\nRunning experiments...")
    
    exp1_results = experiment_1_multimodal_fusion(
        model_adaptive, tokenizer, device, config, config['paths']['logs_dir']
    )
    
    exp2_results = experiment_2_adaptive_gating(
        model_adaptive, model_fixed, tokenizer, device, config, config['paths']['logs_dir']
    )
    
    exp3_results = experiment_3_quantization(
        model_adaptive, tokenizer, device, config, config['paths']['logs_dir']
    )
    
    results = {
        'metrics_adaptive': metrics_adaptive,
        'metrics_fixed': metrics_fixed,
        'experiment_1': exp1_results,
        'experiment_2': exp2_results,
        'experiment_3': exp3_results
    }
    
    results_path = os.path.join(config['paths']['logs_dir'], 'results.json')
    with open(results_path, 'w') as f:
        serializable_results = {}
        for exp_name, exp_results in results.items():
            serializable_results[exp_name] = {}
            for k, v in exp_results.items():
                if isinstance(v, (list, dict, str, int, float, bool, type(None))):
                    serializable_results[exp_name][k] = v
                else:
                    serializable_results[exp_name][k] = str(v)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nExperiment results saved to {results_path}")
    print("\nExperiment completed successfully!")

def main():
    """
    Main entry point for the experiment.
    """
    parser = argparse.ArgumentParser(description='Run MM-BTLM experiment')
    parser.add_argument('--config', type=str, default='config/mm_btlm_config.json',
                        help='Path to configuration file')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with minimal data and iterations')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    run_experiment(config, test_mode=args.test)

if __name__ == "__main__":
    main()
