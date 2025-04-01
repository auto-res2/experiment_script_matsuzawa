import os
import argparse
import torch
import json
import sys
import time
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
    print("=" * 80)
    print("=" * 30 + " MM-BTLM EXPERIMENT " + "=" * 30)
    print("=" * 80)
    
    print("\n📋 EXPERIMENT CONFIGURATION:")
    print(f"  • Test Mode: {'Enabled' if test_mode else 'Disabled'}")
    print(f"  • Model Architecture: {config['model']['name']}")
    print(f"  • Text Embedding Dimension: {config['model']['text_dim']}")
    print(f"  • Vision Embedding Dimension: {config['model']['vision_dim']}")
    print(f"  • Cross-Attention Heads: {config['model']['num_attention_heads']}")
    print(f"  • Adaptive Gating: {'Enabled for adaptive model' if config['model'].get('use_adaptive_gate') else 'Disabled'}")
    
    device = setup_device()
    print("\n💻 HARDWARE CONFIGURATION:")
    if device.type == "cuda":
        print(f"  • Device: {torch.cuda.get_device_name(0)}")
        print(f"  • CUDA Version: {torch.version.cuda}")
        print(f"  • Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  • PyTorch CUDA Available: {torch.cuda.is_available()}")
    else:
        print("  • Device: CPU")
        print(f"  • PyTorch Version: {torch.__version__}")
    
    print("\n📁 SETTING UP DIRECTORIES:")
    setup_directories(config)
    
    print("\n🔤 LOADING TOKENIZER:")
    print(f"  • Model: GPT2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f"  • Vocabulary Size: {len(tokenizer)}")
    
    print("\n📊 PREPARING DATA:")
    data = prepare_data(config, test_mode=test_mode)
    print(f"  • Training Samples: {len(data['train']['images'])}")
    print(f"  • Validation Samples: {len(data['val']['images'])}")
    print(f"  • Test Samples: {len(data['test']['images'])}")
    print(f"  • Image Shape: {tuple(data['train']['images'][0].shape)}")
    print(f"  • Text Sequence Length: {data['train']['text_ids'].shape[1]}")
    
    print("\n🧠 CREATING MODELS:")
    
    print("  • Creating MM-BTLM with Adaptive Gating:")
    config_adaptive = config.copy()
    config_adaptive['model']['use_adaptive_gate'] = True
    model_adaptive = create_model(config_adaptive, device)
    
    print("  • Creating MM-BTLM with Fixed Gating:")
    config_fixed = config.copy()
    config_fixed['model']['use_adaptive_gate'] = False
    model_fixed = create_model(config_fixed, device)
    
    if not test_mode or config.get('force_training', False):
        print("\n🏋️ TRAINING MODELS:")
        
        print("\n  • Training Adaptive Model:")
        print(f"    - Learning Rate: {config['training']['learning_rate']}")
        print(f"    - Batch Size: {config['training']['batch_size']}")
        print(f"    - Epochs: {config['training']['num_epochs']}")
        train_model(model_adaptive, data, config_adaptive, device, test_mode=test_mode)
        
        print("\n  • Training Fixed Model:")
        train_model(model_fixed, data, config_fixed, device, test_mode=test_mode)
    else:
        print("\n🏃 SKIPPING TRAINING (Test Mode Enabled)")
    
    print("\n📈 EVALUATING MODELS:")
    print("  • Evaluating Adaptive Model:")
    metrics_adaptive = evaluate_model(model_adaptive, data, device)
    
    print("  • Evaluating Fixed Model:")
    metrics_fixed = evaluate_model(model_fixed, data, device)
    
    print("\n🧪 RUNNING EXPERIMENTS:")
    
    print("\n  • EXPERIMENT 1: Multimodal Fusion and Cross-Attention")
    print("    Testing how multimodal fusion affects model performance")
    exp1_results = experiment_1_multimodal_fusion(
        model_adaptive, tokenizer, device, config, config['paths']['logs_dir']
    )
    
    print("\n  • EXPERIMENT 2: Adaptive Context and Modality Balancing")
    print("    Comparing adaptive vs. fixed gating mechanisms")
    exp2_results = experiment_2_adaptive_gating(
        model_adaptive, model_fixed, tokenizer, device, config, config['paths']['logs_dir']
    )
    
    print("\n  • EXPERIMENT 3: Edge-Device Inference and Quantization")
    print("    Measuring performance impact of model quantization")
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
    
    print("\n📝 EXPERIMENT SUMMARY:")
    print(f"  • Adaptive Model Test Accuracy: {metrics_adaptive['accuracy']:.2f}%")
    print(f"  • Fixed Model Test Accuracy: {metrics_fixed['accuracy']:.2f}%")
    print(f"  • Adaptive Model Inference Time: {metrics_adaptive['inference_time_ms']:.2f} ms")
    print(f"  • Fixed Model Inference Time: {metrics_fixed['inference_time_ms']:.2f} ms")
    
    if 'speedup' in exp3_results:
        print(f"  • Quantization Speedup: {exp3_results['speedup']:.2f}x")
    if 'size_reduction_percent' in exp3_results:
        print(f"  • Model Size Reduction: {exp3_results['size_reduction_percent']:.1f}%")
    
    print(f"\n💾 Results saved to: {results_path}")
    print("\n" + "=" * 80)
    print("=" * 30 + " EXPERIMENT COMPLETE " + "=" * 29)
    print("=" * 80)

def main():
    """
    Main entry point for the experiment.
    """
    print("\n" + "=" * 80)
    print("=" * 25 + " MM-BTLM EXPERIMENT RUNNER " + "=" * 25)
    print("=" * 80)
    
    parser = argparse.ArgumentParser(description='Run MM-BTLM experiment')
    parser.add_argument('--config', type=str, default='config/mm_btlm_config.json',
                        help='Path to configuration file')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with minimal data and iterations')
    args = parser.parse_args()
    
    print(f"\n🔧 Command-line Arguments:")
    print(f"  • Config Path: {args.config}")
    print(f"  • Test Mode: {'Enabled' if args.test else 'Disabled'}")
    
    print(f"\n📂 Loading configuration from: {args.config}")
    config = load_config(args.config)
    print(f"  • Configuration loaded successfully")
    
    print(f"\n🔍 Environment Information:")
    print(f"  • PyTorch Version: {torch.__version__}")
    print(f"  • CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  • CUDA Version: {torch.version.cuda}")
        print(f"  • GPU Device: {torch.cuda.get_device_name(0)}")
    
    print("\n🚀 Starting experiment execution...")
    start_time = time.time()
    run_experiment(config, test_mode=args.test)
    end_time = time.time()
    
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n⏱️ Total Execution Time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
