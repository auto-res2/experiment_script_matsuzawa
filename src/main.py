# The main experiment script for running HFID experiments.
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# Import local modules
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model
from utils.models import HFIDModel, BaseMethodModel
from utils.metrics import calculate_fid, LPIPS, compute_mig, compute_ssim, compute_perceptual_loss
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.default_config import config

def setup_environment():
    """
    Set up the experiment environment.
    """
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    return device

def experiment1(train_loader, val_loader, config, device):
    """
    Experiment 1: Global vs. Joint Quality and Disentanglement Comparison
    """
    print("\n" + "="*80)
    print("Experiment 1: Global vs. Joint Quality and Disentanglement Comparison")
    print("="*80)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir="./logs/experiment1")
    
    # Initialize models
    hfid_model = HFIDModel(use_isometry=True, use_consistency=True).to(device)
    base_model = BaseMethodModel().to(device)
    
    # Train models
    hfid_model, hfid_history = train_model(
        hfid_model, train_loader, val_loader, config, model_name="HFID"
    )
    
    base_model, base_history = train_model(
        base_model, train_loader, val_loader, config, model_name="BaseMethod"
    )
    
    # Evaluate models
    hfid_metrics = evaluate_model(hfid_model, val_loader, config, model_name="HFID")
    base_metrics = evaluate_model(base_model, val_loader, config, model_name="BaseMethod")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(hfid_history['train_loss'], label="HFID Loss")
    plt.plot(base_history['train_loss'], label="BaseMethod Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison (Experiment 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./logs/experiment1_loss.pdf")
    plt.close()
    
    print("\nExperiment 1 complete.")
    
    return hfid_model, base_model, hfid_metrics, base_metrics

def experiment2(train_loader, val_loader, config, device):
    """
    Experiment 2: Ablation Study on Hierarchical Structure Components
    """
    print("\n" + "="*80)
    print("Experiment 2: Ablation Study on Hierarchical Structure Components")
    print("="*80)
    
    results = {}
    
    # Train and evaluate different model variants
    for variant in config['variants']:
        model_name = variant['name']
        print(f"\nTraining variant: {model_name}")
        
        # Initialize model with specific configuration
        model = HFIDModel(
            use_isometry=variant['use_isometry'],
            use_consistency=variant['use_consistency']
        ).to(device)
        
        # Train model
        model, history = train_model(
            model, train_loader, val_loader, config, model_name=model_name
        )
        
        # Evaluate model
        metrics = evaluate_model(model, val_loader, config, model_name=model_name)
        
        # Store results
        results[model_name] = {
            'model': model,
            'history': history,
            'metrics': metrics
        }
    
    # Plot comparison of metrics
    labels = [variant['name'] for variant in config['variants']]
    ssim_vals = [results[name]['metrics']['ssim'] for name in labels]
    percept_vals = [results[name]['metrics']['perceptual_loss'] for name in labels]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x, ssim_vals, width, color='g', label='SSIM (higher is better)')
    ax1.set_ylabel('SSIM')
    ax1.set_ylim(0, 1.1)
    
    ax2 = ax1.twinx()
    ax2.bar(x + width, percept_vals, width, color='r', label='Perceptual Loss (lower is better)')
    ax2.set_ylabel('Perceptual Loss')
    
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title("Ablation Study: SSIM and Perceptual Loss")
    plt.tight_layout()
    plt.savefig("./logs/experiment2_ablation.pdf")
    plt.close()
    
    print("\nExperiment 2 complete.")
    
    return results

def experiment3(train_loader, val_loader, config, device):
    """
    Experiment 3: Computational Efficiency and Scalability Analysis
    """
    print("\n" + "="*80)
    print("Experiment 3: Computational Efficiency and Scalability Analysis")
    print("="*80)
    
    # Initialize models
    hfid_model = HFIDModel(use_isometry=True, use_consistency=True).to(device)
    base_model = BaseMethodModel().to(device)
    
    # Initialize optimizers
    hfid_optimizer = torch.optim.Adam(hfid_model.parameters(), lr=config['learning_rate'])
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=config['learning_rate'])
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir="./logs/experiment3")
    
    # Measure training time and memory usage
    hfid_times = []
    base_times = []
    hfid_memory = []
    base_memory = []
    
    # Training loop for HFID model
    print("\nMeasuring HFID model performance...")
    hfid_model.train()
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            hfid_optimizer.zero_grad()
            
            # Forward pass
            _, loss_dict = hfid_model(data)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            hfid_optimizer.step()
            
            # Log batch level metrics
            if batch_idx % 10 == 0:
                print(f"HFID - Epoch {epoch+1}/{config['num_epochs']}, Batch {batch_idx}/{len(train_loader)}")
        
        # Record time
        epoch_time = time.time() - start_time
        hfid_times.append(epoch_time)
        
        # Record memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9  # GB
            hfid_memory.append(memory_allocated)
            torch.cuda.empty_cache()
        
        print(f"HFID - Epoch {epoch+1} completed in {epoch_time:.2f}s")
        writer.add_scalar("HFID/epoch_time", epoch_time, epoch)
    
    # Training loop for Base model
    print("\nMeasuring Base model performance...")
    base_model.train()
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            base_optimizer.zero_grad()
            
            # Forward pass
            _, loss_dict = base_model(data)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            base_optimizer.step()
            
            # Log batch level metrics
            if batch_idx % 10 == 0:
                print(f"Base - Epoch {epoch+1}/{config['num_epochs']}, Batch {batch_idx}/{len(train_loader)}")
        
        # Record time
        epoch_time = time.time() - start_time
        base_times.append(epoch_time)
        
        # Record memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9  # GB
            base_memory.append(memory_allocated)
            torch.cuda.empty_cache()
        
        print(f"Base - Epoch {epoch+1} completed in {epoch_time:.2f}s")
        writer.add_scalar("Base/epoch_time", epoch_time, epoch)
    
    # Plot comparison of training times
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config['num_epochs']+1), hfid_times, 'b-', label='HFID')
    plt.plot(range(1, config['num_epochs']+1), base_times, 'r-', label='Base Method')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Training Time per Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./logs/experiment3_time.pdf')
    plt.close()
    
    # Plot comparison of memory usage
    if torch.cuda.is_available():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, config['num_epochs']+1), hfid_memory, 'b-', label='HFID')
        plt.plot(range(1, config['num_epochs']+1), base_memory, 'r-', label='Base Method')
        plt.xlabel('Epoch')
        plt.ylabel('Memory Usage (GB)')
        plt.title('GPU Memory Usage per Epoch')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./logs/experiment3_memory.pdf')
        plt.close()
    
    print("\nExperiment 3 complete.")
    
    return {
        'hfid_times': hfid_times,
        'base_times': base_times,
        'hfid_memory': hfid_memory,
        'base_memory': base_memory
    }

def main():
    """
    Main function to run the HFID experiments.
    """
    print("Starting HFID experiments...")
    print("="*80)
    print("Hierarchically Factorized Isometric Diffusion (HFID) Experiment")
    print("="*80)
    
    # Setup environment
    device = setup_environment()
    
    # Print experiment configuration
    print("\nExperiment Configuration:")
    print(f"  Image Size: {config['image_size']}x{config['image_size']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Number of Epochs: {config['num_epochs']}")
    print(f"  Training Samples: {config['train_samples']}")
    print(f"  Validation Samples: {config['val_samples']}")
    print(f"  Using Isometry: {config.get('use_isometry', True)}")
    print(f"  Using Consistency: {config.get('use_consistency', True)}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    train_loader, val_loader = preprocess_data(config)
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Run experiments based on configuration
    results = {}
    
    if config['run_experiment1']:
        print("\nRunning Experiment 1: Global vs. Joint Quality and Disentanglement Comparison")
        hfid_model, base_model, hfid_metrics, base_metrics = experiment1(
            train_loader, val_loader, config, device
        )
        results['experiment1'] = {
            'hfid_metrics': hfid_metrics,
            'base_metrics': base_metrics
        }
        
        # Print comparison summary
        print("\nExperiment 1 Summary:")
        print("  HFID vs BaseMethod Comparison:")
        print(f"  FID Score: {hfid_metrics['fid']:.2f} vs {base_metrics['fid']:.2f}")
        print(f"  SSIM: {hfid_metrics['ssim']:.2f} vs {base_metrics['ssim']:.2f}")
        print(f"  MIG: {hfid_metrics['mig']:.2f} vs {base_metrics['mig']:.2f}")
        print(f"  Perceptual Loss: {hfid_metrics['perceptual_loss']:.2f} vs {base_metrics['perceptual_loss']:.2f}")
    
    if config['run_experiment2']:
        print("\nRunning Experiment 2: Ablation Study on Hierarchical Structure Components")
        ablation_results = experiment2(
            train_loader, val_loader, config, device
        )
        results['experiment2'] = ablation_results
        
        # Print ablation summary
        print("\nExperiment 2 Summary:")
        print("  Ablation Study Results:")
        for variant in config['variants']:
            name = variant['name']
            metrics = ablation_results[name]['metrics']
            print(f"  {name}:")
            print(f"    FID: {metrics['fid']:.2f}")
            print(f"    SSIM: {metrics['ssim']:.2f}")
            print(f"    MIG: {metrics['mig']:.2f}")
            print(f"    Perceptual Loss: {metrics['perceptual_loss']:.2f}")
    
    if config.get('run_experiment3', False):
        print("\nRunning Experiment 3: Computational Efficiency and Scalability Analysis")
        efficiency_results = experiment3(
            train_loader, val_loader, config, device
        )
        results['experiment3'] = efficiency_results
        
        # Print efficiency summary
        print("\nExperiment 3 Summary:")
        print("  Computational Efficiency Results:")
        hfid_avg_time = sum(efficiency_results['hfid_times']) / len(efficiency_results['hfid_times'])
        base_avg_time = sum(efficiency_results['base_times']) / len(efficiency_results['base_times'])
        print(f"  Average Epoch Time - HFID: {hfid_avg_time:.2f}s, Base: {base_avg_time:.2f}s")
        
        if torch.cuda.is_available():
            hfid_avg_mem = sum(efficiency_results['hfid_memory']) / len(efficiency_results['hfid_memory'])
            base_avg_mem = sum(efficiency_results['base_memory']) / len(efficiency_results['base_memory'])
            print(f"  Average Memory Usage - HFID: {hfid_avg_mem:.2f}GB, Base: {base_avg_mem:.2f}GB")
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("Results saved to ./logs/ directory")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()
