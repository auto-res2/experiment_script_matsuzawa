"""
Main script for running the D-DAME experiments.

This script implements three experiments for the Dynamic Discriminative Anti-Memorization 
Ensemble (D-DAME) method:
1. Controlled comparison with fixed-threshold baseline
2. Ablation study on components of D-DAME
3. Sensitivity analysis of adaptive anti-gradient control
"""
import os
import sys
import time
import datetime

print("="*80)
print("D-DAME Experiment Script")
print("="*80)
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")
print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy not found. Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "requirements.txt")])
    print("Packages installed successfully.")
    import numpy as np
    print(f"NumPy version: {np.__version__}")

try:
    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
except ImportError:
    print("PyTorch not found. Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                          os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "requirements.txt")])
    print("Packages installed successfully.")
    import torch
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
except ImportError:
    print("Matplotlib not found. Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib>=3.4.0"])
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")

from preprocess import get_dataloader
from train import (
    UNet, DMRE, DDAMEWrapper,
    train_epoch_baseline, train_epoch_ddame
)
from evaluate import evaluate_model, compare_models
from utils.model_utils import set_seed, plot_metrics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.ddame_config import CONFIG

print("All imports successful.")
print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def experiment1():
    """
    Experiment 1: Controlled Comparison with Fixed-Threshold Baseline
    
    This experiment compares the baseline diffusion model with fixed threshold
    to the D-DAME approach.
    """
    print("Starting Experiment 1: Controlled Comparison with Fixed-Threshold Baseline")
    
    T = CONFIG['T']
    batch_size = CONFIG['batch_size']
    dataloader = get_dataloader(CONFIG['dataset'], batch_size=batch_size, num_workers=CONFIG['num_workers'])
    
    base_model_baseline = UNet(T, ch=CONFIG['ch']).to(device)
    optimizer_baseline = optim.Adam(base_model_baseline.parameters(), lr=CONFIG['lr'])
    
    base_model_ddame = UNet(T, ch=CONFIG['ch']).to(device)
    dmre_module = DMRE(input_dim=CONFIG['ch']).to(device)
    ddame_model = DDAMEWrapper(base_model_ddame, dmre_module).to(device)
    optimizer_ddame = optim.Adam(ddame_model.parameters(), lr=CONFIG['lr'])
    
    writer = SummaryWriter(log_dir='./logs/experiment1')
    
    num_epochs = CONFIG['num_epochs']
    max_iters = CONFIG['max_iters']
    baseline_metrics = {"loss": [], "grad_norm": []}
    ddame_metrics = {"loss": [], "grad_norm": [], "risk": [], "damping": []}
    
    for epoch in range(num_epochs):
        print(f"=== Experiment 1 Epoch {epoch} ===")
        m1 = train_epoch_baseline(base_model_baseline, optimizer_baseline, dataloader, T,
                                 lambda_threshold=0.5, epoch_num=epoch, writer=writer, max_iters=max_iters)
        m2 = train_epoch_ddame(ddame_model, optimizer_ddame, dataloader, T, use_probe=False,
                              epoch_num=epoch, writer=writer, max_iters=max_iters, variant="full")
        
        baseline_metrics["loss"] += m1["loss"]
        baseline_metrics["grad_norm"] += m1["grad_norm"]
        ddame_metrics["loss"] += m2["loss"]
        ddame_metrics["grad_norm"] += m2["grad_norm"]
        ddame_metrics["risk"] += m2["risk"]
        ddame_metrics["damping"] += m2["damping"]
    
    writer.close()
    
    torch.save(base_model_baseline.state_dict(), "models/baseline_model.pth")
    torch.save(ddame_model.state_dict(), "models/ddame_model.pth")
    
    compare_models(baseline_metrics, ddame_metrics, filename_prefix="experiment1")
    print("Experiment 1 Completed.\n")

def experiment2():
    """
    Experiment 2: Ablation Study on Components of D-DAME
    
    This experiment tests different variants of D-DAME to understand
    the contribution of each component.
    """
    print("Starting Experiment 2: Ablation Study on Components of D-DAME")
    
    T = CONFIG['T']
    batch_size = CONFIG['batch_size']
    dataloader = get_dataloader(CONFIG['dataset'], batch_size=batch_size, num_workers=CONFIG['num_workers'])
    writer = SummaryWriter(log_dir='./logs/experiment2')
    num_epochs = CONFIG['num_epochs']
    max_iters = CONFIG['max_iters']
    
    base_model1 = UNet(T, ch=CONFIG['ch']).to(device)
    dmre_module1 = DMRE(input_dim=CONFIG['ch']).to(device)
    model_full = DDAMEWrapper(base_model1, dmre_module1).to(device)
    optimizer_full = optim.Adam(model_full.parameters(), lr=CONFIG['lr'])
    
    base_model2 = UNet(T, ch=CONFIG['ch']).to(device)
    class DummyDMRE(torch.nn.Module):
        def forward(self, features):
            batch_size = features.size(0)
            risk = torch.zeros(batch_size,1, device=features.device)
            damping = torch.ones(batch_size,1, device=features.device)*0.5
            return risk, damping
    dummy_dmre = DummyDMRE().to(device)
    model_no_dmre = DDAMEWrapper(base_model2, dummy_dmre).to(device)
    optimizer_no_dmre = optim.Adam(model_no_dmre.parameters(), lr=CONFIG['lr'])
    
    base_model3 = UNet(T, ch=CONFIG['ch']).to(device)
    dmre_module3 = DMRE(input_dim=CONFIG['ch']).to(device)
    model_no_ensemble = DDAMEWrapper(base_model3, dmre_module3).to(device)
    optimizer_no_ensemble = optim.Adam(model_no_ensemble.parameters(), lr=CONFIG['lr'])
    
    metrics = {
        "full": {"loss": [], "risk": []},
        "no_dmre": {"loss": []},
        "no_ensemble": {"loss": [], "risk": []}
    }
    
    for epoch in range(num_epochs):
        print(f"=== Experiment 2 Epoch {epoch} ===")
        m_full = train_epoch_ddame(model_full, optimizer_full, dataloader, T, use_probe=False,
                                  epoch_num=epoch, writer=writer, max_iters=max_iters, variant="full")
        m_no_dmre = train_epoch_ddame(model_no_dmre, optimizer_no_dmre, dataloader, T, use_probe=False,
                                     epoch_num=epoch, writer=writer, max_iters=max_iters, variant="no_dmre")
        m_no_ensemble = train_epoch_ddame(model_no_ensemble, optimizer_no_ensemble, dataloader, T, use_probe=False,
                                         epoch_num=epoch, writer=writer, max_iters=max_iters, variant="full")  # single pass
        
        metrics["full"]["loss"] += m_full["loss"]
        metrics["full"]["risk"] += m_full["risk"]
        metrics["no_dmre"]["loss"] += m_no_dmre["loss"]
        metrics["no_ensemble"]["loss"] += m_no_ensemble["loss"]
        metrics["no_ensemble"]["risk"] += m_no_ensemble["risk"]
    
    writer.close()
    
    plot_metrics({"Full D-DAME Loss": metrics["full"]["loss"],
                  "No DMRE Loss": metrics["no_dmre"]["loss"],
                  "No Ensemble Loss": metrics["no_ensemble"]["loss"]}, 
                 filename_prefix="experiment2_loss")
    plot_metrics({"Full D-DAME Risk": metrics["full"]["risk"],
                  "No Ensemble Risk": metrics["no_ensemble"]["risk"]}, 
                 filename_prefix="experiment2_risk")
    print("Experiment 2 Completed.\n")

def experiment3():
    """
    Experiment 3: Sensitivity Analysis of Adaptive Anti-Gradient Control
    
    This experiment tests the sensitivity of D-DAME to different probe intensities,
    which simulate memorization tendencies.
    """
    print("Starting Experiment 3: Sensitivity Analysis of Adaptive Anti-Gradient Control")
    
    T = CONFIG['T']
    batch_size = CONFIG['batch_size']
    dataloader = get_dataloader(CONFIG['dataset'], batch_size=batch_size, num_workers=CONFIG['num_workers'])
    writer = SummaryWriter(log_dir='./logs/experiment3')
    
    base_model = UNet(T, ch=CONFIG['ch']).to(device)
    dmre_module = DMRE(input_dim=CONFIG['ch']).to(device)
    ddame_model = DDAMEWrapper(base_model, dmre_module).to(device)
    optimizer = optim.Adam(ddame_model.parameters(), lr=CONFIG['lr'])
    
    metrics = {"loss": [], "grad_norm": [], "risk": [], "damping": []}
    
    probe_intensities = [0.0, 0.1, 0.2, 0.3]
    epoch = 0
    for probe_intensity in probe_intensities:
        print(f"--- Sensitivity Analysis with probe_intensity = {probe_intensity} ---")
        m = train_epoch_ddame(ddame_model, optimizer, dataloader, T, use_probe=(probe_intensity>0),
                             probe_intensity=probe_intensity, epoch_num=epoch, writer=writer, 
                             max_iters=CONFIG['max_iters'], variant="full")
        metrics["loss"] += m["loss"]
        metrics["grad_norm"] += m["grad_norm"]
        metrics["risk"] += m["risk"]
        metrics["damping"] += m["damping"]
    
    writer.close()
    
    plot_metrics({"Loss": metrics["loss"]}, filename_prefix="experiment3_loss")
    plot_metrics({"Grad Norm": metrics["grad_norm"]}, filename_prefix="experiment3_gradnorm")
    plot_metrics({"Risk": metrics["risk"],
                  "Damping": metrics["damping"]}, filename_prefix="experiment3_risk_damping")
    print("Experiment 3 Completed.\n")

def test_experiment():
    """
    Quick test to ensure that all the code executes. 
    The test runs one or two mini-iterations for each experiment.
    """
    print("\n" + "="*80)
    print("STARTING D-DAME TEST EXPERIMENTS")
    print("="*80)
    
    print("Setting random seed for reproducibility...")
    set_seed(42)
    
    print("Configuring experiment parameters...")
    CONFIG['max_iters'] = 2  # Override to use fewer iterations
    CONFIG['num_epochs'] = 1  # Override to use fewer epochs
    print(f"Using configuration: max_iters={CONFIG['max_iters']}, num_epochs={CONFIG['num_epochs']}")
    
    try:
        print("\n" + "-"*40)
        print("EXPERIMENT 1: Controlled Comparison")
        print("-"*40)
        start_time = time.time()
        experiment1()
        exp1_time = time.time() - start_time
        print(f"Experiment 1 completed in {exp1_time:.2f} seconds")
        
        print("\n" + "-"*40)
        print("EXPERIMENT 2: Ablation Study")
        print("-"*40)
        start_time = time.time()
        experiment2()
        exp2_time = time.time() - start_time
        print(f"Experiment 2 completed in {exp2_time:.2f} seconds")
        
        print("\n" + "-"*40)
        print("EXPERIMENT 3: Sensitivity Analysis")
        print("-"*40)
        start_time = time.time()
        experiment3()
        exp3_time = time.time() - start_time
        print(f"Experiment 3 completed in {exp3_time:.2f} seconds")
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print(f"Total runtime: {exp1_time + exp2_time + exp3_time:.2f} seconds")
        print("Results saved to:")
        print("  - PDF plots: experiment*_metrics.pdf")
        print("  - Model checkpoints: models/")
        print("  - TensorBoard logs: logs/")
        print("="*80)
    except Exception as e:
        print("\n" + "!"*80)
        print(f"ERROR DURING EXPERIMENTS: {e}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        print("!"*80)
        raise

if __name__ == "__main__":
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
        
    test_experiment()
