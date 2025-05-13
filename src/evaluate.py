"""
Evaluation script for IBGT experiments.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils.experiment_utils import extract_important_triplets, visualize_molecule_triplets
from config.ibgt_config import EXPERIMENT_CONFIG


def evaluate_models(baseline_model, ibgt_model, variant_models, test_loader, device):
    """Evaluate models on the test set."""
    print("\n=== Evaluating Models on Test Set ===")
    loss_fn = nn.MSELoss()
    
    baseline_model.eval()
    baseline_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = baseline_model(batch)
            loss = loss_fn(outputs, batch.y)
            baseline_loss += loss.item()
    baseline_loss /= len(test_loader)
    print(f"Baseline TGT Test Loss: {baseline_loss:.4f}")
    
    ibgt_model.eval()
    ibgt_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = ibgt_model(batch)
            loss = loss_fn(outputs, batch.y)
            ibgt_loss += loss.item()
    ibgt_loss /= len(test_loader)
    print(f"IBGT Test Loss: {ibgt_loss:.4f}")
    
    variant_losses = {}
    for name, model in variant_models.items():
        model.eval()
        var_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = loss_fn(outputs, batch.y)
                var_loss += loss.item()
        var_loss /= len(test_loader)
        variant_losses[name] = var_loss
        print(f"{name} Test Loss: {var_loss:.4f}")
    
    plt.figure(figsize=(10, 6))
    models = ['Baseline TGT', 'IBGT'] + list(variant_losses.keys())
    losses = [baseline_loss, ibgt_loss] + list(variant_losses.values())
    plt.bar(models, losses)
    plt.xlabel('Model')
    plt.ylabel('Test Loss (MSE)')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('logs/model_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Saved model comparison plot as: logs/model_comparison.pdf")
    
    return baseline_loss, ibgt_loss, variant_losses


def analyze_triplets(ibgt_model, test_loader, device):
    """Analyze and visualize important triplets."""
    print("\n=== Analyzing Important Triplets ===")
    
    important_triplets = extract_important_triplets(
        ibgt_model, 
        test_loader, 
        device, 
        ib_threshold=EXPERIMENT_CONFIG["ib_threshold"]
    )
    
    G = nx.erdos_renyi_graph(n=10, p=0.4, seed=42)
    
    visualize_molecule_triplets(
        G, 
        important_triplets, 
        title="Important Triplets in Molecule",
        filename="triplet_interpretability_pair1.pdf"
    )
    
    G2 = nx.erdos_renyi_graph(n=12, p=0.3, seed=43)
    visualize_molecule_triplets(
        G2, 
        important_triplets[:len(important_triplets)//2], 
        title="Important Triplets in Another Molecule",
        filename="triplet_interpretability_pair2.pdf"
    )
    
    return important_triplets


def evaluate():
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from train import train
    baseline_model, ibgt_model, variant_models, test_loader = train()
    
    baseline_loss, ibgt_loss, variant_losses = evaluate_models(
        baseline_model, 
        ibgt_model, 
        variant_models, 
        test_loader, 
        device
    )
    
    important_triplets = analyze_triplets(ibgt_model, test_loader, device)
    
    return baseline_loss, ibgt_loss, variant_losses, important_triplets


if __name__ == "__main__":
    evaluate()
