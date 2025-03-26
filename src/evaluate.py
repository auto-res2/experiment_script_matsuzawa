"""
Scripts for evaluation.
Implements functions for evaluating models and visualizing results.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from utils.diffusion_utils import extract_features

def evaluate(net, test_loader, device):
    """
    Evaluate model accuracy on the test set.
    
    Args:
        net: PyTorch model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        accuracy: Accuracy percentage on test set
    """
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy



def visualize_features(net, test_loader, device, output_dir='./logs', filename_prefix='tsne'):
    """
    Extract features and visualize them using t-SNE.
    
    Args:
        net: PyTorch model
        test_loader: DataLoader for test data
        device: Device to run extraction on
        output_dir: Directory to save visualizations
        filename_prefix: Prefix for saved files
        
    Returns:
        silhouette: Silhouette score for the clustering
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting features...")
    features, labels = extract_features(net, test_loader, device)
    print(f"Extracted features with shape: {features.shape}")
    
    print("Applying t-SNE dimensionality reduction (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        x=features_2d[:, 0], 
        y=features_2d[:, 1],
        hue=labels, 
        palette="bright", 
        legend=False, 
        s=30, 
        alpha=0.7
    )
    plt.title(f"t-SNE Visualization of Features")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    
    tsne_path = os.path.join(output_dir, f"{filename_prefix}_visualization.png")
    plt.savefig(tsne_path)
    print(f"Saved t-SNE visualization to {tsne_path}")
    
    silhouette = silhouette_score(features, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    
    with open(os.path.join(output_dir, f"{filename_prefix}_metrics.txt"), 'w') as f:
        f.write(f"Silhouette Score: {silhouette:.4f}\n")
    
    return silhouette

def compare_experiments(results, output_dir='./logs'):
    """
    Create comparison plots for different mixup methods.
    
    Args:
        results: Dictionary with experiment results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(data['train_loss'], label=method)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_loss.png"))
    
    plt.figure(figsize=(10, 6))
    for method, data in results.items():
        plt.plot(data['test_acc'], label=method)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_accuracy.png"))
    
    with open(os.path.join(output_dir, "comparison_metrics.txt"), 'w') as f:
        f.write("Method\tFinal Loss\tFinal Acc\tBest Acc\n")
        for method, data in results.items():
            f.write(f"{method}\t{data['train_loss'][-1]:.4f}\t{data['test_acc'][-1]:.2f}\t{max(data['test_acc']):.2f}\n")
    
    print(f"Saved comparison plots to {output_dir}")

if __name__ == "__main__":
    from src.preprocess import get_dataloaders
    from src.train import get_model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    _, test_loader = get_dataloaders(batch_size=16, num_workers=0)
    net = get_model('resnet18', num_classes=100).to(device)
    
    accuracy = evaluate(net, test_loader, device)
    print(f"Test accuracy with untrained model: {accuracy:.2f}%")
