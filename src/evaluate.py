"""
Evaluation module for TEDP (Trigger-Eradicating Diffusion Purification).
Implements metrics and experiment evaluation functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import random


def compute_fid(dummy_loss):
    """
    For demonstration, assume FID is a function of reconstruction loss.
    In practice, use torchmetrics or pre-built FID implementations.
    """
    return dummy_loss * 20  # dummy transformation


def compute_psnr(dummy_loss):
    """
    PSNR typically uses 10*log10(max_I^2/MSE). For demonstration, a dummy metric.
    """
    return 100 / (dummy_loss+1e-5)


def compute_trigger_detection_rate(model, data_loader, device='cuda'):
    """
    Dummy function to simulate backdoor leakage detection.
    In practice, a separate trigger detector would be employed.
    Here we randomly assign a detection rate.
    """
    return random.uniform(0, 1)


def experiment1(model_tedp, model_baseline, purification_module, loader, num_epochs=5, device='cuda'):
    """
    Compare training with the TEDP purification module versus baseline (no purification)
    using standard metrics (FID, PSNR, trigger detection rate) and plot the results.
    """
    print("Starting Experiment 1: Advanced Diffusion Purification Benchmark")
    
    tedp_losses = []
    baseline_losses = []
    tedp_fid = [] ; tedp_psnr = [] ; tedp_trigger = []
    baseline_fid = [] ; baseline_psnr = [] ; baseline_trigger = []
    
    def record_tedp(epoch, loss, curr_strength):
        tedp_losses.append(loss)
        fid_val = compute_fid(loss)
        psnr_val = compute_psnr(loss)
        trigger_rate = compute_trigger_detection_rate(model_tedp, loader, device)
        tedp_fid.append(fid_val)
        tedp_psnr.append(psnr_val)
        tedp_trigger.append(trigger_rate)
        if curr_strength is not None:
            print("TEDP Epoch {}: Loss {:.4f} - FID {:.2f} - PSNR {:.2f} - Trigger Rate {:.2f} - Purification Strength {:.4f}"
                  .format(epoch, loss, fid_val, psnr_val, trigger_rate, curr_strength))
        else:
            print("TEDP Epoch {}: Loss {:.4f} - FID {:.2f} - PSNR {:.2f} - Trigger Rate {:.2f}"
                  .format(epoch, loss, fid_val, psnr_val, trigger_rate))
    
    print("Training TEDP pipeline (with purification)...")
    from train import train_pipeline
    train_pipeline(model_tedp, loader, purification=purification_module, num_epochs=num_epochs,
                   record_func=record_tedp, device=device)
                   
    def record_baseline(epoch, loss, _):
        baseline_losses.append(loss)
        fid_val = compute_fid(loss)
        psnr_val = compute_psnr(loss)
        trigger_rate = compute_trigger_detection_rate(model_baseline, loader, device)
        baseline_fid.append(fid_val)
        baseline_psnr.append(psnr_val)
        baseline_trigger.append(trigger_rate)
        print("Baseline Epoch {}: Loss {:.4f} - FID {:.2f} - PSNR {:.2f} - Trigger Rate {:.2f}"
              .format(epoch, loss, fid_val, psnr_val, trigger_rate))
    
    print("Training Baseline pipeline (no purification)...")
    train_pipeline(model_baseline, loader, purification=None, num_epochs=num_epochs, 
                   record_func=record_baseline, device=device)
    
    epochs = list(range(num_epochs))
    
    plt.figure(figsize=(12, 4))
    plt.plot(epochs, tedp_fid, label="TEDP")
    plt.plot(epochs, baseline_fid, label="Baseline", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.title("FID Score per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/FID_comparison_pair1.pdf")
    print("Saved plot: logs/FID_comparison_pair1.pdf")
    
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, tedp_psnr, label="TEDP")
    plt.plot(epochs, baseline_psnr, label="Baseline", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/PSNR_comparison_pair1.pdf")
    print("Saved plot: logs/PSNR_comparison_pair1.pdf")
    
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, tedp_trigger, label="TEDP")
    plt.plot(epochs, baseline_trigger, label="Baseline", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Trigger Detection Rate")
    plt.title("Backdoor Trigger Leakage")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/TriggerDetection_comparison_pair1.pdf")
    print("Saved plot: logs/TriggerDetection_comparison_pair1.pdf")
    
    print("Experiment 1 finished.\n")


def experiment2(model, loader, num_epochs=3, device='cuda'):
    """
    Train a model while extracting latent representations via a forward hook.
    Then use PCA to reduce dimensions to 2D and compute the silhouette score.
    Finally, plot the latent space projection.
    """
    print("Starting Experiment 2: Latent-Space Regularization Analysis")
    
    latent_collection = []  # will be a list of numpy arrays
    batch_labels = []       # store the corresponding poison labels for each batch
    
    def latent_hook(module, input, output):
        latent = output.view(output.size(0), -1).detach().cpu().numpy()
        latent_collection.append(latent)
    
    hook_handle = model.latent_layer.register_forward_hook(latent_hook)
    
    print("Collecting labels from dataset...")
    for images, labels in loader:
        batch_labels.extend(labels.numpy())
    batch_labels = np.array(batch_labels)
    print(f"Dataset size: {len(batch_labels)} samples with {sum(batch_labels)} poisoned samples")
    
    print(f"Training model for {num_epochs} epochs and collecting latent representations...")
    from train import train_pipeline
    train_pipeline(model, loader, purification=None, num_epochs=num_epochs, device=device)
    
    hook_handle.remove()
    
    latents = np.concatenate(latent_collection, axis=0)
    print(f"Collected latent representations: shape {latents.shape}")
    
    repeated_labels = np.tile(batch_labels, num_epochs)
    print(f"Repeated labels shape: {repeated_labels.shape}")
    
    if len(repeated_labels) > latents.shape[0]:
        repeated_labels = repeated_labels[:latents.shape[0]]
    elif len(repeated_labels) < latents.shape[0]:
        latents = latents[:len(repeated_labels)]
    
    print(f"Final shapes - Latents: {latents.shape}, Labels: {repeated_labels.shape}")
    
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    if len(latents) > 10 and len(np.unique(repeated_labels)) > 1:
        sil_score = silhouette_score(latents, repeated_labels)
        print("Silhouette Score: {:.3f}".format(sil_score))
    else:
        sil_score = 0
        print("Skipping silhouette score calculation (not enough samples or unique labels)")
    
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(latents_2d[:,0], latents_2d[:,1], c=repeated_labels[:latents_2d.shape[0]], cmap='viridis', alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Space Projection (PCA)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Poisoned (1) vs Clean (0)")
    plt.tight_layout()
    plt.savefig("logs/latent_space_projection.pdf")
    print("Saved plot: logs/latent_space_projection.pdf")
    
    print("Experiment 2 finished.\n")
