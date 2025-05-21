#!/usr/bin/env python3
"""
Evaluation utilities for Iso-LWGAN experiments.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_isometric_regularizer(encoder, generator, data_loader, save_dir="./"):
    """
    Evaluate the effectiveness of isometric regularization by comparing
    pairwise distances in latent and generated spaces.
    """
    device = next(encoder.parameters()).device
    
    with torch.no_grad():
        x = next(iter(data_loader))[0].to(device)
        
        z = encoder(x)
        x_gen = generator(z)
        
        latent_dist = torch.cdist(z, z, p=2)
        gen_dist = torch.cdist(x_gen, x_gen, p=2)
        
        abs_diff = torch.abs(latent_dist - gen_dist)
        avg_diff = torch.mean(abs_diff).item()
        max_diff = torch.max(abs_diff).item()
        
        plt.figure(figsize=(6, 6))
        
        n = latent_dist.size(0)
        indices = torch.triu_indices(n, n, offset=1)
        latent_dist_flat = latent_dist[indices[0], indices[1]].cpu().numpy()
        gen_dist_flat = gen_dist[indices[0], indices[1]].cpu().numpy()
        
        plt.scatter(latent_dist_flat, gen_dist_flat, alpha=0.5)
        
        max_val = max(np.max(latent_dist_flat), np.max(gen_dist_flat))
        plt.plot([0, max_val], [0, max_val], 'r--')
        
        plt.title(f"Distance Preservation (Avg Diff: {avg_diff:.4f})")
        plt.xlabel("Latent Space Distances")
        plt.ylabel("Generated Space Distances")
        plt.tight_layout()
        
        os.makedirs(save_dir, exist_ok=True)
        pdf_filename = f"{save_dir}/distance_preservation.pdf"
        plt.savefig(pdf_filename, bbox_inches="tight", format="pdf")
        plt.close()
        
        print(f"Distance preservation evaluation:")
        print(f"  Average difference: {avg_diff:.4f}")
        print(f"  Maximum difference: {max_diff:.4f}")
        print(f"  Plot saved to {pdf_filename}")
        
    return avg_diff, max_diff

def evaluate_stochastic_generator(encoder, generator, data_loader, sigma_noise=0.1, n_samples=10, save_dir="./"):
    """
    Evaluate the stochastic generator by generating multiple samples from the same latent code
    and measuring the diversity of outputs.
    """
    device = next(encoder.parameters()).device
    
    with torch.no_grad():
        x = next(iter(data_loader))[0][:5].to(device)
        
        z = encoder(x)
        
        all_samples = []
        for i in range(x.size(0)):
            z_i = z[i:i+1]
            samples = []
            for j in range(n_samples):
                x_gen = generator(z_i, sigma_noise).cpu().numpy()[0]
                samples.append(x_gen)
            all_samples.append(np.array(samples))
        
        diversity_metrics = []
        for samples in all_samples:
            n = samples.shape[0]
            total_dist = 0.0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(samples[i] - samples[j])
                    total_dist += dist
                    count += 1
            avg_dist = total_dist / count if count > 0 else 0
            diversity_metrics.append(avg_dist)
        
        fig, axes = plt.subplots(1, len(all_samples), figsize=(4*len(all_samples), 4))
        if len(all_samples) == 1:
            axes = [axes]
            
        for i, (samples, metric) in enumerate(zip(all_samples, diversity_metrics)):
            ax = axes[i]
            ax.scatter(samples[:, 0], samples[:, 1], c='blue', alpha=0.7)
            ax.set_title(f"Sample {i+1}\nDiversity: {metric:.4f}")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
        
        plt.tight_layout()
        
        os.makedirs(save_dir, exist_ok=True)
        pdf_filename = f"{save_dir}/stochastic_diversity_sigma{sigma_noise}.pdf"
        plt.savefig(pdf_filename, bbox_inches="tight", format="pdf")
        plt.close()
        
        avg_diversity = np.mean(diversity_metrics)
        print(f"Stochastic generator evaluation (sigma={sigma_noise}):")
        print(f"  Average diversity: {avg_diversity:.4f}")
        print(f"  Plot saved to {pdf_filename}")
        
    return diversity_metrics

def evaluate_mnist_models(base_encoder, base_generator, iso_encoder, iso_generator, data_loader, save_dir="./"):
    """
    Compare the Base LWGAN and Iso-LWGAN models on MNIST data.
    """
    device = next(base_encoder.parameters()).device
    
    with torch.no_grad():
        x, labels = next(iter(data_loader))
        x = x.to(device)
        
        z_base = base_encoder(x)
        x_gen_base = base_generator(z_base)
        
        z_iso = iso_encoder(x)
        x_gen_iso = iso_generator(z_iso, sigma_noise=0.1)
        
        x_flat = x.view(x.size(0), -1)
        x_gen_base_flat = x_gen_base.view(x_gen_base.size(0), -1)
        x_gen_iso_flat = x_gen_iso.view(x_gen_iso.size(0), -1)
        
        base_recon_error = torch.mean(torch.norm(x_flat - x_gen_base_flat, dim=1)).item()
        iso_recon_error = torch.mean(torch.norm(x_flat - x_gen_iso_flat, dim=1)).item()
        
        n_samples = min(5, x.size(0))
        fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))
        
        for i in range(n_samples):
            axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f"Original {labels[i].item()}")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(x_gen_base[i, 0].cpu().numpy(), cmap='gray')
            axes[1, i].set_title("Base LWGAN")
            axes[1, i].axis('off')
            
            axes[2, i].imshow(x_gen_iso[i, 0].cpu().numpy(), cmap='gray')
            axes[2, i].set_title("Iso-LWGAN")
            axes[2, i].axis('off')
        
        plt.tight_layout()
        
        os.makedirs(save_dir, exist_ok=True)
        pdf_filename = f"{save_dir}/mnist_comparison.pdf"
        plt.savefig(pdf_filename, bbox_inches="tight", format="pdf")
        plt.close()
        
        print(f"MNIST model comparison:")
        print(f"  Base LWGAN reconstruction error: {base_recon_error:.4f}")
        print(f"  Iso-LWGAN reconstruction error: {iso_recon_error:.4f}")
        print(f"  Plot saved to {pdf_filename}")
        
    return base_recon_error, iso_recon_error
