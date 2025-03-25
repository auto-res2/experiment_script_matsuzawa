# Scripts for evaluation.
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import (calculate_fid, LPIPS, compute_mig, 
                             compute_ssim, compute_perceptual_loss)

def evaluate_model(model, data_loader, config, model_name="model"):
    """
    Evaluate a model with various metrics.
    
    Args:
        model: The model to evaluate.
        data_loader: DataLoader for validation data.
        config: Configuration dictionary containing evaluation parameters.
        model_name: Name of the model for logging.
        
    Returns:
        metrics: Dictionary of evaluation metrics.
    """
    print(f"Evaluating {model_name}...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Generate images for evaluation
    num_gen_images = config.get('num_gen_images', 20)
    generated_images = model.generate(num_gen_images, device)
    
    # Get some real images for comparison
    real_images = next(iter(data_loader))[0][:num_gen_images].to(device)
    
    # Calculate metrics
    metrics = {}
    
    # FID score
    metrics['fid'] = calculate_fid(generated_images, data_loader.dataset)
    
    # LPIPS
    lpips_fn = LPIPS(net='alex')
    metrics['lpips'] = lpips_fn(generated_images, real_images).item()
    
    # Other metrics
    metrics['mig'] = compute_mig(model, data_loader)
    metrics['ssim'] = compute_ssim(model, data_loader)
    metrics['perceptual_loss'] = compute_perceptual_loss(model, data_loader)
    
    # Print metrics summary
    print(f"\nEvaluation metrics for {model_name}:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name.upper()}: {metric_value:.4f}")
    
    # Save visualization of generated images
    plt.figure(figsize=(10, 5))
    for i in range(min(5, num_gen_images)):
        plt.subplot(2, 5, i+1)
        img = generated_images[i].detach().cpu().permute(1, 2, 0).numpy()
        # Properly clip values to [0, 1] range to avoid matplotlib warnings
        img = np.clip((img + 1) / 2, 0, 1)
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(2, 5, i+6)
        img = real_images[i].detach().cpu().permute(1, 2, 0).numpy()
        # Properly clip values to [0, 1] range to avoid matplotlib warnings
        img = np.clip((img + 1) / 2, 0, 1)
        plt.imshow(img)
        plt.axis('off')
    
    os.makedirs('./logs', exist_ok=True)
    plt.savefig(f"./logs/{model_name}_samples.png")
    plt.close()
    
    return metrics
