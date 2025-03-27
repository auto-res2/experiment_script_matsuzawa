"""
DFAD Experiment - Model Evaluation Module

This module implements the evaluation metrics and functions for the DFAD experimental framework.
It includes harm detection, IOU computation, and other evaluation metrics.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from memory_profiler import memory_usage

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.dfad_config import PDF_DPI, TEST_MODE

def dummy_harmful_detector(image: Image.Image) -> (float, np.ndarray):
    """
    Dummy function: returns a random harm score and a dummy binary harm mask.
    In real experiments, this function should use a trained harmful-content classifier.
    
    Args:
        image (PIL.Image): Input image to analyze
        
    Returns:
        tuple: (harm_score, harm_mask)
            - harm_score (float): A value between 0 and 1 indicating harm level
            - harm_mask (np.ndarray): Binary mask highlighting harmful regions
    """
    if TEST_MODE:
        np.random.seed(42)
        
    harm_score = np.random.uniform(0, 1)
    
    harm_mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    
    h, w = harm_mask.shape
    for _ in range(np.random.randint(1, 4)):
        x = np.random.randint(0, w - 50)
        y = np.random.randint(0, h - 50)
        size = np.random.randint(20, 50)
        harm_mask[y:y+size, x:x+size] = 1
        
    return harm_score, harm_mask

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1 (np.ndarray): First binary mask
        mask2 (np.ndarray): Second binary mask
        
    Returns:
        float: IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0  # if both masks are empty, define IoU as 1
    return intersection / union

def plot_harm_scores(harm_scores, title="Harm Score Comparison", filename="harm_scores.pdf"):
    """
    Plot harm scores for different methods and save as PDF.
    
    Args:
        harm_scores (dict): Dictionary mapping method names to harm scores
        title (str): Title for the plot
        filename (str): Output filename
    """
    plt.figure(figsize=(6, 4))
    plt.bar(list(harm_scores.keys()), list(harm_scores.values()), color='skyblue')
    plt.xlabel("Method")
    plt.ylabel("Harm Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("logs", filename), dpi=PDF_DPI, format="pdf")
    plt.close()
    print(f"Saved harm score comparison plot as 'logs/{filename}'")

def plot_iou_scores(iou_scores, title="IoU Comparison", filename="iou_scores.pdf"):
    """
    Plot IoU scores for different methods and save as PDF.
    
    Args:
        iou_scores (dict): Dictionary mapping method names to IoU scores
        title (str): Title for the plot
        filename (str): Output filename
    """
    plt.figure(figsize=(6, 4))
    plt.bar(list(iou_scores.keys()), list(iou_scores.values()), color='salmon')
    plt.xlabel("Method")
    plt.ylabel("IoU with Ground-Truth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("logs", filename), dpi=PDF_DPI, format="pdf")
    plt.close()
    print(f"Saved IoU comparison plot as 'logs/{filename}'")

def plot_metric_comparison(metrics, metric_name, title, filename, color='violet'):
    """
    Plot comparison of arbitrary metrics across methods and save as PDF.
    
    Args:
        metrics (dict): Dictionary mapping method names to metric values
        metric_name (str): Name of the metric (for y-axis label)
        title (str): Title for the plot
        filename (str): Output filename
        color (str): Bar color
    """
    plt.figure(figsize=(6, 4))
    plt.bar(list(metrics.keys()), list(metrics.values()), color=color)
    plt.xlabel("Method/Variant")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("logs", filename), dpi=PDF_DPI, format="pdf")
    plt.close()
    print(f"Saved {metric_name} comparison plot as 'logs/{filename}'")

def generate_with_dfad(model_id, prompt, adapter):
    """
    Load a given diffusion model from its identifier, generate image with DFAD adapter,
    and return the generated image and elapsed time.
    In TEST_MODE, a dummy image is returned and a dummy elapsed time.
    
    Args:
        model_id (str): Model identifier
        prompt (str): Input prompt
        adapter: DFAD adapter instance
        
    Returns:
        tuple: (PIL.Image, float)
            - Generated image
            - Elapsed time in seconds
    """
    from train import generate_image
    
    if TEST_MODE:
        np.random.seed(42)
        start_time = time.time()
        img = generate_image(prompt, adapter=adapter, model_id=model_id)
        elapsed = time.time() - start_time + np.random.uniform(0.01, 0.05)  # dummy elapsed time
        return img, elapsed
    else:
        start_time = time.time()
        img = generate_image(prompt, adapter=adapter, model_id=model_id)
        elapsed = time.time() - start_time
        return img, elapsed

def measure_memory_usage(func, *args, **kwargs):
    """
    Measure memory usage of a function.
    
    Args:
        func: Function to measure
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        tuple: (function_result, max_memory_usage)
    """
    mem_usage, result = memory_usage(
        (func, args, kwargs), 
        retval=True,
        max_iterations=1
    )
    return result, max(mem_usage)
