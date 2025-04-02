"""
Evaluation script for the MEAB-DG experiments.
Evaluates the trained models and generates plots.
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, AutoTokenizer, AutoModel

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import DualModalEncoder, DynamicContextModel
from src.preprocess import get_dataloaders, prepare_data
from src.utils.model_utils import get_device, set_seed
from src.utils.visualization_utils import (
    plot_comparison_bar, plot_confusion_matrix, save_plot
)

from config.experiment_config import (
    RANDOM_SEED, BATCH_SIZE, NUM_CLASSES, USE_AMP,
    TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM, FUSION_DIM
)

def evaluate_multimodal_gating(model_path="models/dual_modal_encoder.pt", logs_dir="logs"):
    """
    Evaluate the Dual Modal Encoder with Dynamic Gating.
    This implements evaluation for Experiment 1.
    
    Args:
        model_path: Path to the trained model
        logs_dir: Directory to save logs and plots
    
    Returns:
        Dict with evaluation results
    """
    print("\n" + "="*80)
    print("Evaluating Experiment 1: Dynamic Gating Mechanism for Multimodal Fusion")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    os.makedirs(logs_dir, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    dataloaders = get_dataloaders(tokenizer)
    multimodal_loader = dataloaders["multimodal"]
    
    model = DualModalEncoder(
        num_labels=NUM_CLASSES,
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        fusion_dim=FUSION_DIM
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: Model weights not found at {model_path}. Using untrained model.")
    
    model.eval()
    
    print("Evaluating with dynamic gating...")
    dynamic_correct = 0
    dynamic_total = 0
    dynamic_gate_values = []
    
    all_labels = []
    all_predictions_dynamic = []
    
    with torch.no_grad():
        for text_inputs, images, labels in multimodal_loader:
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            images = images.to(device)
            labels = labels.to(device)
            
            outputs, gate_values = model(text_inputs, images, use_dynamic_gate=True, return_gate=True)
            _, predicted = torch.max(outputs.data, 1)
            
            dynamic_total += labels.size(0)
            dynamic_correct += (predicted == labels).sum().item()
            
            text_gate = gate_values[0].mean().item()
            image_gate = gate_values[1].mean().item()
            dynamic_gate_values.append((text_gate, image_gate))
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions_dynamic.extend(predicted.cpu().numpy())
    
    dynamic_accuracy = dynamic_correct / dynamic_total
    
    print("Evaluating with simple average fusion (baseline)...")
    baseline_correct = 0
    baseline_total = 0
    
    all_predictions_baseline = []
    
    with torch.no_grad():
        for text_inputs, images, labels in multimodal_loader:
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(text_inputs, images, use_dynamic_gate=False)
            _, predicted = torch.max(outputs.data, 1)
            
            baseline_total += labels.size(0)
            baseline_correct += (predicted == labels).sum().item()
            
            all_predictions_baseline.extend(predicted.cpu().numpy())
    
    baseline_accuracy = baseline_correct / baseline_total
    
    print(f"Dynamic Gating Accuracy: {dynamic_accuracy:.4f}")
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Improvement: {(dynamic_accuracy - baseline_accuracy) * 100:.2f}%")
    
    avg_text_gate = np.mean([g[0] for g in dynamic_gate_values])
    avg_image_gate = np.mean([g[1] for g in dynamic_gate_values])
    print(f"Average Text Gate: {avg_text_gate:.4f}")
    print(f"Average Image Gate: {avg_image_gate:.4f}")
    
    methods = ["Simple Average", "Dynamic Gating"]
    accuracies = [baseline_accuracy, dynamic_accuracy]
    plot_comparison_bar(
        methods, 
        accuracies, 
        title="Experiment 1: Accuracy Comparison",
        ylabel="Accuracy",
        filename="experiment1_accuracy_comparison"
    )
    
    gate_types = ["Text Gate", "Image Gate"]
    gate_values = [avg_text_gate, avg_image_gate]
    plot_comparison_bar(
        gate_types, 
        gate_values, 
        title="Experiment 1: Average Gate Values",
        ylabel="Gate Value",
        filename="experiment1_gate_values"
    )
    
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for true, pred in zip(all_labels, all_predictions_dynamic):
        cm[true, pred] += 1
    
    plot_confusion_matrix(
        cm, 
        classes=list(range(NUM_CLASSES)),
        title="Experiment 1: Confusion Matrix (Dynamic Gating)",
        filename="experiment1_confusion_matrix"
    )
    
    return {
        "dynamic_accuracy": dynamic_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "improvement": dynamic_accuracy - baseline_accuracy,
        "avg_text_gate": avg_text_gate,
        "avg_image_gate": avg_image_gate
    }


def evaluate_context_splitting(model_path="models/dynamic_context_model.pt", logs_dir="logs"):
    """
    Evaluate the Dynamic Context Splitting Model.
    This implements evaluation for Experiment 2.
    
    Args:
        model_path: Path to the trained model
        logs_dir: Directory to save logs and plots
    
    Returns:
        Dict with evaluation results
    """
    print("\n" + "="*80)
    print("Evaluating Experiment 2: Dynamic Context Splitting for Long-Context Tasks")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    os.makedirs(logs_dir, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    transformer = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
    
    dataloaders = get_dataloaders(tokenizer)
    longtext_loader = dataloaders["longtext"]
    
    model = DynamicContextModel(transformer).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Warning: Model weights not found at {model_path}. Using untrained model.")
    
    model.eval()
    
    print("Evaluating with dynamic context splitting...")
    dynamic_mse = 0.0
    dynamic_total = 0
    
    with torch.no_grad():
        for texts, targets in longtext_loader:
            targets = targets.to(device)
            
            batch_outputs = []
            for text in texts:
                output = model(tokenizer, text)
                batch_outputs.append(output)
            
            outputs = torch.cat(batch_outputs, dim=0)
            mse = nn.MSELoss()(outputs.squeeze(), targets)
            
            dynamic_mse += mse.item() * targets.size(0)
            dynamic_total += targets.size(0)
    
    avg_dynamic_mse = dynamic_mse / dynamic_total
    
    print("Evaluating with direct processing (baseline)...")
    baseline_mse = 0.0
    baseline_total = 0
    
    with torch.no_grad():
        for texts, targets in longtext_loader:
            targets = targets.to(device)
            
            batch_outputs = []
            for text in texts:
                first_segment = text.split("\n\n")[0] if "\n\n" in text else text
                
                tokens = tokenizer(
                    first_segment, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding="max_length", 
                    max_length=128
                ).to(device)
                
                transformer_output = transformer(**tokens)
                first_token_emb = transformer_output.last_hidden_state[:, 0, :]
                
                output = model.head(first_token_emb)
                batch_outputs.append(output)
            
            outputs = torch.cat(batch_outputs, dim=0)
            mse = nn.MSELoss()(outputs.squeeze(), targets)
            
            baseline_mse += mse.item() * targets.size(0)
            baseline_total += targets.size(0)
    
    avg_baseline_mse = baseline_mse / baseline_total
    
    print(f"Dynamic Context Splitting MSE: {avg_dynamic_mse:.4f}")
    print(f"Baseline MSE: {avg_baseline_mse:.4f}")
    print(f"Improvement: {(avg_baseline_mse - avg_dynamic_mse):.4f}")
    
    methods = ["Direct Processing", "Context Splitting"]
    mse_values = [avg_baseline_mse, avg_dynamic_mse]
    plot_comparison_bar(
        methods, 
        mse_values, 
        title="Experiment 2: MSE Comparison",
        ylabel="Mean Squared Error",
        filename="experiment2_mse_comparison"
    )
    
    return {
        "dynamic_mse": avg_dynamic_mse,
        "baseline_mse": avg_baseline_mse,
        "improvement": avg_baseline_mse - avg_dynamic_mse
    }


def evaluate_edge_precision_scaling(logs_dir="logs"):
    """
    Evaluate the Edge-Aware Dynamic Precision Scaling.
    This implements evaluation for Experiment 3.
    
    Args:
        logs_dir: Directory to save logs and plots
    
    Returns:
        Dict with evaluation results
    """
    print("\n" + "="*80)
    print("Evaluating Experiment 3: Edge-Aware Dynamic Precision Scaling")
    print("="*80)
    
    set_seed(RANDOM_SEED)
    
    os.makedirs(logs_dir, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    dummy_text = "A sample text for testing edge-aware dynamic precision scaling."
    text_inputs = tokenizer(
        dummy_text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=128
    ).to(device)
    
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    model = DualModalEncoder(
        num_labels=NUM_CLASSES,
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        image_embedding_dim=IMAGE_EMBEDDING_DIM,
        fusion_dim=FUSION_DIM
    ).to(device)
    
    def measure_inference_time(model, inputs, modality, num_runs=10):
        """Measure inference time for a model."""
        model.eval()
        
        for _ in range(3):
            with torch.no_grad():
                if modality == "text":
                    _ = model(inputs, None, use_dynamic_gate=False)
                else:
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        _ = model(inputs, dummy_image, use_dynamic_gate=True)
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                if modality == "text":
                    _ = model(inputs, None, use_dynamic_gate=False)
                else:
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        _ = model(inputs, dummy_image, use_dynamic_gate=True)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    text_only_time = measure_inference_time(model, text_inputs, "text")
    print(f"Text-only inference time: {text_only_time:.6f} seconds")
    
    multimodal_time = measure_inference_time(model, text_inputs, "multimodal")
    print(f"Multimodal inference time with mixed precision: {multimodal_time:.6f} seconds")
    
    def get_memory_usage(model, inputs, modality):
        """Estimate memory usage for a model."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            if modality == "text":
                _ = model(inputs, None, use_dynamic_gate=False)
            else:
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    _ = model(inputs, dummy_image, use_dynamic_gate=True)
        
        memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        return memory_usage
    
    if torch.cuda.is_available():
        text_only_memory = get_memory_usage(model, text_inputs, "text")
        multimodal_memory = get_memory_usage(model, text_inputs, "multimodal")
        
        print(f"Text-only memory usage: {text_only_memory:.2f} MB")
        print(f"Multimodal memory usage with mixed precision: {multimodal_memory:.2f} MB")
        print(f"Memory savings: {(multimodal_memory - text_only_memory):.2f} MB")
        
        modalities = ["Text-only", "Multimodal (Mixed Precision)"]
        memory_values = [text_only_memory, multimodal_memory]
        plot_comparison_bar(
            modalities, 
            memory_values, 
            title="Experiment 3: Memory Usage Comparison",
            ylabel="Memory Usage (MB)",
            filename="experiment3_memory_comparison"
        )
    else:
        print("CUDA not available, skipping memory usage measurement.")
        text_only_memory = 0
        multimodal_memory = 0
    
    modalities = ["Text-only", "Multimodal (Mixed Precision)"]
    time_values = [text_only_time, multimodal_time]
    plot_comparison_bar(
        modalities, 
        time_values, 
        title="Experiment 3: Inference Time Comparison",
        ylabel="Inference Time (seconds)",
        filename="experiment3_inference_time"
    )
    
    speedup = text_only_time / multimodal_time if multimodal_time > 0 else 0
    
    return {
        "text_only_time": text_only_time,
        "multimodal_time": multimodal_time,
        "speedup": speedup,
        "text_only_memory": text_only_memory,
        "multimodal_memory": multimodal_memory,
        "memory_savings": multimodal_memory - text_only_memory
    }


if __name__ == "__main__":
    print("Testing evaluation module...")
    
    gating_results = evaluate_multimodal_gating()
    print(f"Evaluation completed. Results: {gating_results}")
    
    context_results = evaluate_context_splitting()
    print(f"Evaluation completed. Results: {context_results}")
    
    scaling_results = evaluate_edge_precision_scaling()
    print(f"Evaluation completed. Results: {scaling_results}")
