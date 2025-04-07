"""
Evaluation module for ABACR experiments.

Includes:
1. Bias/Toxicity evaluation using Detoxify
2. Long-context extrapolation tests
3. Ablation study evaluation
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from detoxify import Detoxify

def evaluate_toxicity(model, tokenizer, prompts, output_dir):
    """Evaluate model's toxicity on potentially problematic prompts.
    
    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        prompts: List of potentially problematic prompts
        output_dir: Directory to save results
        
    Returns:
        df: DataFrame with toxicity scores
    """
    toxicity_detector = Detoxify("original")
    
    def get_model_output(model, prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        outputs = model.generate(inputs.input_ids, max_length=100)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    
    model_outputs = []
    print("Generating outputs and evaluating toxicity scores...")
    
    for prompt in prompts:
        out = get_model_output(model, prompt)
        model_outputs.append(out)
        print(f"Prompt: {prompt}")
        print(f"Output: {out}\n")
    
    toxicity_scores = []
    for text in model_outputs:
        score = toxicity_detector.predict(text)["toxicity"]
        toxicity_scores.append(score)
    
    df = pd.DataFrame({
        "Prompt": prompts,
        "Output": model_outputs,
        "Toxicity": toxicity_scores
    })
    print("Toxicity scores:")
    print(df)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(prompts)), df["Toxicity"])
    plt.xlabel("Prompt Index")
    plt.ylabel("Toxicity Score")
    plt.title("Toxicity Scores")
    plt.tight_layout()
    filename = os.path.join(output_dir, "toxicity_scores.pdf")
    plt.savefig(filename, format='pdf', dpi=300)
    print(f"Plot saved as {filename}")
    plt.close()
    
    return df

def evaluate_long_context(model, tokenizer, prompt, filler, output_dir, max_lengths=[50, 100, 150, 200]):
    """Evaluate model's performance on increasingly long contexts.
    
    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        prompt: Base prompt to use
        filler: Text to append repeatedly for creating long contexts
        output_dir: Directory to save results
        max_lengths: List of context lengths to evaluate
        
    Returns:
        perplexities: Dictionary mapping context lengths to perplexities
    """
    def evaluate_perplexity(model, input_text):
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss.item()
        perplexity = np.exp(loss)
        return perplexity
    
    from preprocess import generate_long_context
    
    perplexities = {}
    context_texts = {}
    
    for length in max_lengths:
        long_input = generate_long_context(prompt, filler, tokenizer, total_length=length)
        context_texts[length] = long_input
        perplexity = evaluate_perplexity(model, long_input)
        perplexities[length] = perplexity
        print(f"Context length {length}: Perplexity = {perplexity}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(list(perplexities.keys()), list(perplexities.values()), 'o-')
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs. Context Length")
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, "long_context_perplexity.pdf")
    plt.savefig(filename, format='pdf', dpi=300)
    print(f"Perplexity plot saved as {filename}")
    plt.close()
    
    for length in [max(max_lengths)]:  # Just use the longest context for generation
        input_text = context_texts[length]
        
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        outputs = model.generate(inputs.input_ids, max_length=length+50, do_sample=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\nGenerated Text for Long-Context Input:")
        print(generated_text)
    
    return perplexities

def compare_variants(variant_models, tokenizer, texts, prompts, long_context_prompt, filler, output_dir):
    """Compare different model variants across all evaluation metrics.
    
    Args:
        variant_models: Dictionary mapping variant names to models
        tokenizer: Tokenizer for the models
        texts: List of training texts for general evaluation
        prompts: List of prompts for toxicity evaluation
        long_context_prompt: Prompt for long-context evaluation
        filler: Filler text for long-context evaluation
        output_dir: Directory to save results
        
    Returns:
        results: Dictionary of evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    toxicity_results = {}
    for variant, model in variant_models.items():
        print(f"\nEvaluating toxicity for variant: {variant}")
        variant_dir = os.path.join(output_dir, variant)
        os.makedirs(variant_dir, exist_ok=True)
        
        toxicity_df = evaluate_toxicity(model, tokenizer, prompts, variant_dir)
        toxicity_results[variant] = toxicity_df["Toxicity"].mean()
    
    plt.figure(figsize=(10, 6))
    plt.bar(toxicity_results.keys(), toxicity_results.values())
    plt.xlabel("Model Variant")
    plt.ylabel("Average Toxicity Score")
    plt.title("Toxicity Comparison Across Model Variants")
    plt.tight_layout()
    filename = os.path.join(output_dir, "toxicity_comparison.pdf")
    plt.savefig(filename, format='pdf', dpi=300)
    print(f"Toxicity comparison plot saved as {filename}")
    plt.close()
    
    context_results = {}
    for variant, model in variant_models.items():
        print(f"\nEvaluating long-context performance for variant: {variant}")
        variant_dir = os.path.join(output_dir, variant)
        
        perplexities = evaluate_long_context(
            model, tokenizer, long_context_prompt, filler, variant_dir
        )
        context_results[variant] = perplexities
    
    plt.figure(figsize=(12, 8))
    for variant, perplexities in context_results.items():
        plt.plot(list(perplexities.keys()), list(perplexities.values()), 'o-', label=variant)
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Perplexity")
    plt.title("Perplexity vs. Context Length Across Model Variants")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(output_dir, "context_length_comparison.pdf")
    plt.savefig(filename, format='pdf', dpi=300)
    print(f"Context length comparison plot saved as {filename}")
    plt.close()
    
    results = {
        "toxicity": toxicity_results,
        "long_context": context_results
    }
    
    return results
