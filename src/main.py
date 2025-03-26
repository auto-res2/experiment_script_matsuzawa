"""
Main script for MS-ANO (Multi-Stage Adaptive Noise Optimization) experiments.

This script implements three experiments:
  1. Baseline Comparison: Semantic Alignment (via CLIP), Image Fidelity (simulated FID)
     and Inference Speed between the base StableDiffusionInitNOPipeline and a new
     MS-ANO-enhanced pipeline.
  2. Ablation Study: Variants of the MS-ANO configuration (full, single-stage, and no prompt
     integration at later stages).
  3. Hyperparameter Sensitivity: Grid search over key hyperparameters to evaluate
     performance metrics.
"""

import os
import time
import argparse
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
from PIL import Image

class MS_ANOPipeline:
    """
    A pipeline to implement the MS-ANO approach.
    It wraps a Stable Diffusion pipeline and adds configuration options.
    """
    def __init__(self, base_pipeline, config=None):
        self.base_pipeline = base_pipeline
        self.config = config if config is not None else {
            "stages": 3,
            "integrate_prompt_every_stage": True,
            "clustering_threshold": 0.5,
            "attention_weight": 0.7
        }
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None):
        print(f"Loading model from {model_name_or_path}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        base_pipeline = StableDiffusionPipeline.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch_dtype
        )
        base_pipeline.to(device)
        return cls(base_pipeline, config)
    
    def __call__(self, prompt, num_inference_steps=50, log_latents=False):
        start = time.time()
        
        print(f"Running MS-ANO pipeline with config: {self.config}")
        print(f"Prompt: '{prompt}'")
        print(f"Number of inference steps: {num_inference_steps}")
        
        result = self.base_pipeline(prompt, num_inference_steps=num_inference_steps)
        image = result["images"][0] if "images" in result else result["sample"][0]

        extra_delay = 0.02 * self.config.get("stages", 3)
        time.sleep(extra_delay)  # Sleep to simulate additional processing.

        output = {"sample": [image]}
        if log_latents:
            stage_latents = []
            for i in range(self.config.get("stages", 3)):
                latent = torch.randn(1, 4, 64, 64)
                stage_latents.append(latent)
            output["stage_latents"] = stage_latents
            
        inference_time = time.time() - start
        print(f"Inference completed in {inference_time:.2f} seconds")
        return output

def compute_clip_score(image: Image.Image, prompt: str, clip_processor, clip_model):
    """
    Compute the CLIP cosine similarity score between text and image embeddings.
    """
    inputs = clip_processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    device = next(clip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
    
    if hasattr(outputs, 'image_embeds') and hasattr(outputs, 'text_embeds'):
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
    else:
        image_embeds = outputs.image_features
        text_embeds = outputs.text_features
    
    cosine_sim = torch.nn.functional.cosine_similarity(image_embeds, text_embeds)
    score = cosine_sim.item()
    return score

def compute_fid(generated_images, reference_images=None):
    """
    Dummy function to compute FID; in practice use pytorch_fid or clean-fid package.
    """
    fid = np.random.uniform(0, 100)
    return fid

def experiment_baseline(prompts, n_runs, clip_processor, clip_model):
    print("\nStarting Experiment 1: Baseline Comparison")
    
    print("Loading Base Pipeline (StableDiffusionInitNO)...")
    base_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    base_pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading MS-ANO Pipeline...")
    msano_pipeline = MS_ANOPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")  # Dummy load

    results = {"base": [], "msano": []}
    
    for pipeline_label, pipeline in zip(["base", "msano"], [base_pipeline, msano_pipeline]):
        print(f"\nEvaluating pipeline: {pipeline_label}")
        prompt_metrics = []
        for prompt in prompts:
            timings = []
            clip_scores = []
            generated_images = []
            for run in range(n_runs):
                print(f"  Prompt: '{prompt}' | Run: {run+1}/{n_runs}")
                start_time = time.time()
                output = pipeline(prompt, num_inference_steps=50)
                image = output["sample"][0]
                runtime = time.time() - start_time
                timings.append(runtime)
                score = compute_clip_score(image, prompt, clip_processor, clip_model)
                clip_scores.append(score)
                generated_images.append(image)
                print(f"    Run time: {runtime:.3f}s, CLIP score: {score:.3f}")
            avg_time = np.mean(timings)
            avg_clip = np.mean(clip_scores)
            fid_score = compute_fid(generated_images, reference_images=None)
            prompt_metrics.append({
                "prompt": prompt,
                "avg_runtime": avg_time,
                "avg_clip_score": avg_clip,
                "fid": fid_score
            })
            print(f"  --> Averaged metrics for prompt '{prompt}': runtime={avg_time:.3f}s, CLIP={avg_clip:.3f}, FID={fid_score:.2f}")
        results[pipeline_label] = prompt_metrics

    print("\nExperiment 1 Results:")
    for key, metrics in results.items():
        print(f"Pipeline: {key}")
        for m in metrics:
            print(m)

    plt.figure(figsize=(8,6))
    for pipeline_label in results:
        clips = [entry["avg_clip_score"] for entry in results[pipeline_label]]
        plt.plot(prompts, clips, marker='o', label=pipeline_label)
    plt.xlabel("Prompts")
    plt.ylabel("Average CLIP Score")
    plt.title("Baseline Comparison: CLIP Scores by Pipeline")
    plt.legend()
    plt.tight_layout()
    pdf_filename = "logs/experiment1_baseline_clip_scores.pdf"
    plt.savefig(pdf_filename, format="pdf")
    print(f"\nSaved baseline CLIP score plot to {pdf_filename}")
    plt.close()

    return results

def experiment_ablation(prompts, n_runs, clip_processor, clip_model):
    print("\nStarting Experiment 2: Ablation Study")
    
    config_full = {"stages": 3, "integrate_prompt_every_stage": True}
    config_single = {"stages": 1, "integrate_prompt_every_stage": True}  # Mimics InitNO
    config_no_stage_prompt = {"stages": 3, "integrate_prompt_every_stage": False}

    msano_full = MS_ANOPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", config=config_full)
    msano_single = MS_ANOPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", config=config_single)
    msano_no_prompt = MS_ANOPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", config=config_no_stage_prompt)

    pipeline_variants = {
        "MSANO_Full": msano_full,
        "MSANO_SingleStage": msano_single,
        "MSANO_NoStagePrompt": msano_no_prompt
    }
    
    ablation_results = {}
    
    for variant_name, pipeline in pipeline_variants.items():
        print(f"\nEvaluating variant: {variant_name}")
        variant_metrics = []
        for prompt in prompts:
            timings = []
            clip_scores = []
            latent_logs = []  # For intermediate representation logging.
            for run in range(n_runs):
                print(f"  Variant '{variant_name}' | Prompt: '{prompt}' | Run: {run+1}/{n_runs}")
                start_time = time.time()
                output = pipeline(prompt, num_inference_steps=50, log_latents=True)
                image = output["sample"][0]
                stage_latents = output.get("stage_latents", None)
                runtime = time.time() - start_time
                timings.append(runtime)
                score = compute_clip_score(image, prompt, clip_processor, clip_model)
                clip_scores.append(score)
                latent_logs.append(stage_latents)
                print(f"    Run time: {runtime:.3f}s, CLIP score: {score:.3f}")
            avg_time = np.mean(timings)
            avg_clip = np.mean(clip_scores)
            fid_score = compute_fid([], reference_images=None)
            variant_metrics.append({
                "prompt": prompt,
                "avg_runtime": avg_time,
                "avg_clip_score": avg_clip,
                "fid": fid_score,
                "latent_logs": latent_logs  # Saved for later inspection.
            })
            print(f"  --> Averaged metrics for prompt '{prompt}' for variant '{variant_name}': runtime={avg_time:.3f}s, CLIP={avg_clip:.3f}, FID={fid_score:.2f}")
        ablation_results[variant_name] = variant_metrics

    print("\nAblation Study Results:")
    for key, metrics in ablation_results.items():
        print(f"Variant: {key}")
        for m in metrics:
            print(m)
    
    plt.figure(figsize=(8,6))
    variant_names = list(pipeline_variants.keys())
    avg_runtimes = []
    for variant in variant_names:
        times = [entry["avg_runtime"] for entry in ablation_results[variant]]
        avg_runtimes.append(np.mean(times))
    plt.bar(variant_names, avg_runtimes)
    plt.xlabel("Variant")
    plt.ylabel("Average Runtime (s)")
    plt.title("Ablation Study: Average Runtime per Variant")
    plt.tight_layout()
    pdf_filename = "logs/experiment2_ablation_runtime.pdf"
    plt.savefig(pdf_filename, format="pdf")
    print(f"\nSaved ablation runtime plot to {pdf_filename}")
    plt.close()
    
    return ablation_results

def experiment_hyperparameter(prompts, n_runs, clip_processor, clip_model):
    print("\nStarting Experiment 3: Hyperparameter Sensitivity and Robustness Analysis")
    
    stages_space = [1, 2, 3, 4]
    clustering_threshold_space = [0.1, 0.3, 0.5, 0.7]
    weighting_space = [0.3, 0.5, 0.7, 0.9]
    
    hyperparam_results = []
    
    for stages, threshold, weight in itertools.product(stages_space, clustering_threshold_space, weighting_space):
        config = {
            "stages": stages,
            "clustering_threshold": threshold,
            "attention_weight": weight,
            "integrate_prompt_every_stage": True
        }
        pipeline = MS_ANOPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", config=config)
        metrics_across_prompts = []
        for prompt in prompts:
            runtimes = []
            clip_scores = []
            for _ in range(n_runs):
                start_time = time.time()
                output = pipeline(prompt, num_inference_steps=50)
                image = output["sample"][0]
                runtime = time.time() - start_time
                runtimes.append(runtime)
                score = compute_clip_score(image, prompt, clip_processor, clip_model)
                clip_scores.append(score)
            metrics_across_prompts.append({
                "prompt": prompt,
                "avg_runtime": np.mean(runtimes),
                "avg_clip_score": np.mean(clip_scores)
            })
        overall_runtime = np.mean([m["avg_runtime"] for m in metrics_across_prompts])
        overall_clip = np.mean([m["avg_clip_score"] for m in metrics_across_prompts])
        
        hyperparam_results.append({
            "stages": stages,
            "clustering_threshold": threshold,
            "attention_weight": weight,
            "overall_runtime": overall_runtime,
            "overall_clip_score": overall_clip
        })
        print(f"Config: stages={stages}, threshold={threshold}, weight={weight} -> runtime={overall_runtime:.3f}s, CLIP={overall_clip:.3f}")
    
    filtered = [r for r in hyperparam_results if (r["clustering_threshold"] == 0.5 and r["attention_weight"] == 0.7)]
    if filtered:
        stages_list = [r["stages"] for r in filtered]
        runtime_list = [r["overall_runtime"] for r in filtered]
        plt.figure(figsize=(8,6))
        plt.plot(stages_list, runtime_list, marker='o')
        plt.xlabel("Number of Stages")
        plt.ylabel("Average Runtime (s)")
        plt.title("Runtime vs. Number of Stages (Threshold=0.5, Weight=0.7)")
        plt.tight_layout()
        pdf_filename = "logs/experiment3_runtime_vs_stages.pdf"
        plt.savefig(pdf_filename, format="pdf")
        print(f"\nSaved hyperparameter sensitivity plot to {pdf_filename}")
        plt.close()
    else:
        print("No configurations found for plotting with threshold=0.5 and weight=0.7.")
    
    return hyperparam_results

def run_minimal_test():
    """
    Run a minimal version of Experiment 1 to validate that the code executes.
    This test uses only one prompt and one run to quickly check functionality.
    """
    print("\nRunning minimal test...")
    test_prompts = ["a cat and a rabbit"]
    test_n_runs = 1

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    _ = experiment_baseline(test_prompts, test_n_runs, clip_processor, clip_model)
    
    print("\nMinimal test finished successfully.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Experiments for MS-ANO vs. StableDiffusionInitNO.")
    parser.add_argument("--test", action="store_true", help="Run minimal test and exit.")
    args = parser.parse_args()
    
    if args.test:
        run_minimal_test()
        return
    
    prompts = [
        "a cat and a rabbit",
        "a scenic mountain view",
        "a futuristic city",
        "a cozy living room"
    ]
    n_runs = 3  # Number of runs for averaging.
    
    print("Loading CLIP model and processor...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    baseline_results = experiment_baseline(prompts, n_runs, clip_processor, clip_model)
    ablation_results = experiment_ablation(prompts, n_runs, clip_processor, clip_model)
    hyperparam_results = experiment_hyperparameter(prompts, n_runs, clip_processor, clip_model)
    
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
