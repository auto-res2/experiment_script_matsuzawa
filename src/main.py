"""
Main module for Graph-GaussianAssembler experiments.

This module implements and runs the three experiments:
1. End-to-End Comparison on Standard Text-to-3D Benchmarks
2. Ablation Study on the Graph-Based Reassembly Module
3. Evaluation of Adaptive Time Scheduling in Graph Denoising
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from preprocess import create_datasets, create_dataloaders, get_device, ensure_directories_exist
from train import (
    GaussianDreamer, DiffAssemble, GraphGaussianAssembler, 
    GraphDenoisingModule, IndependentDenoisingModule, AdaptiveTimeScheduler,
    save_model, load_model
)
from evaluate import (
    convert_gaussians_to_pointcloud, compute_chamfer_distance, compute_redundancy_metric,
    compute_image_quality_metrics, denoising_process, save_plot,
    plot_chamfer_comparison, plot_redundancy_comparison, plot_denoising_trajectory
)

try:
    from pytorch3d.renderer import (
        FoVPerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        SoftPhongShader
    )
    from pytorch3d.structures import Meshes
    PYTORCH3D_AVAILABLE = True
except ImportError:
    print("PyTorch3D not available, some rendering functionality will be limited.")
    PYTORCH3D_AVAILABLE = False


def convert_gaussians_to_mesh(asset):
    """
    Convert Gaussian parameters to a mesh.
    
    Args:
        asset (dict): Asset containing Gaussian parameters
        
    Returns:
        Meshes: Mesh representation
    """
    if not PYTORCH3D_AVAILABLE:
        print("PyTorch3D not available, mesh conversion skipped.")
        return None
    
    pointcloud = convert_gaussians_to_pointcloud(asset)
    vertices = torch.tensor(pointcloud, dtype=torch.float32)
    n = vertices.shape[0]
    if n < 3:
        faces = torch.empty((0, 3), dtype=torch.int64)
    else:
        faces = torch.tensor([[i, (i+1)%n, (i+2)%n] for i in range(n-2)], dtype=torch.int64)
    
    mesh = Meshes(verts=[vertices], faces=[faces])
    return mesh


def setup_renderer(device):
    """
    Set up a PyTorch3D renderer.
    
    Args:
        device (torch.device): Device to use for rendering
        
    Returns:
        MeshRenderer or None: Renderer object or None if PyTorch3D is not available
    """
    if not PYTORCH3D_AVAILABLE:
        print("PyTorch3D not available, renderer setup skipped.")
        return None
    
    cameras = FoVPerspectiveCameras(device=device)
    raster_settings = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras)
    )
    
    return renderer


def experiment1(device):
    """
    Run Experiment 1: End-to-End Comparison on Standard Text-to-3D Benchmarks.
    
    Args:
        device (torch.device): Device to use for computation
    """
    print("\n=== Experiment 1: End-to-End Comparison ===")
    
    baseline_model = GaussianDreamer(config_path="config/gaussiandreamer.yaml")
    recon_variant = DiffAssemble(config_path="config/diffassemble.yaml")
    full_model = GraphGaussianAssembler()  # Default denoising
    
    prompts = ["a futuristic city", "a medieval castle", "a natural landscape"]
    results = []
    
    renderer = setup_renderer(device)
    
    for prompt in prompts:
        print(f"\n[Experiment 1] Processing prompt: '{prompt}'")
        
        asset_baseline = baseline_model.generate_3d_asset(prompt)
        asset_recon = recon_variant.generate_3d_asset(prompt)
        asset_full = full_model.generate_3d_asset(prompt)
        
        pc_baseline = convert_gaussians_to_pointcloud(asset_baseline)
        pc_recon = convert_gaussians_to_pointcloud(asset_recon)
        pc_full = convert_gaussians_to_pointcloud(asset_full)
        
        chamfer_baseline = compute_chamfer_distance(pc_baseline, pc_full)
        chamfer_recon = compute_chamfer_distance(pc_recon, pc_full)
        
        print(f"Chamfer Distance (Baseline vs Full): {chamfer_baseline:.4f}")
        print(f"Chamfer Distance (Recon vs Full):    {chamfer_recon:.4f}")
        
        ssim_val, psnr_val = None, None
        if renderer is not None:
            mesh_full = convert_gaussians_to_mesh(asset_full).to(device)
            rendered_image = renderer(mesh_full.extend(1))
            
            rendered_np = rendered_image[0, ..., :3].cpu().numpy()
            reference_np = np.ones_like(rendered_np) * 0.5
            
            ssim_val, psnr_val = compute_image_quality_metrics(rendered_np, reference_np)
            if ssim_val is not None and psnr_val is not None:
                print(f"Rendered image quality -> SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}")
        
        results.append({
            "prompt": prompt,
            "chamfer_baseline": chamfer_baseline,
            "chamfer_recon": chamfer_recon,
            "ssim": ssim_val,
            "psnr": psnr_val
        })
    
    prompts_list = [r["prompt"] for r in results]
    plot_chamfer_comparison(results, prompts_list, "experiment1_chamfer_comparison.pdf")
    
    return results


def generate_asset_variant(prompt, use_graph=True):
    """
    Generate an asset with either the full graph module or a simple module.
    
    Args:
        prompt (str): Text prompt describing the asset
        use_graph (bool): Whether to use the graph-based denoising module
        
    Returns:
        dict: Asset dictionary
    """
    if use_graph:
        denoise_module = GraphDenoisingModule(in_channels=3, out_channels=3)
        print(f"[Asset Variant] Using Graph-based denoising for prompt '{prompt}'.")
    else:
        denoise_module = IndependentDenoisingModule(in_channels=3, out_channels=3)
        print(f"[Asset Variant] Using Independent (non-graph) denoising for prompt '{prompt}'.")
    
    model = GraphGaussianAssembler(denoise_module=denoise_module)
    asset = model.generate_3d_asset(prompt)
    return asset


def experiment2():
    """
    Run Experiment 2: Ablation Study on the Graph-Based Reassembly Module.
    """
    print("\n=== Experiment 2: Ablation Study ===")
    test_prompts = ["complex mechanical part", "intricate sculpture"]
    
    graph_metrics = []
    simple_metrics = []
    
    for prompt in test_prompts:
        asset_graph = generate_asset_variant(prompt, use_graph=True)
        asset_simple = generate_asset_variant(prompt, use_graph=False)
        
        pc_graph = convert_gaussians_to_pointcloud(asset_graph)
        pc_simple = convert_gaussians_to_pointcloud(asset_simple)
        
        redundancy_graph = compute_redundancy_metric(pc_graph)
        redundancy_simple = compute_redundancy_metric(pc_simple)
        
        graph_metrics.append(redundancy_graph)
        simple_metrics.append(redundancy_simple)
        
        print(f"\nPrompt: {prompt}")
        print(f"Redundancy Metric (Graph Module): {redundancy_graph:.4f}")
        print(f"Redundancy Metric (Simple Module): {redundancy_simple:.4f}")
    
    plot_redundancy_comparison(test_prompts, graph_metrics, simple_metrics, "experiment2_redundancy_comparison.pdf")
    
    return test_prompts, graph_metrics, simple_metrics


def experiment3():
    """
    Run Experiment 3: Evaluation of Adaptive Time Scheduling in Graph Denoising.
    """
    print("\n=== Experiment 3: Adaptive Time Scheduling Evaluation ===")
    prompts = ["simple bowl", "busy marketplace", "dense forest scene"]
    
    results = []
    
    for prompt in prompts:
        print(f"\n[Experiment 3] Processing prompt: '{prompt}'")
        model = GraphGaussianAssembler()  # Use default denoising_step
        asset_initial = model.initialize_asset(prompt)
        
        asset_adaptive, iters_adaptive, hist_iters_adaptive, hist_pos_adaptive = denoising_process(
            model, asset_initial, adaptive_schedule=True
        )
        
        asset_fixed, iters_fixed, hist_iters_fixed, hist_pos_fixed = denoising_process(
            model, asset_initial, adaptive_schedule=False
        )
        
        print(f"Prompt: {prompt}")
        print(f"Adaptive Scheduler iterations: {iters_adaptive}")
        print(f"Fixed Scheduler iterations:    {iters_fixed}")
        
        pc_adaptive = convert_gaussians_to_pointcloud(asset_adaptive)
        pc_fixed = convert_gaussians_to_pointcloud(asset_fixed)
        chamfer_adaptive = compute_chamfer_distance(pc_adaptive, pc_adaptive)
        chamfer_fixed = compute_chamfer_distance(pc_fixed, pc_fixed)
        print(f"Chamfer Distance -> Adaptive: {chamfer_adaptive:.4f}, Fixed: {chamfer_fixed:.4f}")
        
        plot_denoising_trajectory(
            hist_iters_adaptive, hist_pos_adaptive,
            hist_iters_fixed, hist_pos_fixed,
            prompt, f"experiment3_trajectory_{prompt.replace(' ', '_')}.pdf"
        )
        
        results.append({
            "prompt": prompt,
            "iters_adaptive": iters_adaptive,
            "iters_fixed": iters_fixed,
            "chamfer_adaptive": chamfer_adaptive,
            "chamfer_fixed": chamfer_fixed
        })
    
    return results


def test_code():
    """
    Run a quick test to check that the code executes correctly.
    """
    print("\n=== Running quick test... ===")
    
    prompt_test = "quick test prompt"
    baseline_model = GaussianDreamer(config_path="config/gaussiandreamer.yaml")
    asset = baseline_model.generate_3d_asset(prompt_test)
    pc = convert_gaussians_to_pointcloud(asset)
    print(f"Test asset point cloud shape: {pc.shape}")
    
    test_input = torch.randn(10, 3)
    gnn = GraphDenoisingModule(in_channels=3, out_channels=3)
    indep = IndependentDenoisingModule(in_channels=3, out_channels=3)
    out_gnn = gnn(test_input)
    out_indep = indep(test_input)
    print(f"GraphDenoisingModule output shape: {out_gnn.shape}")
    print(f"IndependentDenoisingModule output shape: {out_indep.shape}")
    
    scheduler = AdaptiveTimeScheduler(init_steps=50)
    dummy_complexity = torch.tensor([0.5])
    steps = scheduler(dummy_complexity)
    print(f"AdaptiveTimeScheduler predicted steps: {steps}")
    
    print("Quick test finished.\n")


def main():
    """
    Main function to run all experiments.
    """
    start_time = time.time()
    
    ensure_directories_exist()
    
    device = get_device()
    
    print("\n=== Starting Graph-GaussianAssembler Experiments ===")
    print(f"Running on device: {device}")
    
    test_code()
    
    experiment1(device)
    experiment2()
    experiment3()
    
    end_time = time.time()
    print(f"\n=== All experiments completed ===")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
