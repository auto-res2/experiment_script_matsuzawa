"""
Training script for IBGT experiments.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from utils.tgt_models import TGT_Distance, IBGT_Distance, IBGT_Distance_NoFilter, IBGT_Distance_FixedThreshold
from utils.experiment_utils import profile_epoch, set_random_seed
from config.ibgt_config import MODEL_CONFIG, TRAIN_CONFIG, EXPERIMENT_CONFIG


def train_baseline(train_loader, device):
    """Train the baseline TGT model."""
    print("\n=== Training Baseline TGT Model ===")
    baseline_model = TGT_Distance(
        model_height=MODEL_CONFIG["model_height"],
        layer_multiplier=MODEL_CONFIG["layer_multiplier"],
        upto_hop=MODEL_CONFIG["upto_hop"],
        embed_3d_type=MODEL_CONFIG["embed_3d_type"],
        num_3d_kernels=MODEL_CONFIG["num_3d_kernels"],
        num_dist_bins=MODEL_CONFIG["num_dist_bins"],
        node_width=MODEL_CONFIG["node_width"],
        edge_width=MODEL_CONFIG["edge_width"],
    )
    baseline_model.to(device)
    
    optimizer = optim.Adam(baseline_model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    loss_fn = nn.MSELoss()
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}:")
        epoch_loss, duration, mem_usage = profile_epoch(baseline_model, train_loader, optimizer, loss_fn, device)
    
    return baseline_model


def train_ibgt(train_loader, device):
    """Train the IBGT model."""
    print("\n=== Training IBGT Model ===")
    ibgt_model = IBGT_Distance(
        model_height=MODEL_CONFIG["model_height"],
        layer_multiplier=MODEL_CONFIG["layer_multiplier"],
        upto_hop=MODEL_CONFIG["upto_hop"],
        embed_3d_type=MODEL_CONFIG["embed_3d_type"],
        num_3d_kernels=MODEL_CONFIG["num_3d_kernels"],
        num_dist_bins=MODEL_CONFIG["num_dist_bins"],
        node_width=MODEL_CONFIG["node_width"],
        edge_width=MODEL_CONFIG["edge_width"],
    )
    ibgt_model.to(device)
    
    optimizer = optim.Adam(ibgt_model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    loss_fn = nn.MSELoss()
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}:")
        epoch_loss, duration, mem_usage = profile_epoch(ibgt_model, train_loader, optimizer, loss_fn, device)
    
    return ibgt_model


def train_model_variants(train_loader, device):
    """Train different model variants for the ablation study."""
    print("\n=== Training Model Variants for Ablation Study ===")
    
    model_full = IBGT_Distance(
        model_height=MODEL_CONFIG["model_height"],
        layer_multiplier=MODEL_CONFIG["layer_multiplier"],
        upto_hop=MODEL_CONFIG["upto_hop"],
        embed_3d_type=MODEL_CONFIG["embed_3d_type"],
        num_3d_kernels=MODEL_CONFIG["num_3d_kernels"],
        num_dist_bins=MODEL_CONFIG["num_dist_bins"],
        node_width=MODEL_CONFIG["node_width"],
        edge_width=MODEL_CONFIG["edge_width"],
    )
    
    model_no_filter = IBGT_Distance_NoFilter(
        model_height=MODEL_CONFIG["model_height"],
        layer_multiplier=MODEL_CONFIG["layer_multiplier"],
        upto_hop=MODEL_CONFIG["upto_hop"],
        embed_3d_type=MODEL_CONFIG["embed_3d_type"],
        num_3d_kernels=MODEL_CONFIG["num_3d_kernels"],
        num_dist_bins=MODEL_CONFIG["num_dist_bins"],
        node_width=MODEL_CONFIG["node_width"],
        edge_width=MODEL_CONFIG["edge_width"],
    )
    
    model_fixed = IBGT_Distance_FixedThreshold(
        fixed_threshold=EXPERIMENT_CONFIG["fixed_threshold"],
        model_height=MODEL_CONFIG["model_height"],
        layer_multiplier=MODEL_CONFIG["layer_multiplier"],
        upto_hop=MODEL_CONFIG["upto_hop"],
        embed_3d_type=MODEL_CONFIG["embed_3d_type"],
        num_3d_kernels=MODEL_CONFIG["num_3d_kernels"],
        num_dist_bins=MODEL_CONFIG["num_dist_bins"],
        node_width=MODEL_CONFIG["node_width"],
        edge_width=MODEL_CONFIG["edge_width"],
    )
    
    variant_models = {
        "Full_IBGT": model_full,
        "No_Filter": model_no_filter,
        "Fixed_Threshold": model_fixed
    }
    
    from utils.experiment_utils import ablation_study
    ablation_results = ablation_study(variant_models, train_loader, device, lr=TRAIN_CONFIG["learning_rate"])
    
    return variant_models


def train():
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    set_random_seed(TRAIN_CONFIG["random_seed"])
    
    from preprocess import preprocess
    train_loader, test_loader = preprocess()
    
    baseline_model = train_baseline(train_loader, device)
    ibgt_model = train_ibgt(train_loader, device)
    variant_models = train_model_variants(train_loader, device)
    
    return baseline_model, ibgt_model, variant_models, test_loader


if __name__ == "__main__":
    train()
