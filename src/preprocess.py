"""
Preprocessing script for IBGT experiments.
"""
import os
import torch
from torch_geometric.data import DataLoader
from utils.experiment_utils import create_dummy_qm9_dataset, set_random_seed
from config.ibgt_config import TRAIN_CONFIG, EXPERIMENT_CONFIG


def create_dataset():
    """Create a dummy dataset for the experiments."""
    print("Creating dummy QM9 dataset...")
    set_random_seed(TRAIN_CONFIG["random_seed"])
    
    dataset = create_dummy_qm9_dataset(num_samples=EXPERIMENT_CONFIG["num_samples"])
    
    test_size = TRAIN_CONFIG["test_size"]
    test_samples = int(len(dataset) * test_size)
    train_dataset = dataset[:-test_samples]
    test_dataset = dataset[-test_samples:]
    
    print(f"Created dataset with {len(train_dataset)} training and {len(test_dataset)} testing samples.")
    return train_dataset, test_dataset


def create_dataloaders(train_dataset, test_dataset):
    """Create dataloaders for the train and test datasets."""
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAIN_CONFIG["batch_size"], 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False
    )
    return train_loader, test_loader


def preprocess():
    """Main preprocessing function."""
    os.makedirs("logs", exist_ok=True)
    train_dataset, test_dataset = create_dataset()
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset)
    return train_loader, test_loader


if __name__ == "__main__":
    preprocess()
