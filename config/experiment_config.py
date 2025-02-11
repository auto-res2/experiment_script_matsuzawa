"""Configuration for the machine learning experiment."""
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # Data parameters
    train_size: float = 0.8
    random_seed: int = 42
    
    # Model parameters
    hidden_size: int = 64
    num_layers: int = 2
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 10
    
    # Paths
    data_dir: str = "data"
    model_dir: str = "models"
    
config = ExperimentConfig()
