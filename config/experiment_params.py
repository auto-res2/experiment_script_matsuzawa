"""
Configuration parameters for ANGAS experiments.

This file contains parameters that can be adjusted to control the experiments.
"""

MODEL_PARAMS = {
    'in_channels': 3,
    'hidden_channels': 16
}

DATA_PARAMS = {
    'batch_size': 16,
    'image_size': 32,
    'channels': 3,
    'num_samples': 1000
}

EXPERIMENT_PARAMS = {
    'num_steps': 50,  # Default number of steps for full experiments
    'dt': 0.1,        # Time step size
    'seed': 0         # Random seed for reproducibility
}

THRESHOLD_VALUES = [0.5, 1.0, 1.5, 2.0]
