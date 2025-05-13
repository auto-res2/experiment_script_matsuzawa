"""
Configuration for IBGT experiments.
"""

MODEL_CONFIG = {
    "model_height": 4,
    "layer_multiplier": 1,
    "upto_hop": 32,
    "embed_3d_type": "gaussian",
    "num_3d_kernels": 128,
    "num_dist_bins": 128,
    "node_width": 128,
    "edge_width": 128,
}

TRAIN_CONFIG = {
    "num_epochs": 1,  # Set to 1 for quick test run
    "batch_size": 8,
    "learning_rate": 1e-3,
    "test_size": 0.2,
    "random_seed": 42,
}

EXPERIMENT_CONFIG = {
    "num_samples": 50,  # Small sample size for test run
    "ib_threshold": 0.5,
    "fixed_threshold": 0.5,
    "anchors": [0.0, 0.3, 0.7, 1.0],  # Quantization anchor values
    "beta": 0.01,  # IB trade-off parameter
}

STATUS_CONFIG = {
    "status_enum": "stopped",  # Will be set to "stopped" after completion
}
