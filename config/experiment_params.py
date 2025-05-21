#!/usr/bin/env python3
"""
Configuration parameters for the Iso-LWGAN experiments.
"""

RANDOM_SEED = 42
DEVICE = "cuda"  # Use "cpu" if no GPU is available
STATUS_ENUM = "running"  # Will be set to "stopped" after experiments complete

EXP1_PARAMS = {
    "lambda_iso": 1.0,           # Weight for isometric loss
    "num_epochs": 20,            # Number of training epochs
    "batch_size": 128,           # Batch size for training
    "z_dim": 3,                  # Latent space dimension
    "n_samples": 2000,           # Number of synthetic data samples
    "learning_rate": 1e-3,       # Learning rate for optimizer
    "test_lambda_iso": [0.0, 0.5, 1.0, 2.0]  # Lambda values to test
}

EXP2_PARAMS = {
    "sigma_noise": 0.1,          # Noise magnitude for stochastic generator
    "num_epochs": 20,            # Number of training epochs
    "batch_size": 128,           # Batch size for training
    "z_dim": 3,                  # Latent space dimension
    "n_samples": 1500,           # Number of synthetic data samples
    "learning_rate": 1e-3,       # Learning rate for optimizer
    "test_sigma_noise": [0.0, 0.1, 0.5]  # Sigma values to test
}

EXP3_PARAMS = {
    "lambda_iso": 1.0,           # Weight for isometric loss
    "sigma_noise": 0.1,          # Noise magnitude for stochastic generator
    "num_epochs": 5,             # Number of training epochs
    "batch_size": 128,           # Batch size for training
    "z_dim": 20,                 # Latent space dimension
    "learning_rate": 1e-3        # Learning rate for optimizer
}

TEST_PARAMS = {
    "num_epochs": 1,             # Small number of epochs for testing
    "batch_size": 32,            # Small batch size for testing
    "z_dim": 2,                  # Reduced dimension for testing
    "lambda_iso": 0.5,           # Weight for isometric loss in testing
    "sigma_noise": 0.1,          # Noise magnitude for testing
    "test_lambda_iso": [0.0, 1.0],  # Reduced set of lambda values for testing
    "test_sigma_noise": [0.0, 0.1],  # Reduced set of sigma values for testing
    "n_samples": 500             # Reduced number of samples for testing
}
