"""
Configuration for DALWGAN experiments
"""

RANDOM_SEED = 42
DEVICE = 'cuda'  # Use 'cuda' for GPU, 'cpu' for CPU
LOG_DIR = './logs'
MODEL_DIR = './models'

EXP1_CONFIG = {
    'dataset': 'swiss_roll',
    'n_samples': 1500,
    'noise': 0.1,
    'latent_dim': 2,
    'diffusion_steps': 10,
    'integration_method': 'heun',
    'epochs': 100,
    'batch_size': 64,
    'lr': 0.0002,
    'lambda_rank': 0.01,
    'lambda_diff': 0.1,
}

EXP2_CONFIG = {
    'dataset': 'mnist',
    'img_size': 64,
    'latent_dim': 100,
    'diffusion_steps': 10,
    'integration_method': 'heun',
    'epochs': 20,
    'batch_size': 64,
    'lr': 0.0002,
    'lambda_rank': 0.01,
    'lambda_diff': 0.1,
}

EXP3_CONFIG = {
    'dataset': 'swiss_roll',
    'n_samples': 500,
    'noise': 0.1,
    'latent_dim': 2,
    'configurations': [
        {'num_steps': 5, 'method': 'euler'},
        {'num_steps': 5, 'method': 'heun'},
        {'num_steps': 15, 'method': 'euler'},
        {'num_steps': 15, 'method': 'heun'}
    ],
    'epochs': 10,
    'batch_size': 64,
}
