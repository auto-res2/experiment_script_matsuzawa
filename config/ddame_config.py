"""
Configuration for D-DAME experiments.
"""

CONFIG = {
    'dataset': 'CIFAR10',  # Options: CIFAR10, CIFAR100, AFHQ-DOG
    'batch_size': 64,
    'num_workers': 2,
    
    'T': 1000,  # Number of diffusion steps
    'ch': 64,   # Base channel dimension
    'ch_mult': (1, 2, 4, 8),  # Channel multipliers
    'attn': True,  # Whether to use attention
    'num_res_blocks': 2,  # Number of residual blocks
    'dropout': 0.1,  # Dropout rate
    
    'lr': 1e-4,  # Learning rate
    'max_iters': 10,  # Maximum number of iterations per epoch for testing
    'num_epochs': 1,  # Number of epochs for testing
    
    'probe_intensity': 0.1,  # Intensity of memorization probes
    'probe_ratio': 0.2,  # Ratio of data to apply probes to
}
