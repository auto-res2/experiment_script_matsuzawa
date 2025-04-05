
MODEL_CONFIG = {
    'max_steps': 10,
    'early_stop_thresh': 0.01,
    'fixed_noise_level': 0.1,
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 5,
    'device': 'cuda',  # Use 'cuda' for GPU, 'cpu' for CPU
}

EXPERIMENT_CONFIG = {
    'test_batch_size': 32,
    'num_workers': 2,
    'test_batches': 2,  # For test run, use a small number of batches
    'eps_list': [4/255, 8/255, 12/255],  # PGD attack epsilon values
    'noise_levels': [0.05, 0.1, 0.15],  # Synthetic noise levels for ablation study
}
