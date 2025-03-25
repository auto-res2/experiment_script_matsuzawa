"""
Default configuration for the HFID experiment.
"""

config = {
    # Data parameters
    'train_samples': 100,
    'val_samples': 20,
    'image_size': 128,  # Smaller size for quicker testing
    'batch_size': 8,
    
    # Model parameters
    'use_isometry': True,
    'use_consistency': True,
    
    # Training parameters
    'learning_rate': 2e-4,
    'num_epochs': 2,  # Reduced for faster testing
    'save_freq': 2,   # Save model every 2 epochs
    
    # Evaluation parameters
    'num_gen_images': 10,
    
    # Experiment parameters
    'experiment_name': 'hfid_experiment',
    'run_experiment1': True,  # Global vs. Joint Quality and Disentanglement Comparison
    'run_experiment2': True,  # Ablation Study on Hierarchical Structure Components
    'run_experiment3': True,  # Computational Efficiency and Scalability Analysis
    
    # Variants for ablation study
    'variants': [
        {'name': 'Full_HFID', 'use_isometry': True, 'use_consistency': True},
        {'name': 'No_Isometry', 'use_isometry': False, 'use_consistency': True},
        {'name': 'No_Consistency', 'use_isometry': True, 'use_consistency': False}
    ]
}
