#!/usr/bin/env python
"""
Preprocessing module for the HEDA experiments.
This module prepares data structures for experiments.
"""
import torch
import json
import os
from utils import initialize_states

def load_config(config_path='config/heda_config.json'):
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def prepare_experiment_data(experiment_type='convergence', config=None):
    """
    Prepare data for different experiment types.
    
    Args:
        experiment_type: Type of experiment ('convergence', 'resource', 'hyperparameter', 'test')
        config: Configuration dictionary (loaded from config file)
    
    Returns:
        Dictionary containing experiment data and parameters
    """
    if config is None:
        config = load_config()
    
    exp_config = config['experiments'][experiment_type]
    
    if experiment_type == 'convergence':
        shape = exp_config.get('shape', [100, 2])
        state_init = initialize_states(shape)
        return {
            'state_init': state_init,
            'max_iter': exp_config.get('max_iter', 100),
            'tol': exp_config.get('tol', 1e-3),
            'step_size': exp_config.get('step_size', 0.1),
            'memory_size': exp_config.get('memory_size', 3)
        }
    elif experiment_type == 'resource':
        shape = exp_config.get('shape', [1, 4, 64, 64])
        state_init = initialize_states(shape)
        return {
            'state_init': state_init,
            'max_iter': exp_config.get('max_iter', 50),
            'step_size': exp_config.get('step_size', 0.1),
            'memory_size': exp_config.get('memory_size', 3)
        }
    elif experiment_type == 'hyperparameter':
        shape = exp_config.get('shape', [100, 2])
        state_init = initialize_states(shape)
        return {
            'state_init': state_init,
            'step_sizes': exp_config.get('step_sizes', [0.05, 0.1, 0.2]),
            'memory_sizes': exp_config.get('memory_sizes', [1, 3, 5]),
            'max_iter': exp_config.get('max_iter', 100),
            'tol': exp_config.get('tol', 1e-3)
        }
    elif experiment_type == 'test':
        # Small test data for quick verification
        shape_conv = (10, 2)
        state_init_conv = initialize_states(shape_conv)
        
        shape_time = (1, 4, 32, 32)  # smaller resolution for faster test
        state_init_time = initialize_states(shape_time)
        
        return {
            'state_init_conv': state_init_conv,
            'state_init_time': state_init_time,
            'max_iter': exp_config.get('max_iter', 10),
            'tol': exp_config.get('tol', 1e-2),
            'step_size': 0.1,
            'memory_size': 1
        }
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
