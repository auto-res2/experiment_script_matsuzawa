"""
Configuration parameters for RARD experiments.
"""

RANDOM_SEED = 42
VERBOSE = True

EXPERIMENT1_CONFIG = {
    'x0': 1.0,
    't_final': 3.0,
    'initial_dt': 0.01,
    'momentum': 0.9,
}

EXPERIMENT2_CONFIG = {
    'x0': 1.0,
    't_final': 3.0,
}

EXPERIMENT3_CONFIG = {
    'n_samples': 500,
    'n_steps': 100,
}
