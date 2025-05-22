#!/usr/bin/env python3
"""
Configuration parameters for the Adaptive Characteristic Simulation (ACS) experiments.
"""

SAVE_DIR = "./logs"
STATUS_ENUM = "stopped"  # Will be set to "stopped" at the end of the experiment

EXP1_PARAMS = {
    "y0": 0.5,
    "t0": 0.0,
    "tend": 2.0,
    "dt_fixed": 0.05,
    "dt_initial": 0.05,
    "tol": 1e-3
}

EXP2_PARAMS = {
    "T": 2.0,
    "dt": 0.01,
    "noise_scale": 0.3,
    "tol": 0.1
}

EXP3_PARAMS = {
    "num_steps": 50,
    "eta": 0.0,
    "tol": 0.1,
    "image_size": 32,
    "channels": 3,
    "features": 64
}
