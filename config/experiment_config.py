"""
Configuration parameters for D2PTR experiments.
"""

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = "cuda"  # Use GPU for computation

DATASET = "cifar10"  # Using CIFAR-10 for experimentation
DATA_DIR = "./data"

LATENT_DIM = 128
   
DIFFUSION_STEPS = 10
STEP_SIZE = 0.1
   
EPSILON = 0.03  # Perturbation magnitude for FGSM attack
   
DIVERGENCE_THRESHOLD = 5.0
REVERSION_LR = 0.1
REVERSION_STEPS = 10
   
MAX_ADAPTIVE_STEPS = 5
STEP_SIZE_DECAY = 0.9
