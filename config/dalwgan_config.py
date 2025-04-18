"""
Configuration parameters for DALWGAN experiments
"""

DEVICE = "cuda"  # Use "cpu" for CPU-only environment

SAVE_DIR = "logs"
RANDOM_SEED = 42

SYNTH_NUM_SAMPLES = 1000
SYNTH_NOISE = 0.1
LATENT_DIM = 2

MNIST_BATCH_SIZE = 32  # Reduced batch size for memory efficiency
MNIST_IMAGE_SIZE = 64
GEN_LATENT_DIM = 100
LEARNING_RATE = 0.001

ABLATION_NUM_SAMPLES = 300
ABLATION_ITERATIONS = 10
