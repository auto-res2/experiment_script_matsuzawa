"""
Configuration parameters for the ANCD experiments.
"""

RANDOM_SEED = 42
DEVICE = "cuda"  # Use GPU by default

DATASET_NAME = "cifar10"
BATCH_SIZE = 128
TEST_BATCH_SIZE = 256

FEATURE_DIM = 64  # Dimension of feature maps

LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
TEST_MODE_EPOCHS = 1
TEST_MODE_MAX_ITER = 2

MAX_BATCH_SIZE = 128
