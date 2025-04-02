"""
Configuration parameters for the MEAB-DG experiments.
"""

RANDOM_SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

TEXT_EMBEDDING_DIM = 768
IMAGE_EMBEDDING_DIM = 512
FUSION_DIM = 512
HIDDEN_DIM = 256
NUM_CLASSES = 10

MAX_TEXT_LENGTH = 128
IMAGE_SIZE = 224

MAX_SEGMENT_LENGTH = 128
CONTEXT_DELIMITER = "\n\n"

USE_AMP = True  # Automatic Mixed Precision

NUM_WORKERS = 4

T4_CONFIG = {
    "max_batch_size": 64,
    "gradient_accumulation_steps": 2,
    "mixed_precision": True,
    "optimize_memory_usage": True,
    "max_memory": {
        "cuda:0": "14GB"  # Reserve some memory for system
    }
}
