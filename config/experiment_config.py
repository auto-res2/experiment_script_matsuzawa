"""
Configuration file for UPR Defense experiments.
"""
import torch

BATCH_SIZE = 32
TRIGGER_INTENSITIES = [0.2, 0.5, 0.8, 1.0]
MAX_STEPS = 50
LOSS_THRESHOLD = 1e-3
MIX_COEFFICIENTS = [0.1, 0.5, 0.9]
NOISE_SCHEDULES = ['constant', 'linear', 'exponential']

ITERATIONS = 10
LEARNING_RATE = 0.01

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use 'cuda' for GPU, 'cpu' for CPU

FIGURES_DIR = 'logs'
DATA_DIR = 'data'
MODELS_DIR = 'models'

DPI = 300  # High quality for academic papers
FIG_FORMAT = 'pdf'
