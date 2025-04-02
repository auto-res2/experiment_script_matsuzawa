"""
Configuration file for AFiT experiments.
"""

batch_size = 2
image_size = 16  # Use a small image size for demonstration
in_channels = 3
feature_dim = 32

seq_length = 10  # Number of tokens in a sequence
momentum = 0.9  # Momentum for token acceleration

num_epochs = 5  # For a quick test
learning_rate = 1e-3
acceleration_loss_weight = 1.0  # Weight for secondary loss

num_iterations = 5  # For token acceleration evaluation
num_steps = 10  # For ODE solver evaluation
step_size = 0.1  # Step size for diffusion process
