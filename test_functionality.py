"""
Simple test script to verify the functionality of D2PTR implementation.
"""
import os
import torch
import matplotlib.pyplot as plt
from src.utils.models import SimpleCNN, LatentEncoder, DiffusionPurifier
from src.utils.diffusion_utils import set_seed, fgsm_attack
from config.experiment_config import RANDOM_SEED, LATENT_DIM, DIFFUSION_STEPS, STEP_SIZE

os.makedirs("logs", exist_ok=True)

device = "cpu"
print(f"Using device: {device}")

set_seed(RANDOM_SEED)
print(f"Random seed set to: {RANDOM_SEED}")

print("\nTesting model instantiation...")
classifier = SimpleCNN(num_classes=10).to(device)
encoder = LatentEncoder(latent_dim=LATENT_DIM).to(device)
purifier = DiffusionPurifier(num_steps=DIFFUSION_STEPS, step_size=STEP_SIZE).to(device)
print("All models instantiated successfully!")

print("\nTesting forward pass with dummy data...")
dummy_input = torch.randn(2, 3, 32, 32).to(device)
classifier_output = classifier(dummy_input)
encoder_output = encoder(dummy_input)
purifier_output = purifier(dummy_input)

print(f"Classifier output shape: {classifier_output.shape}")
print(f"Encoder output shape: {encoder_output.shape}")
print(f"Purifier output shape: {purifier_output.shape}")

print("\nTesting plot generation...")
plt.figure(figsize=(6, 4))
plt.plot([1, 2, 3, 4], [10, 20, 15, 25], marker='o')
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Test Plot")
plt.grid(True)
plt.savefig("logs/test_plot.pdf", format="pdf", dpi=300)
plt.close()
print("Test plot saved as 'logs/test_plot.pdf'")

print("\nAll functionality tests passed!")
