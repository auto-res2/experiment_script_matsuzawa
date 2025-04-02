import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_image(image_path):
    """
    Loads and preprocesses an image.
    For demonstration, if image_path is not found, create a dummy image.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image from {image_path}")
    except Exception as e:
        print(f"Could not load image from {image_path}. Creating a dummy image. Error: {e}")
        image = Image.new('RGB', (256, 256), color='blue')
        d = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
            d.text((10, 10), "Dummy Image", fill=(255, 255, 0), font=font)
        except:
            d.text((10, 10), "Dummy Image", fill=(255, 255, 0))
    return img_transforms(image).unsqueeze(0)  # Add batch dimension

def generate_long_text(target_tokens):
    """
    Generates a long text by repeating a base sentence.
    """
    base_sentence = "This is a sample sentence for long-context testing. "
    tokens_per_sentence = len(base_sentence.split())
    repeats = (target_tokens // tokens_per_sentence) + 1
    repeated_text = base_sentence * repeats
    tokens = repeated_text.split()[:target_tokens]
    return " ".join(tokens)

def create_dummy_inputs():
    """Create dummy input tensors for model testing."""
    return torch.rand(1, 768)
