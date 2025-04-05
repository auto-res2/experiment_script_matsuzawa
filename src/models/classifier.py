import torch
import torch.nn as nn
import torchvision

def load_classifier(checkpoint_path='./models/resnet18_cifar10.pth'):
    """
    Load a pre-trained ResNet18 classifier for CIFAR-10
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18(num_classes=10)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded classifier from checkpoint.")
    except Exception as e:
        print(f"Warning: Checkpoint not found ({e}). Using randomly initialized model for demonstration.")
    model = model.to(device)
    model.eval()
    return model
