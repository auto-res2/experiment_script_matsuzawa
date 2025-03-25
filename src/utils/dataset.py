import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_fake_dataset(num_samples=100, image_size=256):
    """
    Create a fake dataset for testing.
    In a real implementation, use a real dataset like CelebA-HQ or other.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    # FakeData produces random images along with labels
    dataset = torchvision.datasets.FakeData(
        size=num_samples, 
        image_size=(3, image_size, image_size), 
        transform=transform
    )
    return dataset

def get_dataloader(dataset, batch_size=16, shuffle=True):
    """
    Create a DataLoader from a dataset.
    """
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=2
    )
