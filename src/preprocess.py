import torch
from torchvision import datasets, transforms

def load_cifar10(data_dir='./data'):
    """Load CIFAR-10 dataset for testing the optimizer."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                  download=True, transform=transform)
    
    return train_dataset, test_dataset

def get_dataloaders(train_dataset, test_dataset, batch_size=128):
    """Create data loaders for training and testing."""
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader
