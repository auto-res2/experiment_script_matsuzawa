"""
Scripts for preprocessing data.
Implements functions for loading and preprocessing CIFAR-100 dataset.
"""
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_transforms(augment=True):
    """
    Get transforms for training and testing.
    
    Args:
        augment: Whether to apply data augmentation for training
        
    Returns:
        train_transform: Transforms for training data
        test_transform: Transforms for test data
    """
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = test_transform
    
    return train_transform, test_transform

def get_dataloaders(batch_size=128, num_workers=4, data_dir='./data', augment=True):
    """
    Get DataLoaders for CIFAR-100 dataset.
    
    Args:
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        data_dir: Directory to store dataset
        augment: Whether to apply data augmentation for training
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    train_transform, test_transform = get_transforms(augment)
    
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_class_names():
    """
    Get class names for CIFAR-100 dataset.
    
    Returns:
        class_names: List of class names
    """
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=16, num_workers=0)
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    
    class_names = get_class_names()
    for i in range(min(5, len(labels))):
        print(f"Sample {i}: Class {labels[i]} ({class_names[labels[i]]})")
