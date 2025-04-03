import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_cifar10(batch_size=128, download=True):
    """Load the CIFAR-10 dataset.
    
    Args:
        batch_size: Batch size for data loaders
        download: Whether to download the dataset
        
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=download, 
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=download, 
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

def generate_adversarial_examples(model, data_loader, attack_type='PGD', eps=8/255, 
                                steps=10, alpha=2/255):
    """Generate adversarial examples using different attack methods.
    
    Args:
        model: Target model
        data_loader: Data loader for clean samples
        attack_type: Type of attack ('PGD' or 'FGSM')
        eps: Maximum perturbation
        steps: Number of attack steps (for PGD)
        alpha: Step size (for PGD)
        
    Returns:
        adv_examples: Adversarial examples
        clean_labels: Original labels
    """
    from advertorch.attacks import PGDAttack, FGSM
    
    device = next(model.parameters()).device
    adv_examples = []
    clean_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    if attack_type == 'PGD':
        adversary = PGDAttack(
            model, loss_fn=criterion, eps=eps,
            nb_iter=steps, eps_iter=alpha,
            rand_init=True, clip_min=0.0, clip_max=1.0
        )
    elif attack_type == 'FGSM':
        adversary = FGSM(
            model, loss_fn=criterion, eps=eps,
            clip_min=0.0, clip_max=1.0
        )
    else:
        raise ValueError(f"Attack type {attack_type} not supported.")
        
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = adversary.perturb(images, labels)
        
        adv_examples.append(adv_images.cpu())
        clean_labels.append(labels.cpu())
        
    return torch.cat(adv_examples), torch.cat(clean_labels)
