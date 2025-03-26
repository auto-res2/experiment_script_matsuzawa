"""
Scripts for training models.
Implements functions for model training and mixup variants.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from evaluate import evaluate

def get_model(model_name='resnet18', num_classes=100, pretrained=False):
    """
    Get a model for training.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        model: PyTorch model
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def train_one_epoch(net, train_loader, criterion, optimizer, device, 
                   mixup_mode='standard', diffu_module=None, mixup_alpha=1.0):
    """
    Train the model for one epoch.
    
    Args:
        net: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        mixup_mode: Type of mixup to use ('standard', 'synermix', 'diffusynermix')
        diffu_module: DiffuSynerMix module (required if mixup_mode is 'diffusynermix')
        mixup_alpha: Alpha parameter for mixup
        
    Returns:
        train_loss: Average training loss for the epoch
    """
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if mixup_mode == 'standard':
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            index = torch.randperm(inputs.size(0)).to(device)
            mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
            y_a, y_b = targets, targets[index]
        elif mixup_mode == 'synermix':
            lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
            index = torch.randperm(inputs.size(0)).to(device)
            mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
            y_a, y_b = targets, targets[index]
        elif mixup_mode == 'diffusynermix':
            if diffu_module is None:
                raise ValueError("diffu_module must be provided for diffusynermix mode")
            mixed_x, y_a, y_b, lam = diffu_module(inputs, targets)
        else:
            mixed_x, y_a, y_b = inputs, targets, targets
            lam = 1.0
        
        optimizer.zero_grad()
        outputs = net(mixed_x)
        
        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        
        correct_a = predicted.eq(y_a).float()
        correct_b = predicted.eq(y_b).float()
        correct += (lam * correct_a + (1 - lam) * correct_b).sum().item()
        
        pbar.set_postfix({
            'loss': train_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return train_loss / len(train_loader)

def train_model(net, train_loader, test_loader, criterion, optimizer, scheduler=None,
               num_epochs=20, device='cuda', mixup_mode='standard', diffu_module=None,
               mixup_alpha=1.0, save_path='./models'):
    """
    Train the model for multiple epochs.
    
    Args:
        net: PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to run training on
        mixup_mode: Type of mixup to use
        diffu_module: DiffuSynerMix module
        mixup_alpha: Alpha parameter for mixup
        save_path: Path to save model checkpoints
        
    Returns:
        net: Trained model
        history: Dictionary with training history
    """
    os.makedirs(save_path, exist_ok=True)
    
    history = {
        'train_loss': [],
        'test_acc': []
    }
    
    best_acc = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch+1}/{num_epochs}")
        
        train_loss = train_one_epoch(
            net, train_loader, criterion, optimizer, device,
            mixup_mode, diffu_module, mixup_alpha
        )
        
        test_acc = evaluate(net, test_loader, device)
        
        if scheduler is not None:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), os.path.join(save_path, f'model_{mixup_mode}_best.pth'))
            print(f"Saved best model with accuracy: {best_acc:.2f}%")
    
    torch.save(net.state_dict(), os.path.join(save_path, f'model_{mixup_mode}_final.pth'))
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Best test accuracy: {best_acc:.2f}%")
    
    return net, history

if __name__ == "__main__":
    from src.preprocess import get_dataloaders
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, test_loader = get_dataloaders(batch_size=16, num_workers=0)
    
    net = get_model('resnet18', num_classes=100).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    train_one_epoch(net, train_loader, criterion, optimizer, device)
    
    test_acc = evaluate(net, test_loader, device)
    print(f"Test accuracy after one epoch: {test_acc:.2f}%")
