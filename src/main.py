import torch
import torch.nn as nn
from preprocess import load_cifar10, get_dataloaders
from train import SimpleConvNet, train_model
from evaluate import evaluate_model
from aamg import AAMG
import json
import os

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading CIFAR-10 dataset...')
    train_dataset, test_dataset = load_cifar10()
    train_loader, test_loader = get_dataloaders(train_dataset, test_dataset)
    
    # Initialize model
    model = SimpleConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizers for comparison
    optimizers = {
        'AAMG': AAMG(model.parameters(), lr=0.01, beta_factors=[0.9, 0.99]),
        'SGD': torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'Adam': torch.optim.Adam(model.parameters(), lr=0.01)
    }
    
    # Training and evaluation results
    results = {}
    
    # Test run with fewer epochs
    epochs = 2  # Small number of epochs for testing
    
    for opt_name, optimizer in optimizers.items():
        print(f'\nTraining with {opt_name}...')
        
        # Train
        train_losses = train_model(model, train_loader, optimizer, criterion, device, epochs=epochs)
        
        # Evaluate
        eval_results = evaluate_model(model, test_loader, criterion, device)
        
        results[opt_name] = {
            'train_losses': train_losses,
            'test_accuracy': eval_results['accuracy'],
            'test_loss': eval_results['loss']
        }
        
        print(f'{opt_name} Results:')
        print(f'Final Training Loss: {train_losses[-1]:.4f}')
        print(f'Test Accuracy: {eval_results["accuracy"]:.2f}%')
        print(f'Test Loss: {eval_results["loss"]:.4f}')
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/optimizer_comparison.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
