"""Model training utilities."""
import torch
import torch.nn as nn
from tqdm import tqdm

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        layers = []
        current_size = input_size
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(hidden_size, 2))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, config):
    """Train the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
    
    return model
