"""
Model definitions and training functions for G-DS3 Transformer experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GatingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(GatingModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        gate = self.net(x)
        return gate

class LightweightGatingModule(nn.Module):
    def __init__(self, input_dim):
        super(LightweightGatingModule, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class ComplexGatingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ComplexGatingModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class GDS3Layer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, use_gate=True):
        super(GDS3Layer, self).__init__()
        self.use_gate = use_gate
        self.state_proj = nn.Linear(d_model, d_state)
        self.conv_proj = nn.Conv1d(d_model, d_conv, kernel_size=3, padding=1)
        if use_gate:
            self.gate = GatingModule(d_state)
        else:
            self.gate = None
        self.out_proj = nn.Linear(d_state + d_conv, d_model)
    
    def forward(self, x):
        state = self.state_proj(x)
        conv = self.conv_proj(x.transpose(1, 2)).transpose(1, 2)  # (batch, seq_len, d_conv)
        
        if self.use_gate and self.gate is not None:
            gate_val = self.gate(state)
            updated_state = gate_val * state + (1 - gate_val) * torch.tanh(state)
        else:
            updated_state = 0.5 * state + 0.5 * torch.tanh(state)
        combined = torch.cat([updated_state, conv], dim=-1)
        out = self.out_proj(combined)
        return out

class TransformerModel(nn.Module):
    def __init__(self, d_model=32, d_state=16, d_conv=8, num_layers=2, use_gate=True):
        super(TransformerModel, self).__init__()
        self.layers = nn.ModuleList([
            GDS3Layer(d_model, d_state, d_conv, use_gate=use_gate) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)

class AblationGDS3Layer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, gating_module=None):
        super(AblationGDS3Layer, self).__init__()
        self.state_proj = nn.Linear(d_model, d_state)
        self.conv_proj = nn.Conv1d(d_model, d_conv, kernel_size=3, padding=1)
        self.gate = gating_module  # May be None
        self.out_proj = nn.Linear(d_state + d_conv, d_model)
    
    def forward(self, x):
        state = self.state_proj(x)
        conv = self.conv_proj(x.transpose(1,2)).transpose(1,2)
        if self.gate is not None:
            gate_val = self.gate(state)
            updated_state = gate_val * state + (1 - gate_val) * torch.tanh(state)
        else:
            updated_state = 0.5 * state + 0.5 * torch.tanh(state)
        combined = torch.cat([updated_state, conv], dim=-1)
        out = self.out_proj(combined)
        return out

class AblationTransformerModel(nn.Module):
    def __init__(self, d_model=32, d_state=16, d_conv=8, num_layers=2, gating_module_factory=None):
        super(AblationTransformerModel, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            gate_mod = gating_module_factory(d_state) if gating_module_factory is not None else None
            self.layers.append(AblationGDS3Layer(d_model, d_state, d_conv, gating_module=gate_mod))
        self.classifier = nn.Linear(d_model, 10)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)

def train_model(model, dataloader, num_epochs=3, lr=1e-3, device="cpu"):
    """
    Train a model on the given dataloader.
    
    Args:
        model: The model to train
        dataloader: The dataloader containing the training data
        num_epochs: Number of epochs to train for
        lr: Learning rate
        device: Device to train on (cpu or cuda)
        
    Returns:
        List of average losses per epoch
    """
    print("Training on device:", device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for seq, label in dataloader:
            embed = F.one_hot(seq, num_classes=10).float().to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = model(embed)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print("Epoch", epoch, "Loss:", avg_loss)
        
    return losses
