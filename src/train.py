import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time

class TokenAcceleration(nn.Module):
    def __init__(self, input_dim, momentum=0.9):
        super(TokenAcceleration, self).__init__()
        self.momentum = momentum
        self.linear = nn.Linear(input_dim, input_dim)
    
    def forward(self, token_features, previous_features):
        transformed = self.linear(token_features)
        update = transformed + self.momentum * previous_features
        return update

class TransformerBlock(nn.Module):
    def __init__(self, input_dim):
        super(TransformerBlock, self).__init__()
        self.token_acceleration = TokenAcceleration(input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, token_features, previous_features):
        accelerated_tokens = self.token_acceleration(token_features, previous_features)
        attended_tokens, _ = self.attention(accelerated_tokens, accelerated_tokens, accelerated_tokens)
        out = self.mlp(attended_tokens)
        return out

class DummyBackbone(nn.Module):
    def __init__(self, in_channels=3, feature_dim=32):
        super(DummyBackbone, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        out = self.conv(x)  # shape: (batch, feature_dim, H, W)
        batch_size, feature_dim, H, W = out.shape
        tokens = out.view(batch_size, feature_dim, -1).permute(2, 0, 1)
        return tokens

class AccelerationSampler(nn.Module):
    def __init__(self, feature_dim, momentum=0.9):
        super(AccelerationSampler, self).__init__()
        self.token_acceleration = TokenAcceleration(feature_dim, momentum)
    
    def forward(self, token_features, previous_features):
        return self.token_acceleration(token_features, previous_features)

class AFiTModel(nn.Module):
    def __init__(self, backbone, sampler):
        super(AFiTModel, self).__init__()
        self.backbone = backbone
        self.sampler = sampler

    def forward(self, x, previous_features):
        token_features = self.backbone(x)
        tokens_updated = self.sampler(token_features, previous_features)
        return tokens_updated

def compute_composite_loss(predicted_tokens, ground_truth_tokens, diffusion_loss_weight=1.0, acceleration_loss_weight=1.0):
    loss_diffusion = ((predicted_tokens - ground_truth_tokens) ** 2).mean()
    loss_acceleration = nn.functional.l1_loss(predicted_tokens, ground_truth_tokens)
    loss_total = diffusion_loss_weight * loss_diffusion + acceleration_loss_weight * loss_acceleration
    return loss_total

def train_model(config, data):
    """
    Train the AFiT model.
    
    Args:
        config: Configuration parameters
        data: Preprocessed data for training
        
    Returns:
        trained_model: The trained model
        training_stats: Statistics from training (loss history, etc.)
    """
    images = data['images']
    ground_truth_tokens = data['ground_truth_tokens']
    
    os.makedirs('logs', exist_ok=True)
    
    backbone = DummyBackbone(config.in_channels, config.feature_dim)
    sampler = AccelerationSampler(config.feature_dim, momentum=config.momentum)
    model = AFiTModel(backbone, sampler)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    loss_history = []
    previous_features = torch.zeros(images.shape[2] * images.shape[3], images.shape[0], config.feature_dim)
    
    print(f"Starting training for {config.num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        predicted_tokens = model(images, previous_features)
        loss = compute_composite_loss(
            predicted_tokens, 
            ground_truth_tokens, 
            diffusion_loss_weight=1.0, 
            acceleration_loss_weight=config.acceleration_loss_weight
        )
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Composite Loss: {loss_val:.4f}")
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, config.num_epochs+1), loss_history, marker='o', linestyle='-', color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss")
    plt.title("AFiT Joint Training Loss Curve")
    plt.grid(True)
    plt.savefig("logs/training_loss_joint_full.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        'model': model,
        'loss_history': loss_history,
        'training_time': end_time - start_time
    }
