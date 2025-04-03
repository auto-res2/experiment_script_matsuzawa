import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyLoss(nn.Module):
    """Consistency loss for enforcing consistency across noise levels."""
    def __init__(self):
        super(ConsistencyLoss, self).__init__()
        
    def forward(self, outputs_t1, outputs_t2, weight=1.0):
        """Compute consistency loss between outputs at different noise levels.
        
        Args:
            outputs_t1: Output at noise level t1
            outputs_t2: Output at noise level t2
            weight: Weight for the loss
            
        Returns:
            Consistency loss value
        """
        return weight * F.mse_loss(outputs_t1, outputs_t2)

class AdaptiveConsistency(nn.Module):
    """Adaptive consistency module for CAP method."""
    def __init__(self, input_dim=3):
        super(AdaptiveConsistency, self).__init__()
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """Predict adaptive consistency weight.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted weight
        """
        return self.weight_predictor(x)
