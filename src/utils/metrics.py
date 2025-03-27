"""
Utility functions for computing metrics for video super-resolution.
"""

import torch
import numpy as np

def temporal_consistency(sequence):
    """
    Computes a simple temporal consistency metric: the average inter-frame absolute difference.
    
    Args:
        sequence: List of tensors representing video frames
        
    Returns:
        float: Average temporal consistency score (lower is better)
    """
    diffs = []
    for i in range(1, len(sequence)):
        diffs.append(torch.mean(torch.abs(sequence[i] - sequence[i-1])).item())
    return np.mean(diffs) if diffs else 0.0
