"""
Data utilities for video super-resolution.
"""

import torch
import torch.nn.functional as F

def get_dummy_dataloader(num_sequences=2, num_frames=3, channels=3, height=64, width=64):
    """
    Generates a dummy dataloader that yields tuples (lr_sequence, hr_sequence).
    Each sequence is a list of tensors simulating video frames.
    
    Args:
        num_sequences: Number of sequences to generate
        num_frames: Number of frames per sequence
        channels: Number of channels per frame
        height: Height of high-resolution frames
        width: Width of high-resolution frames
        
    Yields:
        tuple: (lr_sequence, hr_sequence) where each sequence is a list of tensors
    """
    for _ in range(num_sequences):
        hr_sequence = [torch.rand(channels, height, width) for _ in range(num_frames)]
        lr_sequence = [F.interpolate(frame.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)
                       for frame in hr_sequence]
        yield lr_sequence, hr_sequence
