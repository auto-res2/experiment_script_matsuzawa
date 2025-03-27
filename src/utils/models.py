"""
Model implementations for video super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StableVSR(nn.Module):
    """
    Dummy implementation of StableVSR, which acts as a baseline.
    In reality, this would implement a temporal conditioning module etc.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
    
    def forward(self, lr_sequence):
        output_sequence = []
        for frame in lr_sequence:
            inp = frame.unsqueeze(0)
            out = F.relu(self.conv(inp))
            out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=False)
            output_sequence.append(out.squeeze(0))
        return output_sequence


class ATRD(nn.Module):
    """
    Implementation of ATRD that includes an OTAR (on-the-fly temporal attention regulation) module.
    Supports setting diffusion steps.
    """
    def __init__(self, diffusion_steps=40):
        super().__init__()
        self.diffusion_steps = diffusion_steps  # simulate number of iterative steps
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.otar_enabled = True
        self.iteration_count = diffusion_steps  # will be updated if needed
        
    def set_diffusion_steps(self, steps):
        self.diffusion_steps = steps
        self.iteration_count = steps  # For simulation: iteration_count = diffusion_steps
    
    def forward(self, lr_sequence):
        output_sequence = []
        for frame in lr_sequence:
            inp = frame.unsqueeze(0)
            if self.otar_enabled:
                out = F.relu(self.conv(inp))
                out = out + 0.01 * torch.tanh(out)  # dummy extra processing
            else:
                out = F.relu(self.conv(inp))
            out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=False)
            output_sequence.append(out.squeeze(0))
        return output_sequence


class ATRD_NoOTAR(ATRD):
    """
    Variant of ATRD where the OTAR mechanism is disabled.
    """
    def __init__(self, diffusion_steps=40):
        super().__init__(diffusion_steps=diffusion_steps)
        self.otar_enabled = False
        self.iteration_count = diffusion_steps + 5
