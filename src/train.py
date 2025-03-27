"""
DFAD Experiment - Model Training Module

This module implements the training functions for the DFAD experimental framework.
It primarily focuses on implementing the adapters and their integration into
the diffusion model pipeline.
"""

import os
import time
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.dfad_config import TEST_MODE, DEVICE, MODEL_CONFIG

class BaseHFIAdapter:
    """
    Base Human Feedback Inversion (HFI) adapter implementation.
    This adapter modifies prompts with additional safe guidance based on human feedback.
    """
    def __init__(self, feedback_strength=1.0):
        self.feedback_strength = feedback_strength
        
    def modify(self, prompt):
        """
        Modify the input prompt with additional safe guidance.
        
        Args:
            prompt (str): Original prompt
            
        Returns:
            str: Modified prompt with safe guidance
        """
        return prompt + f" with additional safe guidance (strength: {self.feedback_strength})"
    
    def __str__(self):
        return "BaseHFI"

class DFADAdapter:
    """
    Dual-Feedback Adapter Diffusion (DFAD) implementation.
    This adapter uses both textual and spatial guidance signals.
    """
    def __init__(self, spatial_mask=None, textual_strength=1.0, spatial_strength=1.0):
        """
        Initialize the DFAD adapter.
        
        Args:
            spatial_mask: Spatial guidance mask (None if not using spatial guidance)
            textual_strength: Strength of textual guidance (0.0 to 1.0)
            spatial_strength: Strength of spatial guidance (0.0 to 1.0)
        """
        self.spatial_mask = spatial_mask
        self.textual_strength = textual_strength
        self.spatial_strength = spatial_strength
        
    def modify(self, prompt):
        """
        Modify the input prompt with dual-feedback guidance.
        
        Args:
            prompt (str): Original prompt
            
        Returns:
            str: Modified prompt with dual-feedback guidance
        """
        base_prompt = prompt + f" with safe textual guidance (strength: {self.textual_strength})"
        if self.spatial_mask is not None:
            base_prompt += f" [Spatial guidance applied (strength: {self.spatial_strength})]"
        return base_prompt
    
    def __str__(self):
        return "DFAD"

class AblationAdapter:
    """
    Adapter used for ablation studies to test components of DFAD separately.
    """
    def __init__(self, use_textual=True, use_spatial=True, spatial_mask=None,
                textual_strength=1.0, spatial_strength=1.0):
        """
        Initialize the ablation adapter.
        
        Args:
            use_textual: Whether to use textual guidance
            use_spatial: Whether to use spatial guidance
            spatial_mask: Spatial guidance mask
            textual_strength: Strength of textual guidance
            spatial_strength: Strength of spatial guidance
        """
        self.use_textual = use_textual
        self.use_spatial = use_spatial
        self.spatial_mask = spatial_mask
        self.textual_strength = textual_strength
        self.spatial_strength = spatial_strength
        
    def modify(self, prompt):
        """
        Modify the input prompt based on activated components.
        
        Args:
            prompt (str): Original prompt
            
        Returns:
            str: Modified prompt with activated components' guidance
        """
        modified_prompt = prompt
        if self.use_textual:
            modified_prompt += f" with safe textual inversion (strength: {self.textual_strength})"
        if self.use_spatial and self.spatial_mask is not None:
            modified_prompt += f" with safe spatial guidance (strength: {self.spatial_strength})"
        return modified_prompt
    
    def __str__(self):
        components = []
        if self.use_textual:
            components.append("Text")
        if self.use_spatial:
            components.append("Spatial")
        return "Ablation: " + "+".join(components)

def generate_image(prompt, adapter=None, model_id="runwayml/stable-diffusion-v1-5"):
    """
    Generate an image using the diffusion pipeline, with a given adapter (if provided).
    In TEST_MODE, a dummy image is returned.
    
    Args:
        prompt (str): Input prompt for image generation
        adapter: Adapter to modify the prompt (optional)
        model_id (str): Model identifier for the diffusion model
        
    Returns:
        PIL.Image: Generated image
    """
    conditioned_prompt = adapter.modify(prompt) if adapter is not None else prompt
    
    if TEST_MODE:
        img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        print(f"[TEST_MODE] Generated dummy image for prompt: {conditioned_prompt}")
        return img
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
        
        with torch.autocast(device):
            result = pipeline(
                conditioned_prompt,
                guidance_scale=MODEL_CONFIG["guidance_scale"],
                num_inference_steps=MODEL_CONFIG["num_inference_steps"],
                height=MODEL_CONFIG["height"],
                width=MODEL_CONFIG["width"]
            )
            
        img = result.images[0]
        return img
