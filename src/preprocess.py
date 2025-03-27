"""
DFAD Experiment - Data Preprocessing Module

This module handles data preprocessing for the DFAD experimental framework.
In the current implementation, it provides utility functions for generating
spatial masks and handling image transformations.
"""

import os
import numpy as np
import cv2
from PIL import Image

def generate_dummy_spatial_mask(size=(256, 256)):
    """
    Generate a dummy spatial mask for testing purposes.
    
    Args:
        size (tuple): The width and height of the mask
        
    Returns:
        np.ndarray: A binary mask of the specified size
    """
    mask = np.zeros(size, dtype=np.uint8)
    center_x, center_y = size[1] // 2, size[0] // 2
    cv2.circle(mask, (center_x, center_y), radius=50, color=1, thickness=-1)
    return mask

def generate_attribute_mask(image, attribute_type="harmful"):
    """
    Generate a mask highlighting regions with specified attributes.
    This is a dummy implementation that would be replaced with actual
    attribute detection in a real implementation.
    
    Args:
        image (PIL.Image): Input image
        attribute_type (str): Type of attribute to detect
        
    Returns:
        np.ndarray: A binary mask highlighting attribute regions
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
    
    h, w = mask.shape
    for _ in range(np.random.randint(1, 4)):
        x = np.random.randint(0, w - 50)
        y = np.random.randint(0, h - 50)
        size = np.random.randint(20, 50)
        cv2.rectangle(mask, (x, y), (x + size, y + size), 1, -1)
        
    return mask
