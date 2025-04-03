import torch
import numpy as np

def create_multimodal_data(n_points=1000):
    """
    Create a synthetic multimodal 2D dataset from four Gaussian clusters.
    """
    centers = [(-2, -2), (2, 2), (-2, 2), (2, -2)]
    data = []
    for cx, cy in centers:
        pts = torch.randn(n_points // len(centers), 2) + torch.tensor([cx, cy])
        data.append(pts)
    full_data = torch.cat(data, dim=0)
    print("Created multimodal dataset with shape:", full_data.shape)
    return full_data

def preprocess_data(n_points=1000):
    """
    Main preprocessing function that generates the synthetic data
    for the HNCG experiment.
    """
    print("Preprocessing data...")
    data = create_multimodal_data(n_points=n_points)
    return data
