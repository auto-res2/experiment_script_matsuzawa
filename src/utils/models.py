"""
Dummy model components for IBGT implementation.

In a real implementation, these would be replaced with actual implementations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyTGTEncoder(nn.Module):
    """
    TGT_Encoder with IB-guided triplet filtering.
    This implementation computes triplet scores based on mutual information estimation.
    """
    def __init__(self, model_height, layer_multiplier, **kwargs):
        super().__init__()
        self.model_height = model_height
        self.layer_multiplier = layer_multiplier
        self.node_width = kwargs.get('node_width', 128)
        self.edge_width = kwargs.get('edge_width', 128)
        self.lin = nn.Linear(self.node_width, self.edge_width)
        self.mi_estimator = nn.Sequential(
            nn.Linear(self.edge_width, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, g):
        g.e = self.lin(g.e)
        g.triplet_scores = self.mi_estimator(g.e).squeeze(-1)
        return g


class DummyEmbedInput(nn.Module):
    """
    Dummy implementation of EmbedInput.
    """
    def __init__(self, node_width, edge_width, upto_hop, embed_3d_type, num_3d_kernels):
        super().__init__()
        self.node_width = node_width
        self.edge_width = edge_width
        self.node_embed = nn.Linear(10, node_width)
        self.edge_embed = nn.Linear(10, edge_width)

    def forward(self, inputs):
        g = Graph()
        g.x = self.node_embed(inputs.x)  # embedded node features
        g.e = self.edge_embed(inputs.edge_attr)
        g.edge_index = inputs.edge_index
        return g


class Graph:
    """
    Graph class to hold graph data and support triplet operations.
    """
    def __init__(self, x=None, e=None, edge_index=None):
        self.x = x  # Node features
        self.e = e  # Edge features
        self.edge_index = edge_index  # Edge indices
        self.triplet_scores = None  # Will store triplet importance scores
        self.triplet_mask = None  # Will store binary mask for filtered triplets
        self.anchor_values = None  # Will store quantized anchor values
