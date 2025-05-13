"""
Dummy model components for IBGT implementation.

In a real implementation, these would be replaced with actual implementations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyTGTEncoder(nn.Module):
    """
    Dummy implementation of TGT_Encoder.
    """
    def __init__(self, model_height, layer_multiplier, **kwargs):
        super().__init__()
        self.model_height = model_height
        self.layer_multiplier = layer_multiplier
        self.lin = nn.Linear(kwargs.get('node_width', 128), kwargs.get('edge_width', 128))

    def forward(self, g):
        g.e = self.lin(g.e)
        g.triplet_scores = torch.sigmoid(torch.randn(g.e.size(0)))
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
    Dummy Graph class to hold graph data.
    """
    def __init__(self, x=None, e=None, edge_index=None):
        self.x = x
        self.e = e
        self.edge_index = edge_index
        self.triplet_scores = None
