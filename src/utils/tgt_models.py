"""
Model architectures for TGT and IBGT implementations.
"""
import torch
import torch.nn as nn
from .models import DummyTGTEncoder, DummyEmbedInput


class TGT_Distance(nn.Module):
    """
    Baseline TGT_Distance implementation.
    """
    def __init__(self,
                 model_height,
                 layer_multiplier=1,
                 upto_hop=32,
                 embed_3d_type='gaussian',
                 num_3d_kernels=128,
                 num_dist_bins=128,
                 **layer_configs):
        super().__init__()
        self.model_height = model_height
        self.layer_multiplier = layer_multiplier
        self.upto_hop = upto_hop
        self.embed_3d_type = embed_3d_type
        self.num_3d_kernels = num_3d_kernels
        self.num_dist_bins = num_dist_bins

        self.node_width = layer_configs['node_width']
        self.edge_width = layer_configs['edge_width']
        self.layer_configs = layer_configs

        self.encoder = DummyTGTEncoder(model_height, layer_multiplier, **self.layer_configs)
        self.input_embed = DummyEmbedInput(self.node_width, self.edge_width,
                                          upto_hop, embed_3d_type, num_3d_kernels)
        self.final_ln_edge = nn.LayerNorm(self.edge_width)
        self.dist_pred = nn.Linear(self.edge_width, self.num_dist_bins)

    def forward(self, inputs):
        g = self.input_embed(inputs)
        g = self.encoder(g)

        e = g.e
        e = self.final_ln_edge(e)
        e = self.dist_pred(e)
        return e


class IBGT_Distance(TGT_Distance):
    """
    Full IBGT implementation with a learnable IB triplet filtering module.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ib_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs):
        g = self.input_embed(inputs)
        g = self.encoder(g)
        with torch.no_grad():
            gate_thresh = torch.clamp(self.ib_gate, 0, 1).item()
        mask = (g.triplet_scores > gate_thresh)
        ratio = mask.float().mean().item()
        print(f"[IBGT] Gate value: {self.ib_gate.item():.3f}; Selected triplet ratio: {ratio:.3f}")
        g.e = g.e * mask.unsqueeze(1).float()
        e = self.final_ln_edge(g.e)
        e = self.dist_pred(e)
        return e


class IBGT_Distance_NoFilter(TGT_Distance):
    """
    Variant (a): No IB Filtering (i.e., use all triplets like baseline TGT)
    """
    def forward(self, inputs):
        g = self.input_embed(inputs)
        g = self.encoder(g)
        print("[NoFilter] Using all triplets without IB filtering.")
        e = self.final_ln_edge(g.e)
        e = self.dist_pred(e)
        return e


class IBGT_Distance_FixedThreshold(TGT_Distance):
    """
    Variant (b): Fixed-Threshold Filtering (using a constant threshold)
    """
    def __init__(self, fixed_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.fixed_threshold = fixed_threshold

    def forward(self, inputs):
        g = self.input_embed(inputs)
        g = self.encoder(g)
        mask = (g.triplet_scores > self.fixed_threshold)
        ratio = mask.float().mean().item()
        print(f"[FixedThreshold] Fixed threshold: {self.fixed_threshold}; Selected ratio: {ratio:.3f}")
        g.e = g.e * mask.unsqueeze(1).float()
        e = self.final_ln_edge(g.e)
        e = self.dist_pred(e)
        return e
