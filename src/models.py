"""
Model architectures for the MEAB-DG experiments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel
import torchvision.models as vision_models
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DualModalEncoder(nn.Module):
    """
    Dual Modal Encoder with Dynamic Gating for Multimodal Fusion.
    Implements the core architecture of MEAB-DG.
    """
    def __init__(self, num_labels=10, text_embedding_dim=768, image_embedding_dim=512, fusion_dim=512):
        super(DualModalEncoder, self).__init__()
        self.fusion_dim = fusion_dim
        
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(text_embedding_dim, fusion_dim)
        
        self.image_encoder = vision_models.resnet18(pretrained=True)
        self.image_encoder.fc = nn.Identity()
        self.image_proj = nn.Linear(image_embedding_dim, fusion_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, 2),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Linear(fusion_dim, num_labels)
        
    def forward(self, text_inputs, image_inputs=None, use_dynamic_gate=True, return_gate=False):
        text_outputs = self.text_encoder(**text_inputs)
        text_emb = text_outputs.last_hidden_state[:, 0, :]
        text_emb = self.text_proj(text_emb)
        
        if image_inputs is not None:
            image_feat = self.image_encoder(image_inputs)
            image_emb = self.image_proj(image_feat)
        else:
            image_emb = torch.zeros_like(text_emb)
            
        if use_dynamic_gate:
            combined = torch.cat([text_emb, image_emb], dim=1)
            gates = self.gate(combined)  # shape: [batch, 2]
            text_gate = gates[:, 0].unsqueeze(1)
            image_gate = gates[:, 1].unsqueeze(1)
            
            fused_emb = text_gate * text_emb + image_gate * image_emb
            
            if return_gate:
                return self.classifier(fused_emb), (text_gate, image_gate)
        else:
            fused_emb = (text_emb + image_emb) / 2
            
        return self.classifier(fused_emb)


class DynamicContextModel(nn.Module):
    """
    Dynamic Context Splitting Model for Long-Context Tasks.
    Implements context splitting and aggregation.
    """
    def __init__(self, transformer, hidden_dim=768):
        super(DynamicContextModel, self).__init__()
        self.transformer = transformer
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8)
        self.head = nn.Linear(hidden_dim, 1)

    def split_context(self, long_text, delimiter="\n\n"):
        """Split long text into segments based on delimiter."""
        segments = [seg.strip() for seg in long_text.split(delimiter) if seg.strip()]
        return segments

    def forward(self, tokenizer, long_text, delimiter="\n\n", max_length=128):
        """
        Process long text by splitting into segments and then aggregating.
        
        Args:
            tokenizer: Tokenizer to use for text encoding
            long_text: The long text to process
            delimiter: Delimiter to use for splitting text
            max_length: Maximum token length for each segment
        """
        segments = self.split_context(long_text, delimiter)
        segment_embeddings = []
        device = next(self.parameters()).device
        
        for seg in segments:
            tokens = tokenizer(seg, return_tensors="pt", truncation=True, 
                             padding="max_length", max_length=max_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            output = self.transformer(**tokens)
            seg_emb = output.last_hidden_state[:, 0, :]
            segment_embeddings.append(seg_emb)
            
        seg_tensor = torch.stack(segment_embeddings, dim=0)
        
        agg_output, _ = self.attn(seg_tensor, seg_tensor, seg_tensor)
        
        pooled_output = agg_output.mean(dim=0)
        final_output = self.head(pooled_output)
        
        return final_output
