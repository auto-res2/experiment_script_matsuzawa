import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2LMHeadModel

class VisionEncoder(nn.Module):
    def __init__(self):
        super(VisionEncoder, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        
    def forward(self, x):
        return self.encoder(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4)
    
    def forward(self, text_emb, vision_emb):
        fusion, _ = self.attn(query=text_emb, key=vision_emb, value=vision_emb)
        return fusion

class AdaptiveGate(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super(AdaptiveGate, self).__init__()
        self.output_dim = output_dim
        self.gate = nn.Linear(input_dim, 1)
        
        if output_dim is not None and input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = None
    
    def forward(self, modality_feat):
        weight = torch.sigmoid(self.gate(modality_feat))  # shape: [B, 1] values in (0,1)
        
        if self.projection is not None:
            modality_feat = self.projection(modality_feat)
            
        return modality_feat * weight

class MM_BTLM(nn.Module):
    def __init__(self, text_model, vision_encoder, cross_attn):
        super(MM_BTLM, self).__init__()
        self.text_model = text_model
        self.vision_encoder = vision_encoder
        self.cross_attn = cross_attn
        
        vision_dim = 512  # ResNet18 output dimension
        text_dim = text_model.config.n_embd  # GPT2 embedding dimension
        self.vision_projection = nn.Linear(vision_dim, text_dim)
        
    def forward(self, input_ids, vision_image=None):
        text_emb = self.text_model.transformer.wte(input_ids).transpose(0, 1)
        if vision_image is not None:
            vision_feats = self.vision_encoder(vision_image)  # shape: [B, feat_dim]
            vision_feats = self.vision_projection(vision_feats)  # Project to text dimension
            vision_feats = vision_feats.unsqueeze(0)  # [1, B, D]
            text_emb = self.cross_attn(text_emb, vision_feats)
        logits = self.text_model.lm_head(text_emb.transpose(0, 1))
        return logits

class MM_BTLM_Adaptive(nn.Module):
    def __init__(self, text_model, vision_encoder, cross_attn, gate):
        super(MM_BTLM_Adaptive, self).__init__()
        self.text_model = text_model
        self.vision_encoder = vision_encoder
        self.cross_attn = cross_attn
        self.gate = gate
        
        vision_dim = 512  # ResNet18 output dimension
        text_dim = text_model.config.n_embd  # GPT2 embedding dimension
        self.vision_projection = nn.Linear(vision_dim, text_dim)
        
    def forward(self, input_ids, vision_image=None):
        text_emb = self.text_model.transformer.wte(input_ids).transpose(0, 1)
        if vision_image is not None:
            vision_feats = self.vision_encoder(vision_image)  # shape: [B, feat_dim]
            gated_vision = self.gate(vision_feats)
            gated_vision = gated_vision.unsqueeze(0)  # becomes [1, B, D]
            text_emb = self.cross_attn(text_emb, gated_vision)
        logits = self.text_model.lm_head(text_emb.transpose(0, 1))
        return logits
