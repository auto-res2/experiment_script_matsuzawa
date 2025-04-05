import torch
import torch.nn as nn
import torch.nn.functional as F

class PurifyPlusPlus(nn.Module):
    """
    Baseline Purify++ module (simplified implementation)
    """
    def __init__(self):
        super(PurifyPlusPlus, self).__init__()
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        e1 = F.relu(self.encoder1(x))
        e2 = F.relu(self.encoder2(e1))
        d1 = F.relu(self.decoder1(e2))
        d2 = self.decoder2(d1)
        return d2

class TCDP(nn.Module):
    """
    TCDP module with double Tweedie steps and consistency loss
    """
    def __init__(self):
        super(TCDP, self).__init__()
        self.denoiser1_enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.denoiser1_enc2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser1_dec1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser1_dec2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.denoiser2_enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.denoiser2_enc2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser2_dec1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser2_dec2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        e1 = F.relu(self.denoiser1_enc1(x))
        e2 = F.relu(self.denoiser1_enc2(e1))
        d1 = F.relu(self.denoiser1_dec1(e2))
        out1 = self.denoiser1_dec2(d1)
        
        consistency = torch.mean((x - out1)**2)
        
        e1 = F.relu(self.denoiser2_enc1(out1))
        e2 = F.relu(self.denoiser2_enc2(e1))
        d1 = F.relu(self.denoiser2_dec1(e2))
        out2 = self.denoiser2_dec2(d1)
        
        return out2, consistency

class TCDP_NoConsistency(nn.Module):
    """
    Variant B: TCDP without consistency loss (single denoising)
    """
    def __init__(self):
        super(TCDP_NoConsistency, self).__init__()
        self.encoder1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        e1 = F.relu(self.encoder1(x))
        e2 = F.relu(self.encoder2(e1))
        d1 = F.relu(self.decoder1(e2))
        out = self.decoder2(d1)
        return out

class TCDP_FixedNoise(nn.Module):
    """
    Variant C: TCDP with fixed noise schedule
    """
    def __init__(self, fixed_noise_level=0.1):
        super(TCDP_FixedNoise, self).__init__()
        self.fixed_noise_level = fixed_noise_level
        self.denoiser1_enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.denoiser1_enc2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser1_dec1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser1_dec2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.denoiser2_enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.denoiser2_enc2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser2_dec1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser2_dec2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        noise = self.fixed_noise_level * torch.randn_like(x)
        x_noisy = x + noise
        
        e1 = F.relu(self.denoiser1_enc1(x_noisy))
        e2 = F.relu(self.denoiser1_enc2(e1))
        d1 = F.relu(self.denoiser1_dec1(e2))
        out1 = self.denoiser1_dec2(d1)
        
        e1 = F.relu(self.denoiser2_enc1(out1))
        e2 = F.relu(self.denoiser2_enc2(e1))
        d1 = F.relu(self.denoiser2_dec1(e2))
        out2 = self.denoiser2_dec2(d1)
        
        return out2

class TCDP_Adaptive(nn.Module):
    """
    Revised TCDP module with adaptive noise control and early stopping logging
    """
    def __init__(self, max_steps=10, early_stop_thresh=0.01):
        super(TCDP_Adaptive, self).__init__()
        self.denoiser1_enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.denoiser1_enc2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser1_dec1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser1_dec2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.denoiser2_enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.denoiser2_enc2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser2_dec1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.denoiser2_dec2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        self.max_steps = max_steps
        self.early_stop_thresh = early_stop_thresh
    
    def forward(self, x):
        step = 0
        out = x
        consistency_history = []
        
        while step < self.max_steps:
            e1 = F.relu(self.denoiser1_enc1(out))
            e2 = F.relu(self.denoiser1_enc2(e1))
            d1 = F.relu(self.denoiser1_dec1(e2))
            out1 = self.denoiser1_dec2(d1)
            
            consistency = torch.mean((out - out1)**2)
            consistency_history.append(consistency.item())
            
            if consistency < self.early_stop_thresh:
                break
                
            e1 = F.relu(self.denoiser2_enc1(out1))
            e2 = F.relu(self.denoiser2_enc2(e1))
            d1 = F.relu(self.denoiser2_dec1(e2))
            out = self.denoiser2_dec2(d1)
            
            step += 1
            
        return out, step, consistency_history
