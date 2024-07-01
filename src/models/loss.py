import torch
from torch import nn
import torch.nn.functional as F


class BCE(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, p, y):
        p = p.clamp(min=1e-8, max=9.9e-1)
        loss = - (y * torch.log(p + self.eps) + (1-y) * torch.log(1-p)) 
        loss = loss.mean()

        return loss


class KLReg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return KLD
    

class BinaryContrastiveLoss(nn.Module):
    def __init__(self, margin=.5):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        distance = 1 - F.cosine_similarity(z1, z2, dim=1)
        loss = label * distance.pow(2) + (1-label) * torch.clamp(self.margin - distance, min=0).pow(2)

        return loss