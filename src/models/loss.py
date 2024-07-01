import torch
from torch import nn



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
    

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1, z2, labels):
        pass