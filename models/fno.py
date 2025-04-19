# models/fno.py
import torch.nn as nn
from .spectral_conv import SpectralConv1d

class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes=8, width=16, depth=6, dropout=0.1):
        super().__init__()
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                SpectralConv1d(width, width, n_modes),
                nn.Conv1d(width, width, 1),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(depth)
        ])
        self.project = nn.Sequential(
            nn.Conv1d(width, width//2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width//2, out_channels, 1)
        )

    def forward(self, x):
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x) + x
        return self.project(x)
