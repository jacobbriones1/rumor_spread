# models/fno.py
import torch
import torch.nn as nn
from .spectral_conv import SpectralConv1d

class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes=16, width=64, depth=4, dropout=0.1):
        """
        in_channels: number of input channels (e.g. params + time)
        out_channels: number of output channels (e.g. state dim)
        n_modes: number of Fourier modes to keep
        width: internal channel width
        depth: number of spectral blocks
        dropout: dropout probability
        """
        super().__init__()
        self.width = width
        self.lift = nn.Conv1d(in_channels, width, 1)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                SpectralConv1d(width, width, n_modes),
                nn.Conv1d(width, width, 1),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(depth)
        ])

        self.project = nn.Sequential(
            nn.Conv1d(width, width // 2, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width // 2, out_channels, 1)
        )

    def forward(self, x):
        """
        x: tensor of shape [B, in_channels, T]
        returns: tensor of shape [B, out_channels, T]
        """
        x = self.lift(x)  # [B, width, T]
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        return self.project(x)  # [B, out_channels, T]
