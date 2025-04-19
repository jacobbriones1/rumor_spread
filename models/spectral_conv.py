import torch
import torch.nn as nn

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes  # ✅ FIXED: Now properly assigned

        self.scale = 1 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, n_modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        B, C, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)

        L = min(self.n_modes, x_ft.shape[-1])
        weight = self.weight.to(dtype=x_ft.dtype, device=x.device)

        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1], dtype=x_ft.dtype, device=x.device)
        out_ft[:, :, :L] = torch.einsum("bci,coi->boi", x_ft[:, :, :L], weight[:, :, :L])

        x = torch.fft.irfft(out_ft, n=T, dim=-1)
        return x