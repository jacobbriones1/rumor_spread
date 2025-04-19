import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.fno import FNO1d


def plot_aggregated_spectra(
    checkpoint_path: str,
    in_channels: int,
    out_channels: int,
    n_modes: int = 16,
    width: int = 64,
    depth: int = 4,
    save_path: str = "figures/aggregated_spectrum.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = FNO1d(in_channels=in_channels, out_channels=out_channels,
                  n_modes=n_modes, width=width, depth=depth)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    plt.figure(figsize=(10, 6))
    for layer_idx, block in enumerate(model.blocks):
        spectral_layer = block[0]
        weight = spectral_layer.weight  # shape: [in_c, out_c, n_modes]
        mag = weight.abs().detach().numpy()
        mean_mag = mag.mean(axis=(0, 1))  # [n_modes]

        plt.plot(mean_mag, label=f"Layer {layer_idx}", linewidth=2)

    plt.title("Aggregated Fourier Spectrum by Layer")
    plt.xlabel("Fourier Mode Index")
    plt.ylabel("Mean |Weight| (in_c × out_c)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved aggregated spectrum plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot average Fourier spectrum learned by each FNO layer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FNO model checkpoint (.pth)")
    parser.add_argument("--in_channels", type=int, default=4, help="Input channels (params + time)")
    parser.add_argument("--out_channels", type=int, default=3, help="Output channels (e.g., SIR)")
    parser.add_argument("--n_modes", type=int, default=16, help="Number of Fourier modes")
    parser.add_argument("--width", type=int, default=64, help="Internal channel width")
    parser.add_argument("--depth", type=int, default=4, help="Number of spectral blocks")
    parser.add_argument("--save_path", type=str, default="figures/aggregated_spectrum.png", help="Output plot path")

    args = parser.parse_args()
    plot_aggregated_spectra(
        checkpoint_path=args.checkpoint,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        n_modes=args.n_modes,
        width=args.width,
        depth=args.depth,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
