import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.fno import FNO1d


def visualize_filter_matrices(
    checkpoint_path: str,
    in_channels: int,
    out_channels: int,
    n_modes: int = 16,
    width: int = 64,
    depth: int = 4,
    save_dir: str = "figures/filter_matrices"
):
    os.makedirs(save_dir, exist_ok=True)

    model = FNO1d(in_channels=in_channels, out_channels=out_channels,
                  n_modes=n_modes, width=width, depth=depth)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    for layer_idx, block in enumerate(model.blocks):
        spectral_layer = block[0]
        weight = spectral_layer.weight  # shape: [in_c, out_c, n_modes], complex

        weight_mag = weight.abs().detach().numpy()  # [in_c, out_c, n_modes]

        fig, axs = plt.subplots(in_channels, out_channels, figsize=(3 * out_channels, 2.5 * in_channels))
        if in_channels == 1 and out_channels == 1:
            axs = np.array([[axs]])

        for i in range(in_channels):
            for j in range(out_channels):
                ax = axs[i][j]
                ax.imshow(weight_mag[i, j][None, :], aspect='auto', cmap='viridis')
                ax.set_title(f"{i}→{j}", fontsize=9)
                ax.set_yticks([])
                ax.set_xticks([])

        plt.suptitle(f"Spectral Filter Matrix — Layer {layer_idx}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/filter_matrix_layer{layer_idx}.png")
        plt.close()

    print(f"✅ Filter matrix visualizations saved to {save_dir}")


visualize_filter_matrices("checkpoints/fno_forward_heterogeneous.pth", in_channels=5, out_channels=3)

