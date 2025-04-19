# === visualize_topology_inverse.py ===
# Modular visualization script for inverse FNO topology predictions

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from models.fno import FNO1d  # Assumes modular FNO is here
from dynamics.topo_model import TopoRumorModel


def visualize_predictions(checkpoint_path: str, output_path: str, samples_per_type: int = 20):
    checkpoint = torch.load(checkpoint_path)
    model = FNO1d(
        in_channels=checkpoint['config']['input_channels'],
        out_channels=checkpoint['config']['output_channels']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    graph_types = ["ER", "BA", "WS"]
    param_vec = [0.15, 0.1, 0.02, 0.008]
    model_instance = TopoRumorModel()

    preds, trues, labels = [], [], []
    for gtype in graph_types:
        model_instance.graph_type = gtype
        for _ in range(samples_per_type):
            traj, true_feats = model_instance.simulate(param_vec, T=200)
            x_input = torch.cat([
                torch.tensor(param_vec).unsqueeze(1).expand(-1, traj.shape[-1]),
                torch.linspace(0, 1, traj.shape[-1]).unsqueeze(0)
            ], dim=0).unsqueeze(0)

            with torch.no_grad():
                pred = model(x_input).squeeze(0)
            preds.append(pred.numpy())
            trues.append(true_feats.numpy())
            labels.append(gtype)

    preds = np.array(preds)
    trues = np.array(trues)
    labels = np.array(labels)

    features = ["Clustering", "Path Length", "Assortativity"]
    plt.figure(figsize=(12, 4))
    for i, feat in enumerate(features):
        plt.subplot(1, 3, i + 1)
        sns.scatterplot(x=trues[:, i], y=preds[:, i], hue=labels, palette="deep", s=60)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel(f"True {feat}")
        plt.ylabel(f"Predicted {feat}")
        plt.title(f"{feat} Prediction")
        plt.grid(True)

    plt.suptitle("Inverse FNO Topology Predictions (Fixed θ)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize inverse FNO predictions for topology features.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fno_topo_inverse.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="plots/inverse_topology_predictions_modular.png",
                        help="Path to save output image")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples per graph type")
    args = parser.parse_args()

    visualize_predictions(args.checkpoint, args.output, args.samples)
