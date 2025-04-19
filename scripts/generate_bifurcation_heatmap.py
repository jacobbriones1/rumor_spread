import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.fno import FNO1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FNO1d(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load("checkpoints/dong_fno_forward.pth", map_location=device))
model.eval()

T = 50
metric_fn = lambda traj: torch.max(traj[1]).item()

def compute_heatmap(x_range, y_range, fixed_param_index, fixed_value):
    heatmap = np.zeros((len(y_range), len(x_range)))
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            params = np.zeros(3, dtype=np.float32)
            if fixed_param_index == 0:  # fixing alpha
                params[:] = [fixed_value, x, y]
            elif fixed_param_index == 1:  # fixing beta
                params[:] = [x, fixed_value, y]
            else:  # fixing delta
                params[:] = [x, y, fixed_value]

            param_tensor = torch.from_numpy(params).unsqueeze(0).unsqueeze(-1).repeat(1, 1, T).to(torch.float32).to(device)
            with torch.no_grad():
                traj = model(param_tensor).squeeze().cpu()
            heatmap[i, j] = metric_fn(traj)
    return heatmap

# Parameter ranges
alpha_vals = np.linspace(1.5, 2.5, 50, dtype=np.float32)
beta_vals  = np.linspace(0.5, 1.5, 50, dtype=np.float32)
delta_vals = np.linspace(0.01, 0.2, 50, dtype=np.float32)

# Compute heatmaps
heatmaps = [
    ("alpha", "beta", r"$\delta = 0.05$", compute_heatmap(alpha_vals, beta_vals, fixed_param_index=2, fixed_value=0.05)),
    ("alpha", "delta", r"$\beta = 1.0$", compute_heatmap(alpha_vals, delta_vals, fixed_param_index=1, fixed_value=1.0)),
    ("beta",  "delta", r"$\alpha = 2.0$", compute_heatmap(beta_vals,  delta_vals, fixed_param_index=0, fixed_value=2.0)),
]

# Plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for ax, (x_label, y_label, title, heatmap) in zip(axs, heatmaps):
    extent = [eval(f"{x_label}_vals")[0], eval(f"{x_label}_vals")[-1],
              eval(f"{y_label}_vals")[0], eval(f"{y_label}_vals")[-1]]
    ax.imshow(heatmap, origin='lower', extent=extent, aspect='auto', cmap='magma')
    ax.set_xlabel(rf'${x_label}$')
    ax.set_ylabel(rf'${y_label}$')
    ax.set_title(title)

plt.suptitle("Bifurcation Heatmaps: max $I(t)$ across parameter slices", fontsize=14)
plt.tight_layout()
plt.show()
