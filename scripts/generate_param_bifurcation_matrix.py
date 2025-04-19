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
            if fixed_param_index == 0:
                params[:] = [fixed_value, x, y]
            elif fixed_param_index == 1:
                params[:] = [x, fixed_value, y]
            else:
                params[:] = [x, y, fixed_value]
            param_tensor = torch.from_numpy(params).unsqueeze(0).unsqueeze(-1).repeat(1, 1, T).to(torch.float32).to(device)
            with torch.no_grad():
                traj = model(param_tensor).squeeze().cpu()
            heatmap[i, j] = metric_fn(traj)
    return heatmap

# Parameter ranges
alpha_vals = np.linspace(0.6, 1.8, 70, dtype=np.float32)
beta_vals  = np.linspace(0.4, 1.8, 70, dtype=np.float32)
delta_vals = np.linspace(0.01, 0.1, 70, dtype=np.float32)

# Fixed values
fixed_alphas = [1.4, 1.0, 0.6]
fixed_betas  = [1.6, 1.0, 0.4]
fixed_deltas = [0.09, 0.06, 0.03]

# Set up 3x3 grid
fig, axs = plt.subplots(3, 3, figsize=(16, 18))
param_pairs = [(0, alpha_vals, beta_vals, fixed_deltas),
               (1, alpha_vals, delta_vals, fixed_betas),
               (2, beta_vals, delta_vals, fixed_alphas)]
titles = [
    (r"$\alpha$", r"$\beta$"),
    (r"$\alpha$", r"$\delta$"),
    (r"$\beta$", r"$\delta$")
]

for col, (fixed_idx, x_vals, y_vals, fixed_list) in enumerate(param_pairs):
    for row, fixed_val in enumerate(fixed_list):
        heatmap = compute_heatmap(x_vals, y_vals, fixed_param_index=fixed_idx, fixed_value=fixed_val)
        ax = axs[row, col]
        im = ax.imshow(heatmap, origin='lower', aspect='auto',
                       extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
                       cmap='magma')
        ax.set_xlabel(titles[col][0])
        ax.set_ylabel(titles[col][1])
        ax.set_title(f"Fixed {['δ','β','α'][fixed_idx]} = {fixed_val:.2f}")
fig.suptitle("Bifurcation Grid: max $I(t)$ across parameter space slices", fontsize=14)
plt.tight_layout()
plt.show()
