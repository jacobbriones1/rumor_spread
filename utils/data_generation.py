# utils/data_generation.py
import os
import torch
import numpy as np
from tqdm import tqdm


INIT_CONDS = [0.8, 0.1, 0.2]  # Default S, I, N values

def sample_params():
    alpha = torch.empty(1).uniform_(0.1, 0.9)
    beta  = torch.empty(1).uniform_(0.1, 0.9)
    delta = torch.empty(1).uniform_(0.01, 0.9)
    return torch.cat([alpha, beta, delta], dim=0)

# === DATA GENERATION ===
def generate_dataset(model, num_samples, T, dt, inverse=False, file_name=None):
    x_list, y_list = [], []
    time_grid = torch.linspace(0, 1, int(T/dt)).unsqueeze(0)  # Shape: (1, T)

    for _ in range(num_samples):
        params = sample_params()  # Shape: (3,)
        traj = model.simulate(params, T, dt, initial_conditions=INIT_CONDS)  # Shape: (3, T)

        # Repeat parameters across time and append time grid
        param_repeated = params.unsqueeze(-1).repeat(1, time_grid.shape[1])  # Shape: (3, T)
        full_input = torch.cat([param_repeated, time_grid], dim=0)  # Shape: (4, T)

        if inverse:
            full_input = torch.cat([traj, time_grid], dim=0)  # shape [4, T]
            x_list.append(full_input)
            y_list.append(params.unsqueeze(-1).repeat(1, time_grid.shape[1]))
        else:
            x_list.append(full_input)  # Input is [params + time] (4, T)
            y_list.append(traj)        # Output is the true trajectory (3, T)

    x_tensor, y_tensor = torch.stack(x_list), torch.stack(y_list)

    if file_name is not None and isinstance(file_name, str):
        if not file_name.endswith('.pth'):
            file_name += '.pth'
        os.makedirs("../data", exist_ok=True)
        torch.save([x_tensor, y_tensor], os.path.join("../data", file_name))

    return x_tensor, y_tensor
