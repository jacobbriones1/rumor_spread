# utils/data_generation.py
import torch
import numpy as np
from tqdm import tqdm

def sample_params():
    alpha = torch.empty(1).uniform_(0.1, 0.9)
    beta  = torch.empty(1).uniform_(0.1, 0.9)
    delta = torch.empty(1).uniform_(0.01, 0.9)
    return torch.cat([alpha, beta, delta], dim=0)

def generate_dataset(model, num_samples, T=50, dt=0.05, inverse=False):
    """
    Returns (x, y) where:
    - Forward: x = params over time, y = trajectory
    - Inverse: x = trajectory, y = params over time
    """
    x_list, y_list = [], []
    max_tries = num_samples * 10
    collected = 0

    while collected < num_samples and max_tries > 0:
        max_tries -= 1

        params = sample_params()
        traj = model.simulate(params, T, dt, initial_conditions=[0.5, 0.5, 1.0])  # strong dynamics

        # REJECT flat or non-dynamic trajectories
        if torch.std(traj[1]) < 1e-4:  # I(t) must vary
            continue

        if inverse:
            x_list.append(traj)
            y_list.append(params.unsqueeze(-1).repeat(1, T))
        else:
            x_list.append(params.unsqueeze(-1).repeat(1, T))
            y_list.append(traj)

        collected += 1

    print(f"Generated {collected} dynamic samples.")
    return torch.stack(x_list), torch.stack(y_list)
