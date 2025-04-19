# utils/data_generation.py
import os
import torch
from tqdm import tqdm
from typing import Optional, Tuple

DEFAULT_INIT_CONDS = [0.8, 0.1, 0.2]
DEFAULT_DATA_DIR = "../data"

# === PARAMETER SAMPLING ===
def sample_params(param_dim: int) -> torch.Tensor:
    if param_dim == 3:
        return torch.cat([
            torch.empty(1).uniform_(0.1, 0.9),   # alpha
            torch.empty(1).uniform_(0.1, 0.9),   # beta
            torch.empty(1).uniform_(0.01, 0.9)   # delta
        ])
    elif param_dim == 4:
        return torch.cat([
            torch.empty(1).uniform_(0.1, 0.9),   # beta
            torch.empty(1).uniform_(0.1, 0.9),   # alpha
            torch.empty(1).uniform_(0.01, 0.9),  # delta
            torch.empty(1).uniform_(0.01, 0.5)   # i0
        ])
    else:
        raise ValueError(f"Unsupported param_dim: {param_dim}")

# === DATA GENERATION ===
def generate_dataset(
    model,
    num_samples: int,
    T: float,
    dt: float,
    inverse: bool = False,
    file_name: Optional[str] = None,
    data_dir: str = DEFAULT_DATA_DIR,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:

    if seed is not None:
        torch.manual_seed(seed)

    x_list, y_list = [], []
    steps = int(T / dt)
    time_grid = torch.linspace(0, 1, steps).unsqueeze(0)  # Shape: (1, T)

    param_dim = model.parameter_dim() if hasattr(model, "parameter_dim") else 4
    state_dim = model.state_dim() if hasattr(model, "state_dim") else 3
    has_initial_conditions = "initial_conditions" in model.simulate.__code__.co_varnames

    for _ in tqdm(range(num_samples), desc="Generating dataset"):
        params = sample_params(param_dim)

        if has_initial_conditions:
            traj = model.simulate(params, T, dt, initial_conditions=DEFAULT_INIT_CONDS)
        else:
            traj = model.simulate(params, T, dt)

        traj = traj[:, :steps]  # Safety trim
        param_repeated = params.unsqueeze(-1).repeat(1, steps)

        if inverse:
            full_input = torch.cat([traj, time_grid], dim=0)
            target = param_repeated
        else:
            full_input = torch.cat([param_repeated, time_grid], dim=0)
            target = traj

        x_list.append(full_input)
        y_list.append(target)

    x_tensor = torch.stack(x_list)
    y_tensor = torch.stack(y_list)

    if file_name:
        if not file_name.endswith('.pth'):
            file_name += '.pth'
        os.makedirs(data_dir, exist_ok=True)
        torch.save([x_tensor, y_tensor], os.path.join(data_dir, file_name))

    return x_tensor, y_tensor