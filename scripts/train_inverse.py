import torch
from torch.utils.data import DataLoader, TensorDataset
from models.fno import FNO1d
from dynamics.dong_model import DongRumorModel
import os

def sample_params():
    alpha = torch.empty(1).uniform_(1.6, 2.0)
    beta  = torch.empty(1).uniform_(0.8, 1.2)
    delta = torch.empty(1).uniform_(0.02, 0.06)
    return torch.cat([alpha, beta, delta], dim=0)

def generate_dataset(model, num_samples, T=200, dt=0.05):
    x_list, y_list = [], []
    for _ in range(num_samples):
        params = sample_params()
        traj = model.simulate(params, T, dt, initial_conditions=[0.8, 0.2, 1.0])
        x_list.append(traj)
        y_list.append(params.unsqueeze(-1).repeat(1, T))
    return torch.stack(x_list), torch.stack(y_list)

# Init
model_system = DongRumorModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data (Inverse: trajectory -> parameter)
x, y = generate_dataset(model_system, num_samples=500)
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

# FNO Inverse Model
model = FNO1d(in_channels=4, out_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
for epoch in range(50):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb).mean(dim=-1)  # mean across time
        loss = torch.nn.functional.mse_loss(pred, yb[:, :, 0])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"[Epoch {epoch}] Inverse Loss: {sum(losses)/len(losses):.6f}")

# Save
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/dong_fno_inverse.pth")
