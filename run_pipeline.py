import torch
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models.fno import FNO1d
from dynamics.sir_heterogeneous import DegreeAwareSIRModel
from utils.data_generation import generate_dataset
import gc

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === PARAMETERS ===
T, dt = 50, 0.5
T_steps = int(T / dt)
NUM_SAMPLES = 200
BATCH_SIZE = 32

# === FORWARD TRAINING ===
def train_forward(model_system, epochs):
    x, y = generate_dataset(model_system, NUM_SAMPLES, T, dt, inverse=False, file_name='heterogeneous_forward')
    y = y[:, :, :T_steps]
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

    model = FNO1d(in_channels=5, out_channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            if pred.shape[-1] != yb.shape[-1]:
                pred = pred[:, :, :yb.shape[-1]]


            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Epoch {epoch}] Forward Loss: {sum(losses)/len(losses):.6f}")
        torch.cuda.empty_cache()
        gc.collect()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fno_forward_heterogeneous.pth")

# === INVERSE TRAINING ===
def train_inverse(model_system, epochs):
    x, y = generate_dataset(model_system, NUM_SAMPLES, T, dt, inverse=True, file_name="heterogeneous_inverse")
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

    model = FNO1d(in_channels=4, out_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).mean(dim=-1)
            loss = torch.nn.functional.mse_loss(pred, yb[:, :, 0])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Epoch {epoch}] Inverse Loss: {sum(losses)/len(losses):.6f}")
        torch.cuda.empty_cache()
        gc.collect()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fno_inverse_heterogeneous.pth")

# === INFERENCE / VISUALIZATION ===
def inference(model_system):
    model = FNO1d(in_channels=5, out_channels=3).to(device)
    model.load_state_dict(torch.load("checkpoints/fno_forward_heterogeneous.pth", map_location=device))
    model.eval()

    true_params = torch.tensor([0.9, 0.2, 0.14, 0.1])  # beta, alpha, delta, i0
    traj = model_system.simulate(true_params, T, dt)[:, :T_steps]

    time_grid = torch.linspace(0, 1, T_steps).unsqueeze(0).to(device)
    param_repeated = true_params.unsqueeze(-1).repeat(1, T_steps).to(device)
    inp = torch.cat([param_repeated, time_grid], dim=0).unsqueeze(0)

    with torch.no_grad():
        pred = model(inp).cpu().squeeze(0)

    colors = ['blue','red','green']
    labels = ['S(t)', 'I(t)', 'R(t)']
    plt.figure(figsize=(8, 5))
    for i in range(3):
        plt.plot(traj[i], '--',label=f"True {labels[i]}", lw=0.8, color = colors[i])
        plt.plot(pred[i],  label=f"Pred {labels[i]}", lw=1., color = colors[i])
    plt.title("FNO Heterogeneous SIR: Prediction vs Ground Truth")
    plt.xlabel("Time step")
    plt.ylabel("Proportion")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# === MAIN CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run heterogeneous SIR FNO pipeline")
    parser.add_argument("task", choices=["train_forward", "train_inverse", "inference", "all"])
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    model_system = DegreeAwareSIRModel()

    if args.task == "train_forward":
        train_forward(model_system, args.epochs)
    elif args.task == "train_inverse":
        train_inverse(model_system, args.epochs)
    elif args.task == "inference":
        inference(model_system)
    elif args.task == "all":
        train_forward(model_system, args.epochs)
        train_inverse(model_system, args.epochs)
        inference(model_system)
