import torch
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models.fno import FNO1d
from dynamics.dong_model import DongRumorModel
from utils.data_generation import generate_dataset  # ✅ Unified data generation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === PARAMETERS ===
T, dt = 50, 0.01
T_steps = int(T / dt)
INIT_CONDS = [1.5,0.05,7.9]
NUM_SAMPLES = 1000
BATCH_SIZE = 32

# === FORWARD TRAINING ===
def train_forward(model_system, epochs):
    x, y = generate_dataset(model_system, NUM_SAMPLES, T, dt, inverse=False, file_name='trajectories_dong_model.pth')
    y = y[:, :, :T_steps]
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

    model = FNO1d(4, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()  # ✅ Fixed
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Epoch {epoch}] Forward Loss: {sum(losses)/len(losses):.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/dong_fno_forward.pth")

# === INVERSE TRAINING ===
def train_inverse(model_system, epochs):
    x, y = generate_dataset(model_system, NUM_SAMPLES, T, dt, inverse=True, file_name="parameters_dong_model.pth")
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

    model = FNO1d(4, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()  # ✅ Fixed
            pred = model(xb).mean(dim=-1)
            loss = torch.nn.functional.mse_loss(pred, yb[:, :, 0])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Epoch {epoch}] Inverse Loss: {sum(losses)/len(losses):.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/dong_fno_inverse.pth")

# === INFERENCE / VISUALIZATION ===
def inference(model_system):
    model = FNO1d(4, 3).to(device)  # ✅ Fixed input channels
    model.load_state_dict(torch.load("checkpoints/dong_fno_forward.pth"))
    model.eval()

    true_params = torch.tensor([0.3,1.6, 0.28])
    traj = model_system.simulate(true_params, T, dt, initial_conditions=INIT_CONDS)[:, :T_steps]

    time_grid = torch.linspace(0, 1, T_steps).unsqueeze(0).to(device)
    param_repeated = true_params.unsqueeze(-1).repeat(1, T_steps).to(device)
    inp = torch.cat([param_repeated, time_grid], dim=0).unsqueeze(0)

    with torch.no_grad():
        pred = model(inp).cpu().squeeze(0)
    true_S, true_I, true_N = traj
    pred_S, pred_I, pred_N = pred[:,:T_steps]

    labels = ['S(t)', 'I(t)','N(t)']
    print(f"True S(t): mean={true_S.mean():.5f}, std={true_S.std():.5e}")
    print(f"True I(t): mean={true_I.mean():.5f}, std={true_I.std():.5e}")
    print(f"True N(t): mean={true_N.mean():.5f}, std={true_N.std():.5e}")
    print(f"Pred S(t): mean={pred_S.mean():.5f}, std={pred_S.std():.5e}")
    print(f"Pred I(t): mean={pred_I.mean():.5f}, std={pred_I.std():.5e}")
    print(f"Pred N(t): mean={pred_N.mean():.5f}, std={pred_N.std():.5e}")



    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(traj[i], label=f"True {{labels[i]}}")
        plt.plot(pred[i], '--', label=f"Pred {{labels[i]}}")
    plt.title("Dong FNO: Prediction vs Ground Truth")
    plt.xlabel("Time step")
    plt.ylabel("Proportion")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === MAIN CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dong model FNO pipeline")
    parser.add_argument("task", choices=["train_forward", "train_inverse", "inference", "all"])
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()

    model_system = DongRumorModel()

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