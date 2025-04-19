import torch
import argparse
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from models.fno import FNO1d
from dynamics.dong_model import DongRumorModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === PARAMETERS ===
T, dt = 1000, 0.05
INIT_CONDS = [0.8, 0.05, 1.3] #S I N
NUM_SAMPLES = 1000
BATCH_SIZE = 32

# === PARAMETER SAMPLING ===
def sample_params():
    alpha = torch.empty(1).uniform_(0.1,0.9)
    beta  = torch.empty(1).uniform_(0.1, 0.9)
    delta = torch.empty(1).uniform_(0.1,0.9)
    return torch.cat([alpha, beta, delta], dim=0)

# === DATA GENERATION ===
def generate_dataset(model, num_samples, T, dt, inverse=False, file_name=None):
    x_list, y_list = [], []
    for _ in range(num_samples):
        params = sample_params()
        traj = model.simulate(params, T, dt, initial_conditions=INIT_CONDS)
        if inverse:
            x_list.append(traj)
            y_list.append(params.unsqueeze(-1).repeat(1, T))
        else:
            x_list.append(params.unsqueeze(-1).repeat(1, T))
            y_list.append(traj)

    if file_name !=None and type(file_name) == 'str':
        data_path = os.path.join("..","data")

        if file_name[:-4] != '.pth':
            file_name = file_name + '.pth'
        file_path = os.path.join(data_path, file_name)
        torch.save([torch.stack(x_list), torch.stack(y_list)], file_name)
        
    return torch.stack(x_list), torch.stack(y_list)

# === FORWARD TRAINING ===
def train_forward(model_system, epochs):
    x, y = generate_dataset(model_system, NUM_SAMPLES, T, dt, inverse=False, file_name='trajectories_dong_model.pth')
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

    model = FNO1d(3, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Epoch {epoch}] Forward Loss: {sum(losses)/len(losses):.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/dong_fno_forward.pth")

# === INVERSE TRAINING ===
def train_inverse(model_system, epochs):
    x, y = generate_dataset(model_system, NUM_SAMPLES, T, dt, inverse=True, 
                            file_name=f"parameters_dong_model.pth")
    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

    model = FNO1d(3, 3).to(device)
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

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/dong_fno_inverse.pth")

# === INFERENCE / VISUALIZATION ===
def inference(model_system):
    model = FNO1d(3, 3).to(device)
    model.load_state_dict(torch.load("checkpoints/dong_fno_forward.pth"))
    model.eval()

    # True trajectory
    true_params = torch.tensor([0.9, 1.4, 0.9])
    traj = model_system.simulate(true_params, T, dt, initial_conditions=INIT_CONDS)

    # Input shape [1, 3, T]
    inp = true_params.unsqueeze(0).unsqueeze(-1).repeat(1, 1, T).to(device)
    with torch.no_grad():
        pred = model(inp).cpu().squeeze(0)

    labels = ['S(t)', 'I(t)', 'N(t)']
    for name, series in zip(["S", "I", "N"], traj):
        std = torch.std(series)
        mean = torch.mean(series)
        print(f"True {name}(t): mean={mean:.5f}, std={std:.5e}")

    for name, series in zip(["S", "I", "N"], pred):
        std = torch.std(series)
        mean = torch.mean(series)
        print(f"Pred {name}(t): mean={mean:.5f}, std={std:.5e}")

    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(traj[i], label=f"True {labels[i]}")
        plt.plot(pred[i], '--', label=f"Pred {labels[i]}")
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
