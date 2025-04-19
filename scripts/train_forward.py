import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from models.fno import FN1d
from dynamics.dong_model import DongRumorModel
from utils.data_generation import generate_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T, dt = 20, 0.05
NUM_SAMPLES = 1000
BATCH_SIZE = 32

if __name__ == "__main__":
    model_system = DongRumorModel()
    x, y = generate_dataset(model_system, NUM_SAMPLES, T, dt, inverse=False, file_name='trajectories_dong_model.pth')
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

    model = FNO1d(in_channels=4, out_channels=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()  # ✅ FIXED
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Epoch {epoch}] Forward Loss: {sum(losses)/len(losses):.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/dong_fno_forward.pth")