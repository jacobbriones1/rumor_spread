import torch
from torch.utils.data import DataLoader, TensorDataset
from models.fno import FNO1d
from utils.data_generation import generate_dataset
from dynamics.sir_heterogeneous import DegreeAwareSIRModel
import os
import gc

# === Hyperparameters ===
T = 200
dt = 0.1
T_steps = int(T / dt)
NUM_SAMPLES = 258  # reduced for laptop safety
BATCH_SIZE = 16
EPOCHS = 32
LR = 1e-3
SAVE_PATH = "checkpoints/fno_forward_heterogeneous.pth"
DATA_PATH = "../data/heterogeneous_forward.pth"

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Initialize the rumor model
sir_model = DegreeAwareSIRModel()

# === Load or generate dataset ===
if os.path.exists(DATA_PATH):
    print(f"📦 Loading dataset from {DATA_PATH}")
    x, y = torch.load(DATA_PATH, map_location='cpu')
else:
    print("🧪 Generating dataset...")
    x, y = generate_dataset(
        sir_model,
        num_samples=NUM_SAMPLES,
        T=T,
        dt=dt,
        inverse=False,
        file_name="heterogeneous_forward"
    )

# === DataLoader ===
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model ===
model = FNO1d(in_channels=5,out_channels=3, n_modes=16, width=32, depth = 6).to(device)  # 4 params + 1 time
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    losses = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)                    # Output shape: [N, 3, T]
        loss = torch.nn.functional.mse_loss(pred, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"[Epoch {epoch+1:02d}] Forward Loss: {sum(losses)/len(losses):.6f}")

    # === Clean memory ===
    torch.cuda.empty_cache()
    gc.collect()

# === Save Model ===
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ Model saved to {SAVE_PATH}")