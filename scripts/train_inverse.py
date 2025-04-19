import torch
from torch.utils.data import DataLoader, TensorDataset
from models.fno import FNO1d
from utils.data_generation import generate_dataset
from dynamics.sir_heterogeneous import DegreeAwareSIRModel
import os
import gc

# === Hyperparameters ===
T = 50
dt = 0.1
T_steps = int(T / dt)
NUM_SAMPLES = 100   # ✅ reduced for laptop safety
BATCH_SIZE = 16     # ✅ smaller batch
EPOCHS = 50
LR = 1e-3
SAVE_PATH = "checkpoints/fno_inverse_heterogeneous.pth"
DATA_PATH = "../data/heterogeneous_inverse.pth"

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
        inverse=True,
        file_name="heterogeneous_inverse"
    )

# === DataLoader ===
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model ===
model = FNO1d(in_channels=4, out_channels=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    losses = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb).mean(dim=-1)       # Output shape: [N, 4]
        target = yb[:, :, 0]                # Target shape: [N, 4]
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"[Epoch {epoch+1:02d}] Inverse Loss: {sum(losses)/len(losses):.6f}")

    # === Clean memory ===
    torch.cuda.empty_cache()
    gc.collect()

# === Save Model ===
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ Model saved to {SAVE_PATH}")
