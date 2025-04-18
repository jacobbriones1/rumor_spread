import torch
import matplotlib.pyplot as plt
from models.fno import FNO1d
from dynamics.dong_model import DongRumorModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T_steps = 200
dt = 0.1
T = T_steps * dt

# Load trained forward model with correct input shape
model = FNO1d(4, 3).to(device)  # ✅ FIXED input channels
model.load_state_dict(torch.load("checkpoints/dong_fno_forward.pth"))
model.eval()

# True trajectory
dong_model = DongRumorModel()
true_params = torch.tensor([0.15, 0.1, 0.02])
traj = dong_model.simulate(true_params, T_steps, dt, initial_conditions=[0.99,0.01,1.0])

# Create correct input: [params repeated + time]
time_grid = torch.linspace(0, 1, T_steps).unsqueeze(0).to(device)
param_repeated = true_params.unsqueeze(-1).repeat(1, T_steps).to(device)
input_tensor = torch.cat([param_repeated, time_grid], dim=0).unsqueeze(0)

# Predict
with torch.no_grad():
    pred = model(input_tensor).cpu().squeeze(0)

# Plot
labels = ['S(t)', 'I(t)', 'N(t)']
for i in range(3):
    plt.plot(traj[i], label=f"True {labels[i]}")
    plt.plot(pred[i],'--', label=f"Pred {labels[i]}")

plt.xlabel("Time step")
plt.ylabel("Population proportion")
plt.legend()
plt.title("Dong et al. Model Predictions vs Ground Truth")
plt.grid()
plt.show()