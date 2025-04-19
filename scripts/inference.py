import torch
import matplotlib.pyplot as plt
from models.fno import FNO1d
from dynamics.dong_model import DongRumorModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained forward model explicitly
model = FNO1d(3, 3).to(device)
model.load_state_dict(torch.load("checkpoints/dong_fno_forward.pth"))
model.eval()

# True trajectory from Dong model
dong_model = DongRumorModel()
true_params = torch.tensor([0.15, 0.1, 0.02])
traj = dong_model.simulate(true_params, 200, 0.1, initial_conditions=[0.99,0.01,1.0])

# Model input
input_params = true_params.unsqueeze(0).unsqueeze(-1).repeat(1,1,200).to(device)

# Predict clearly
with torch.no_grad():
    pred = model(input_params).cpu().squeeze(0)

# Plot results clearly
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
