import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import Counter
from torch.utils.data import TensorDataset


# === ER Model with Spontaneous Forgetting ===

def simulate_rumor_on_er(N=100, p=0.05, beta=0.3, alpha=0.2, delta=0.1, i0=0.05, T=100, dt=0.1):
    """
    Simulates Maki–Thompson rumor dynamics with spontaneous forgetting (δ)
    on an Erdős–Rényi graph. Returns average [S, I, R] over time.

    Parameters:
        N     : number of nodes
        p     : ER connection probability
        beta  : infection rate (S + I → I)
        alpha : stifling rate (I + I/R → R)
        delta : spontaneous forgetting rate (I → R)
        i0    : initial infected fraction
        T     : number of time steps
        dt    : time step size

    Returns:
        traj : torch.Tensor of shape (3, T) — mean S, I, R over time
    """
    G = nx.erdos_renyi_graph(N, p)
    A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)

    S = torch.ones(N)
    I = (torch.rand(N) < i0).float()
    S -= I
    R = torch.zeros(N)

    beta_dt  = beta * dt
    alpha_dt = alpha * dt
    delta_dt = delta * dt

    traj = []

    for _ in range(T):
        traj.append(torch.stack([S.mean(), I.mean(), R.mean()]))

        infected_neighbors = A @ I

        p_infect   = beta_dt  * infected_neighbors * S
        p_stifling = alpha_dt * (A @ (I + R)) * I
        p_forget   = delta_dt * I

        new_infect = (torch.rand(N) < p_infect).float()
        new_stifle = (torch.rand(N) < p_stifling).float()
        new_forget = (torch.rand(N) < p_forget).float()

        S -= new_infect
        I += new_infect - new_stifle - new_forget
        R += new_stifle + new_forget

    return torch.stack(traj, dim=1)


def generate_dataset_er(N=100, num_samples=1000, T=100, dt=0.1):
    """
    Generates a dataset of rumor dynamics on Erdős–Rényi graphs.
    Saves `theta_tensor` (β, α, δ, i₀) and `traj_tensor` ([S,I,R] trajectories).
    """
    thetas = []
    trajs = []

    for _ in tqdm(range(num_samples), desc="Generating ER Rumor Dataset"):
        beta  = np.random.uniform(0.1, 0.5)
        alpha = np.random.uniform(0.1, 0.4)
        delta = np.random.uniform(0.01, 0.2)
        i0    = np.random.uniform(0.01, 0.1)

        traj = simulate_rumor_on_er(N=N, p=0.05, beta=beta, alpha=alpha, delta=delta, i0=i0, T=T, dt=dt)
        thetas.append(torch.tensor([beta, alpha, delta, i0], dtype=torch.float32))
        trajs.append(traj)

    theta_tensor = torch.stack(thetas)                          # (N, 4)
    traj_tensor  = torch.stack(trajs)                           # (N, 3, T)
    theta_tensor = theta_tensor.unsqueeze(1).expand(-1, T, -1)  # (N, T, 4)

    torch.save((theta_tensor, traj_tensor), "rumor_dataset_er.pt")
    print("Saved dataset to 'rumor_dataset_er.pt'")


# === Mean-Field Scale-Free Model ===

def simulate_rumor_meanfield_sf(N=1000, m=3, beta=0.3, alpha=0.1, delta=0.05, T=100, dt=0.1):
    """
    Simulates rumor dynamics using degree-based mean-field equations
    on a scale-free (Barabási–Albert) network.

    Parameters:
        N     : number of nodes
        m     : edges to attach from a new node (BA model)
        beta  : infection rate
        alpha : stifling rate
        delta : spontaneous forgetting rate
        T     : number of time steps
        dt    : time step size

    Returns:
        traj  : torch.Tensor of shape (3, T) representing avg [S(t), I(t), R(t)]
    """
    G = nx.barabasi_albert_graph(N, m)
    degrees = [deg for _, deg in G.degree()]
    k_vals, counts = np.unique(degrees, return_counts=True)
    Pk = {k: count / N for k, count in zip(k_vals, counts)}
    k_list = sorted(Pk.keys())

    # Initial densities: all ignorant except a small spreader fraction
    rho_S = {k: 1.0 for k in k_list}
    rho_I = {k: 0.01 for k in k_list}
    rho_R = {k: 0.0 for k in k_list}
    for k in k_list:
        rho_S[k] -= rho_I[k]

    traj = []

    for _ in range(T):
        avg_S = sum(Pk[k] * rho_S[k] for k in k_list)
        avg_I = sum(Pk[k] * rho_I[k] for k in k_list)
        avg_R = sum(Pk[k] * rho_R[k] for k in k_list)
        traj.append(torch.tensor([avg_S, avg_I, avg_R]))

        kPk    = {k: k * Pk[k] for k in k_list}
        kPkI   = {k: k * Pk[k] * rho_I[k] for k in k_list}
        k_avg  = sum(kPk.values())
        Theta  = sum(kPkI.values()) / k_avg if k_avg > 0 else 0.0

        new_rho_S = {}
        new_rho_I = {}
        new_rho_R = {}

        for k in k_list:
            rS = rho_S[k]
            rI = rho_I[k]
            rR = rho_R[k]

            dS = -beta * k * rS * Theta
            dI = beta * k * rS * Theta - alpha * k * rI * (rI + rR) - delta * rI
            dR = alpha * k * rI * (rI + rR) + delta * rI

            new_rho_S[k] = rS + dt * dS
            new_rho_I[k] = rI + dt * dI
            new_rho_R[k] = rR + dt * dR

        rho_S, rho_I, rho_R = new_rho_S, new_rho_I, new_rho_R

    return torch.stack(traj, dim=1)
