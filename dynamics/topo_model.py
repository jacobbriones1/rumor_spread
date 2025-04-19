# === Modular Topology-Aware FNO Integration ===
# This integrates topology-based rumor simulations and learning into the modular FNO framework
from .base import DynamicalSystem
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import networkx as nx


class TopoRumorModel(DynamicalSystem):
    def __init__(self, graph_type='ER', N=120, dt=0.1):
        super().__init__()
        self.graph_type = graph_type
        self.N = N
        self.dt = dt

    def generate_graph(self):
        if self.graph_type == "ER":
            G = nx.erdos_renyi_graph(self.N, 0.05)
        elif self.graph_type == "BA":
            G = nx.barabasi_albert_graph(self.N, 3)
        elif self.graph_type == "WS":
            G = nx.watts_strogatz_graph(self.N, 6, 0.1)
        else:
            raise ValueError("Unknown graph type")
        return G

    def extract_topo_features(self, G):
        clustering = nx.average_clustering(G)
        try:
            path_len = nx.average_shortest_path_length(G)
        except:
            path_len = 0.0
        path_len = path_len / self.N if np.isfinite(path_len) else 0.0

        assort = nx.degree_assortativity_coefficient(G)
        assort = assort if np.isfinite(assort) else 0.0

        return torch.tensor([clustering, path_len, assort], dtype=torch.float32)

    def simulate(self, params, T, dt=None, **kwargs):
        beta, alpha, delta, i0 = params
        dt = dt or self.dt
        G = self.generate_graph()
        topo_feats = self.extract_topo_features(G)

        states = np.zeros(self.N, dtype=int)
        infected = np.random.choice(self.N, max(1, int(i0 * self.N)), replace=False)
        states[infected] = 1

        traj = []
        for _ in range(T):
            new_states = states.copy()
            for node in G.nodes:
                if states[node] == 0:
                    infected_neighbors = sum(states[nbr] == 1 for nbr in G.neighbors(node))
                    if np.random.rand() < 1 - np.exp(-beta * infected_neighbors * dt):
                        new_states[node] = 1
                elif states[node] == 1:
                    if np.random.rand() < alpha * dt or np.random.rand() < delta * dt:
                        new_states[node] = 2
            states = new_states
            S = np.sum(states == 0) / self.N
            I = np.sum(states == 1) / self.N
            R = np.sum(states == 2) / self.N
            traj.append([S, I, R])

        traj = torch.tensor(traj, dtype=torch.float32).T  # [3, T]
        return traj, topo_feats

    def parameter_dim(self):
        return 4

    def state_dim(self):
        return 3


# === Dataset Generation ===
def generate_topology_dataset(model: TopoRumorModel, T=200, samples=512):
    x_list, y_list = [], []
    for _ in range(samples):
        theta = [
            np.random.uniform(0.01, 0.2),  # beta
            np.random.uniform(0.01, 0.15), # alpha
            np.random.uniform(0.001, 0.05),# delta
            np.random.uniform(0.001, 0.01) # i0
        ]
        traj, topo = model.simulate(theta, T)
        time = torch.linspace(0, 1, T).unsqueeze(0)       # [1, T]
        theta = torch.tensor(theta).unsqueeze(1).expand(-1, T)  # [4, T]
        x = torch.cat([theta, time], dim=0)               # [5, T]
        x_list.append(x)
        y_list.append(topo)
    return TensorDataset(torch.stack(x_list), torch.stack(y_list))


# === Inverse Model (FNO) Trainer ===
def train_topology_inverse_fno(model, dataset, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch:03d} | Loss: {np.mean(losses):.6f}")

    return model
