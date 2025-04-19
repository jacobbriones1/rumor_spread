from .base import DynamicalSystem
import torch
import networkx as nx
from collections import defaultdict

class DegreeAwareSIRModel(DynamicalSystem):
    def simulate(
        self,
        params,
        T=200,
        dt=0.1,
        N=120,
        graph_type="ER",
        initial_conditions=None
    ):
        beta, alpha, delta, i0 = params

        # 1. Generate the graph
        if graph_type == "ER":
            G = nx.erdos_renyi_graph(N, 0.05)
        elif graph_type == "BA":
            G = nx.barabasi_albert_graph(N, 3)
        elif graph_type == "WS":
            G = nx.watts_strogatz_graph(N, 6, 0.1)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        # 2. Get degrees
        degrees = torch.tensor([G.degree(n) for n in G.nodes()], dtype=torch.int32)
        max_k = degrees.max().item()

        # 3. Estimate conditional degree probability matrix P(k'|k)
        edge_counts = defaultdict(lambda: defaultdict(int))
        for u, v in G.edges():
            ku, kv = degrees[u].item(), degrees[v].item()
            edge_counts[ku][kv] += 1
            edge_counts[kv][ku] += 1

        P_kprime_given_k = torch.zeros((max_k + 1, max_k + 1))
        for k in edge_counts:
            total = sum(edge_counts[k].values())
            if total > 0:
                for k_prime in edge_counts[k]:
                    P_kprime_given_k[k, k_prime] = edge_counts[k][k_prime] / total

        # 4. Initialize states
        states = torch.zeros(N, dtype=torch.int32)  # 0: S, 1: I, 2: R
        num_infected = max(1, int(i0 * N))
        infected_indices = torch.randperm(N)[:num_infected]
        states[infected_indices] = 1

        neighbors = [list(G.neighbors(i)) for i in range(N)]
        T_steps = int(T / dt)
        trajectory = torch.zeros((T_steps, 3), dtype=torch.float32)

        for t in range(T_steps):
            for i in range(N):
                state_i = states[i].item()
                k_i = degrees[i].item()

                if state_i == 0:
                    for j in neighbors[i]:
                        if states[j] == 1:
                            k_j = degrees[j].item()
                            contact_prob = 1 - torch.exp(-beta * P_kprime_given_k[k_j, k_i] * dt)
                            if torch.rand(1).item() < contact_prob:
                                states[i] = 1
                                break

                elif state_i == 1:
                    if torch.rand(1).item() < 1 - torch.exp(-alpha * dt):
                        states[i] = 2
                    elif torch.rand(1).item() < 1 - torch.exp(-delta * dt):
                        states[i] = 2

            S_frac = torch.mean((states == 0).float())
            I_frac = torch.mean((states == 1).float())
            R_frac = torch.mean((states == 2).float())
            trajectory[t] = torch.tensor([S_frac, I_frac, R_frac])

        return trajectory.T  # shape: [3, T_steps]

    def parameter_dim(self):
        return 4

    def state_dim(self):
        return 3
