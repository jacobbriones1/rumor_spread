from .base import DynamicalSystem
import numpy as np
import networkx as nx
import torch

class SIRModel(DynamicalSystem):
    def simulate(self, params, T=200, dt=0.1, N=120, graph_type="ER"):
        beta, alpha, delta, i0 = params

        # Generate graph based on the topology
        if graph_type == "ER":
            G = nx.erdos_renyi_graph(N, 0.05)
        elif graph_type == "BA":
            G = nx.barabasi_albert_graph(N, 3)
        elif graph_type == "WS":
            G = nx.watts_strogatz_graph(N, 6, 0.1)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        states = np.zeros(N, dtype=int)  # 0: Susceptible, 1: Infected, 2: Recovered

        # Initialize infected nodes
        init_infected = np.random.choice(N, max(1, int(i0 * N)), replace=False)
        states[init_infected] = 1

        trajectory = []

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

            # Compute proportions clearly
            S = np.mean(states == 0)
            I = np.mean(states == 1)
            R = np.mean(states == 2)
            
            trajectory.append([S, I, R])

        # Convert trajectory into torch tensor (shape: [3, T])
        return torch.tensor(trajectory, dtype=torch.float32).T

    def parameter_dim(self):
        return
