from .base import DynamicalSystem
import numpy as np
import networkx as nx
import torch

class SIRModel(DynamicalSystem):
    def simulate(self, params, T=200, dt=0.1, N=120, graph_type="ER"):
        beta, alpha, delta, i0 = params

        # 1. Create graph based on type
        if graph_type == "ER":
            G = nx.erdos_renyi_graph(N, 0.05)
        elif graph_type == "BA":
            G = nx.barabasi_albert_graph(N, 3)
        elif graph_type == "WS":
            G = nx.watts_strogatz_graph(N, 6, 0.1)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

        # 2. Node states: 0 = S, 1 = I, 2 = R 
        states = np.zeros(N, dtype=int)
        init_infected = np.random.choice(N, max(1, int(i0 * N)), replace=False)
        states[init_infected] = 1

        trajectory = []
    
        for _ in range(T):
            for node in range(N):
                if states[node] == 0:  # Susceptible
                    infected_neighbors = sum(states[neigh] == 1 for neigh in G.neighbors(node))
                    infection_rate = beta * infected_neighbors
                    # Poisson process: Prob(infection in dt) = 1 - exp(-rate * dt)
                    if np.random.rand() < 1 - np.exp(-infection_rate * dt):
                        states[node] = 1

                elif states[node] == 1:  # Infected
                    # Two independent Poisson processes
                    if np.random.rand() < 1 - np.exp(-alpha * dt):  # stifling
                        states[node] = 2
                    elif np.random.rand() < 1 - np.exp(-delta * dt):  # forgetting
                        states[node] = 2

                # Recovered do nothing

            # Record proportions
            S = np.mean(states == 0)
            I = np.mean(states == 1)
            R = np.mean(states == 2)
            trajectory.append([S, I, R])

        return torch.tensor(trajectory, dtype=torch.float32).T

    def parameter_dim(self):
        return 4
