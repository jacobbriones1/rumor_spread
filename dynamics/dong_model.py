from .base import DynamicalSystem
import torch

class DongRumorModel(DynamicalSystem):
    """
        Nondimensionalized version of the system by Dong et al. (2018)
    """
    def simulate(self, params, T=200, dt=0.05, **kwargs):
        alpha, beta, delta = params.tolist()
        
        # Sensible initial conditions (rumor starts with a small spark)
        S, I, N = kwargs.get("initial_conditions", [0.8, 0.2, 1.0])
        
        def clip(x, low=0.0, high=1.0):
            return max(low, min(x, high))

        max_d = 0.05  # max allowed change per time step
        trajectory = []

        for _ in range(T):
            # Discrete-time Euler update (clipped for stability)

            dS = (S * (1 - N) - alpha * S * I) * dt
            dI = (beta * (N - S - I) - (delta + N) * I) * dt
            dN = ((1 + delta) * S - (delta + N) * N) * dt

            # Clip derivatives
            dS = max(min(dS, max_d), -max_d)
            dI = max(min(dI, max_d), -max_d)
            dN = max(min(dN, max_d), -max_d)

            # Update state
            S += dS
            I += dI
            N += dN

            # Clip states to prevent nonphysical values
            S = clip(S)
            I = clip(I)
            N = clip(N, 0.0, 2.0)

            trajectory.append([S, I, N])

        return torch.tensor(trajectory, dtype=torch.float32).T

    def parameter_dim(self):
        return 3  # alpha, beta, delta

    def state_dim(self):
        return 3  # S, I, N
