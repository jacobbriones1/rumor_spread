# dynamics/base.py
from abc import ABC, abstractmethod

class DynamicalSystem(ABC):
    @abstractmethod
    def simulate(self, params, T, dt, **kwargs):
        pass

    @abstractmethod
    def parameter_dim(self):
        pass

    @abstractmethod
    def state_dim(self):
        pass
