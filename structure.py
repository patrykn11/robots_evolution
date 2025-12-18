import numpy as np
from evogym.utils import is_connected, has_actuator


class Structure:
    def __init__(self, body, connections=None):
        self.body = body
        self.fitness = 0.0
        self.connections = connections

    def __lt__(self, other):
        return self.fitness < other.fitness

    def is_valid(self):
        if np.sum(self.body) == 0:
            return False
        if not is_connected(self.body):
            return False
        if not has_actuator(self.body):
            return False
        return True
    
    def get_1D_body(self):
        return self.body.reshape(-1)
