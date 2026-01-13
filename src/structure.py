import numpy as np
from evogym.utils import is_connected, has_actuator


class Structure:
    def __init__(self, body: np.ndarray, connections: np.ndarray | None = None) -> None:
        self.body = body
        self.fitness = 0.0
        self.connections = connections

    def __lt__(self, other: "Structure") -> bool:
        return self.fitness < other.fitness

    def is_valid(self) -> bool:
        if np.sum(self.body) == 0:
            return False
        if not is_connected(self.body):
            return False
        if not has_actuator(self.body):
            return False
        return True
    
    def get_1D_body(self) -> np.ndarray:
        return self.body.reshape(-1)
    
    def render_robot_and_save(self, save_path: str, env_type: str = 'Walker-v0') -> None:
        import gymnasium as gym
        import evogym.envs
        from PIL import Image

        env = gym.make(env_type, body=self.body, render_mode='rgb_array')
        env.reset()
        frame = env.render()
        
        im = Image.fromarray(frame)
        im.save(save_path)
        env.close()
