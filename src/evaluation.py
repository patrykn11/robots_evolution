import gymnasium as gym
import evogym.envs
import numpy as np
from structure import Structure
from controller import Controller

OMEGA = 5 * np.pi
PHASE_K = -2
AMPLITUDE = 0.5
OFFSET = 1.0


def evaluate(structure: Structure, env_type) -> float:
    if not structure.is_valid():
        return 0.0
    env = gym.make(env_type, body=structure.body)
    env.reset()
    action_space = env.action_space
    if action_space is None or action_space.shape[0] == 0:
        env.close()
        return 0.0
        
    muscle_indices = np.where((structure.body == 3) | (structure.body == 4))
    muscle_x = muscle_indices[1]

    controller = Controller(omega=OMEGA, phase_coefficent=PHASE_K,
                            amplitude=AMPLITUDE, offset=OFFSET)
    
    total_reward = 0

    for i in range(500):
        t = i * 0.019
        action = controller.action_signal(t, muscle_x)
        
        step_result = env.step(action)
        
        if len(step_result) == 5:
            _, reward, done, truncated, _ = step_result
            done = done or truncated
        else:
            _, reward, done, _ = step_result

        total_reward += reward
        if done:
            break

    env.close()
    return total_reward
