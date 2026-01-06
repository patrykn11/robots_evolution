import gymnasium as gym
import numpy as np
from algorithms.default_genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from algorithms.species_genetic_algorithm.genetic_algorithm import SpeciesGeneticAlgorithm
from algorithms.MAPElites_genetic_algorithm.genetic_algorithm import MAPElitesAlgorithm
from controller import Controller
from algorithms.species_genetic_algorithm.species import Species


if __name__ == "__main__":
    ga = MAPElitesAlgorithm(
        experiment_name="test_map_elites",
        iterations=200,
        grid_size=20,
        initial_pop_size=50
    )
    winner = ga.run(min_mutations=1)
    ga.visualize_archive()
    
    muscle_indices = np.where((winner.body == 3) | (winner.body == 4))
    muscle_x = muscle_indices[1]
    controller = Controller(omega=5*np.pi, phase_coefficent=-2, amplitude=0.5, offset=1)
    env = gym.make('Walker-v0', body=winner.body, render_mode="human")
    env.reset()
    
    step = 0
    while True:
        t = step * 0.019
        action = controller.action_signal(t, muscle_x)
        res = env.step(action)
    
        done = res[2] if len(res) == 4 else (res[2] or res[3])
    
        env.render()
        step += 1

        if done:
            env.reset()
            step = 0
