import gymnasium as gym
import numpy as np
import os
from algorithms.MAPElites_genetic_algorithm.genetic_algorithm import MAPElitesAlgorithm
from controller import Controller
from experiment_visualizer import ExperimentVisualizer 

if __name__ == "__main__":
    experiment_name = "test_map_elites"

    ga = MAPElitesAlgorithm(
        experiment_name=experiment_name,
        generations=30,   
        grid_size=20,
        pop_size=50
    )

    winner = ga.run(selection_strategy='tournament', mutation_strategy=1, n_warm_up=100) 

    visualizer = ExperimentVisualizer()
    exp_path = os.path.join("results", experiment_name)
    
    visualizer.visualize(
        experiment_data_path=exp_path,
        plot_fitness_curves=True,
        generate_robot_images=True,
        generate_full_gif=True,    
        generate_full_video=True,  
        image_every_n_generations=2, 
        env_type='Walker-v0'
    )

    if winner is not None:
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