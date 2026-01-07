import json
import os
from collections import defaultdict
import zipfile
import pickle
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from make_gif import create_evolution_gif

from controller import Controller
from structure import Structure


class ExperimentVisualizer:
    
    def __init__(self):
        self.data_path = None
        self.env_type = None
        self.history = None
        self.robots = None

    def _load_data(self):
        """
        Loads data from experiment directory, including history and robot structures.
        """
        self.history = {}  # Key - experiment name, value - history dict
        self.robots = defaultdict(list) # Key - experiment name, value - list of robot structures

        for path in self.data_path:
            exp_name = os.path.basename(path.rstrip(os.sep))
            with open(os.path.join(path, "history.json"), "r") as f:
                self.history[exp_name] = json.load(f)
            
            zip_path = os.path.join(path, "evolution_data.zip")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = sorted(zip_ref.namelist())
                    for file_name in file_list:
                        if file_name.endswith('.pkl'):
                            with zip_ref.open(file_name) as file:
                                robot = pickle.load(file)
                                self.robots[exp_name].append(robot)
            except Exception as e:
                print(f"Error opening zip file for {exp_name}: {e}")
                continue

    def _plot_fitness_curves(self):
        """
        Plots fitness curves for all experiments loaded.
        """
        print("Plotting fitness curves...")
        plt.figure(figsize=(10, 6))
        plot_name = "Fitness Curves-" + "-".join(self.history.keys())
        plt.title(plot_name)
        
        for exp_name, history in self.history.items():
            best_fitness = history.get("best_fitness", [])
            avg_fitness = history.get("avg_fitness", [])
            generations = range(1, len(best_fitness) + 1)
            
            plt.plot(generations, best_fitness, label=f'{exp_name} - Best Fitness')
            plt.plot(generations, avg_fitness, label=f'{exp_name} - Average Fitness', linestyle='--')

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        
        if len(self.data_path) == 1:
            save_path = os.path.join(self.data_path[0], "fitness_curves.png")
        else:
            save_path = f"results/comparisons/{plot_name}.png"
            os.makedirs("results/comparisons", exist_ok=True)
            
        plt.savefig(save_path)
        plt.close()

        print(f"Fitness curves saved to {save_path}")

    def _generate_robot_images(self, every_n_generations: int):
        """
        Generates and saves images of robots at specified generation intervals.
        """
        print("Generating robot images...")
        for path in self.data_path:
            exp_name = os.path.basename(path.rstrip(os.sep))
            robots = self.robots.get(exp_name)
            images_dir = os.path.join(path, "robot_images")
            os.makedirs(images_dir, exist_ok=True)
            
            for i, robot in enumerate(robots):
                if i % every_n_generations != 0:
                    continue
                image_path = os.path.join(images_dir, f"gen_{i+1:03d}.png")
                robot.render_robot_and_save(image_path, env_type=self.env_type)
        print("Robot images generation completed.")
    
    def _generate_full_gif(self):
        """
        Generates full evolution GIFs for each experiment.
        """
        print("Generating full GIFs...")
        for path in self.data_path:
            exp_name = os.path.basename(path.rstrip(os.sep))
            gif_filename = f"{exp_name}_evolution.gif"
            create_evolution_gif(
                experiment_dir=path,
                env_name=self.env_type,
                steps_per_gen=50,
                fps=30,
                output_filename=gif_filename
            )
        print("GIF generation completed.")

    def generate_best_robot_video(self, experiment_path: str, output_filename: str = "best_robot.gif", duration_steps: int = 500, fps: int = 30):
        """
        Generates a video of the best (last) robot from an experiment.
        
        :param experiment_path: Path to the experiment directory.
        :param output_filename: Name of the output video file.
        :param duration_steps: Number of simulation steps to run the robot.
        :param fps: Frames per second for the output video.
        """
        import gymnasium as gym
        import evogym.envs
        import imageio
        
        print(f"Generating best robot video for {experiment_path}...")
        
        exp_name = os.path.basename(experiment_path.rstrip(os.sep))
        robots = self.robots.get(exp_name)
        
        if not robots:
            print(f"No robots found for experiment {exp_name}")
            return
        
        best_robot = robots[-1]
        print(f"Best robot fitness: {best_robot.fitness}")

        try:
            env = gym.make(self.env_type, body=best_robot.body, render_mode='rgb_array')
            env.reset()
        except Exception as e:
            print(f"Error creating environment: {e}")
            return
        
        muscle_indices = np.where((best_robot.body == 3) | (best_robot.body == 4))
        muscle_x = muscle_indices[1]
        
        controller = Controller(omega=5*np.pi, phase_coefficent=-2, amplitude=0.5, offset=1.0)
        
        frames = []
        print(f"Recording {duration_steps} steps...")
        
        for step in range(duration_steps):
            t = step * 0.019
            action = controller.action_signal(t, muscle_x)
            
            res = env.step(action)
            done = res[2] if len(res) == 4 else (res[2] or res[3])
            
            frame = env.render()
            frames.append(frame)
            
            if step % 50 == 0:
                print(f"  Progress: {step}/{duration_steps} steps")
            
            if done:
                print(f"Robot reached terminal state at step {step}")
                for _ in range(duration_steps - step - 1):
                    frames.append(frame)
                break
        
        env.close()
        
        output_path = os.path.join(experiment_path, output_filename)
        print(f"Saving video ({len(frames)} frames) to {output_path}...")
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Done! Best robot video saved to: {output_path}")

    def visualize(
            self,
            experiment_data_path,
            plot_fitness_curves: bool = False,
            generate_robot_images: bool = False,
            generate_full_gif: bool = False,
            generate_full_video: bool = False,
            image_every_n_generations: int = 10,
            env_type: str = 'Walker-v0',
    ):
        """
        Main method to visualize experiment results.
        
        :param experiment_data_path: Path or list of paths to experiment directories.
        :param plot_fitness_curves: Whether to plot fitness curves.
        :param generate_robot_images: Whether to generate robot images.
        :param generate_full_gif: Whether to generate full evolution GIF.
        :param image_every_n_generations: Interval of generations to save robot images.
        :param env_type: Environment type for rendering robots.
        """

        # This ensures that user can provide both a single path or a list of paths (for comparing multiple experiments)
        if isinstance(experiment_data_path, str):
            self.data_path = [experiment_data_path]
        else:
            self.data_path = experiment_data_path

        self.env_type = env_type
        self._load_data()
        
        if plot_fitness_curves:
            self._plot_fitness_curves()

        if generate_robot_images:
            self._generate_robot_images(image_every_n_generations)

        if generate_full_gif:
            self._generate_full_gif()
        
        if generate_full_video:
            for path in self.data_path:
                self.generate_best_robot_video(
                    experiment_path=path,
                    output_filename="best_robot_video.gif",
                    duration_steps=500,
                    fps=30
                )

if __name__ == "__main__":
    visualizer = ExperimentVisualizer()
    
    # visualizer.visualize(
    #     experiment_data_path=["results/genetic_experiment", "results/species_experiment"],
    #     plot_fitness_curves=True,
    #     generate_robot_images=True,
    #     generate_full_gif=False,
    #     env_type='Walker-v0'
    # )
    # visualizer._load_data()
    # visualizer.generate_best_robot_video(
    #     experiment_path="results/species_experiment",
    #     output_filename="best_robot_video.gif",
    #     duration_steps=500,
    #     fps=30
    # )
    visualizer.visualize(
        experiment_data_path="results/species_experiment",
        generate_full_video=True,
        env_type='Walker-v0'
    )
