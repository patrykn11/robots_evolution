import pickle
import gymnasium as gym
import evogym.envs
import numpy as np
import imageio
import os
import sys
import argparse
import glob
from PIL import Image, ImageDraw, ImageFont

from controller import Controller
from structure import Structure 


def add_text_to_frame(frame, text):
    """
    Adds text to the top-left corner of the frame.
    """
    pil_im = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_im)
    font = ImageFont.load_default()
    draw.rectangle([5, 5, 100, 25], fill="white", outline="black")
    draw.text((10, 8), text, fill="black", font=font)
    return np.array(pil_im)


def create_evolution_gif(experiment_dir, env_name='Walker-v0', steps_per_gen=100, fps=30, output_filename='evolution_timelapse.gif'):
    """
    Creates a GIF showing the evolution of structures over generations.
    
    :param experiment_dir: Name of the experiment directory containing generation files.
    :param env_name: Name of the Evogym environment to use.
    :param steps_per_gen: Number of simulation steps to run per generation.
    :param fps: Frames per second for the output GIF.
    """
    search_path = os.path.join(experiment_dir, "gen_*.pkl")
    files = glob.glob(search_path)

    files.sort()
    
    if not files:
        print(f"Files not found in: {search_path}. Run main.py first!")
        return

    print(f"Found {len(files)} generations. Starting rendering...")
    
    frames = []

    controller = Controller(omega=5*np.pi, phase_coefficent=-2, amplitude=0.5, offset=1.0)

    for i, file_path in enumerate(files):
        gen_num = i + 1
        print(f"Processing Generation {gen_num}/{len(files)}: {os.path.basename(file_path)}")
        
        try:
            with open(file_path, "rb") as f:
                structure = pickle.load(f)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        try:
            env = gym.make(env_name, body=structure.body, render_mode='rgb_array')
            env.reset()
        except Exception as e:
            print(f"Error creating environment for gen {gen_num}: {e}")
            continue

        muscle_indices = np.where((structure.body == 3) | (structure.body == 4))
        muscle_x = muscle_indices[1]

        for step in range(steps_per_gen):
            t = step * 0.019
            action = controller.action_signal(t, muscle_x)
            
            res = env.step(action)
            done = res[2] if len(res) == 4 else (res[2] or res[3])
            
            frame = env.render()
            
            frame_with_text = add_text_to_frame(frame, f"Gen: {gen_num}")
            frames.append(frame_with_text)

            if done:
                for _ in range(steps_per_gen - step):
                    frames.append(frame_with_text)
                break
        
        env.close()

    output_path = os.path.join(experiment_dir, output_filename)
    print(f"Saving GIF ({len(frames)} frames)... This may take a while.")
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Done! Your timelapse is at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create evolution GIF from experiment data.")
    
    parser.add_argument("experiment_dir", type=str, help="Path to the experiment directory.")
    
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output GIF (default: 30)")
    parser.add_argument("--steps-per-gen", type=int, default=100, help="Number of simulation steps per generation (default: 100)")
    parser.add_argument("--output-filename", type=str, default="evolution_timelapse.gif", help="Name of the output GIF file (default: evolution_timelapse.gif)")
    
    args = parser.parse_args()
    
    create_evolution_gif(
        experiment_dir=args.experiment_dir,
        steps_per_gen=args.steps_per_gen,
        fps=args.fps,
        output_filename=args.output_filename
    )


if __name__ == "__main__":
    main()
