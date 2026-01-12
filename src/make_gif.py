import pickle
import gymnasium as gym
import evogym.envs
import numpy as np
import imageio
import os
import argparse
import glob
import zipfile
from PIL import Image, ImageDraw, ImageFont

#! FYI Imports for pickle
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
    
    :param experiment_dir: Name of the experiment directory containing generation files or path to a zip file.
    :param env_name: Name of the Evogym environment to use.
    :param steps_per_gen: Number of simulation steps to run per generation.
    :param fps: Frames per second for the output GIF.
    """
    
    files = []
    is_zip = False
    zip_ref = None
    
    if os.path.isfile(experiment_dir) and experiment_dir.lower().endswith('.zip'):
        is_zip = True
        try:
            zip_ref = zipfile.ZipFile(experiment_dir, 'r')
        except Exception as e:
            print(f"Error opening zip file: {e}")
            return

    elif os.path.isdir(experiment_dir):
        search_path = os.path.join(experiment_dir, "gen_*.pkl")
        files = glob.glob(search_path)
        
        if not files:
            possible_zip = os.path.join(experiment_dir, "evolution_data.zip")
            if os.path.isfile(possible_zip):
                print(f"No .pkl files found, but found zip archive: {possible_zip}")
                is_zip = True
                experiment_dir = possible_zip 
                try:
                    zip_ref = zipfile.ZipFile(possible_zip, 'r')
                except Exception as e:
                    print(f"Error opening zip file: {e}")
                    return
    else:
        print(f"Error: {experiment_dir} is not a valid directory or zip file.")
        return

    if is_zip and zip_ref:
        all_files = zip_ref.namelist()
        files = [f for f in all_files if os.path.basename(f).startswith('gen_') and f.endswith('.pkl')]

    files.sort()
    
    if not files:
        msg = f"Files not found in: {experiment_dir}"
        if is_zip:
             msg += " (checked inside zip archive)"
        print(msg)
        if is_zip and zip_ref:
            zip_ref.close()
        return

    print(f"Found {len(files)} generations.")
    if is_zip:
        output_dir = os.path.dirname(experiment_dir)
    else:
        output_dir = experiment_dir
    
    output_path = os.path.join(output_dir, output_filename)
    print(f"Starting rendering to {output_path}...")

    controller = Controller(omega=5*np.pi, phase_coefficent=-2, amplitude=0.5, offset=1.0)

    try:
        with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
            for i, file_path_or_name in enumerate(files):
                gen_num = i + 1
                print(f"Processing Generation {gen_num}/{len(files)}: {os.path.basename(file_path_or_name)}")
                
                try:
                    if is_zip:
                        with zip_ref.open(file_path_or_name) as f:
                            structure = pickle.load(f)
                    else:
                        with open(file_path_or_name, "rb") as f:
                            structure = pickle.load(f)
                except Exception as e:
                    print(f"Error reading file {file_path_or_name}: {e}")
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
                    writer.append_data(frame_with_text)

                    if done:
                        for _ in range(steps_per_gen - step):
                            writer.append_data(frame_with_text)
                        break
                
                env.close()

    except Exception as e:
        print(f"Error during GIF generation: {e}")
    finally:
        if is_zip and zip_ref:
            zip_ref.close()

    print(f"Done! Your timelapse is at: {output_path}")


def main():
    """
    Example usage: python make_gif.py path/to/experiment_dir_or_zip --fps 30 --steps-per-gen 100 --output-filename evolution_timelapse.gif
    """
    parser = argparse.ArgumentParser(description="Create evolution GIF from experiment data (directory or zip file).")
    
    parser.add_argument("experiment_dir", type=str, help="Path to the experiment directory or zip file.")
    
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