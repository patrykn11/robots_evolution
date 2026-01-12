import os
import zipfile
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import math
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))

from structure import Structure


def visualize_species(zip_path: str, output_path: str):
    print(f"Processing {zip_path}...")
    
    species_data = defaultdict(list)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = sorted(zip_ref.namelist())
            
            pkl_files = [f for f in file_list if f.endswith('.pkl') and not f.startswith('__MACOSX') and not os.path.basename(f).startswith('._')]
            
            print(f"Found {len(pkl_files)} valid robot files.")
            
            pattern = re.compile(r"gen_(\d+)_species_(\d+)\.pkl")
            
            for file_name in pkl_files:
                match = pattern.search(file_name)
                if match:
                    gen = int(match.group(1))
                    species_id = int(match.group(2))
                    species_data[species_id].append((gen, file_name))
            
            print(f"Found {len(species_data)} species.")
            
            if not species_data:
                print("No matching files found. Check filename format.")
                return

            limit_species = 8
            sorted_species_ids = sorted(species_data.keys())
            
            if len(sorted_species_ids) > limit_species:
                sorted_species_ids = sorted_species_ids[:limit_species]
                print(f"Limiting to first {limit_species} species.")
            
            num_species = len(sorted_species_ids)
            limit_generations = 4
            print(f"Limiting to {limit_generations} generations (rows).")
            
            fig, axes = plt.subplots(limit_generations, num_species, figsize=(3 * num_species, 3 * limit_generations))
            fig.suptitle("Species Evolution Comparison (Rows: Generations, Cols: Species)", fontsize=16)

            if num_species == 1:
                axes = axes.reshape(limit_generations, 1)
            elif limit_generations == 1:
                axes = axes.reshape(1, num_species)
            
            temp_dir = "temp_renders"
            os.makedirs(temp_dir, exist_ok=True)
            
            for col_idx, species_id in enumerate(sorted_species_ids):
                robots = sorted(species_data[species_id], key=lambda x: x[0])
                
                if len(robots) <= limit_generations:
                    selected_robots = robots
                else:
                    indices = np.linspace(0, len(robots) - 1, limit_generations, dtype=int)
                    selected_robots = [robots[i] for i in indices]
                
                for row_idx in range(limit_generations):
                    ax = axes[row_idx, col_idx]
                    
                    if row_idx < len(selected_robots):
                        gen, file_name = selected_robots[row_idx]

                        try:
                            with zip_ref.open(file_name) as file:
                                robot = pickle.load(file)
                            
                            render_path = os.path.join(temp_dir, f"s{species_id}_g{gen}.png")
                            
                            if hasattr(robot, 'render_robot_and_save'):
                                robot.render_robot_and_save(render_path)
                            else:
                                print(f"Robot object missing render method: {type(robot)}")
                                ax.text(0.5, 0.5, "No Render", ha='center')
                                continue
        
                            if os.path.exists(render_path):
                                img = Image.open(render_path)
                                ax.imshow(img)
                            else:
                                ax.text(0.5, 0.5, "Render Failed", ha='center')
                                
                        except Exception as e:
                            print(f"Error processing {file_name}: {e}")
                            ax.text(0.5, 0.5, "Error", ha='center')
                    
                    ax.axis('off')
                    
                    # Set labels
                    if row_idx == 0:
                        ax.set_title(f"Species {species_id}")
                    if col_idx == 0:
                        if row_idx < len(selected_robots):
                             gen = selected_robots[row_idx][0]
                             ax.text(-0.1, 0.5, f"Gen {gen}", transform=ax.transAxes, 
                                     va='center', ha='right', rotation=0, fontsize=10)

            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
            plt.close()
            
            import shutil
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    zip_file = r"c:\Marcin\Golem\robots_evolution\results\species_experiment\species_zip\species.zip"
    output_file = r"c:\Marcin\Golem\robots_evolution\results\species_experiment\species_comparison.png"
    visualize_species(zip_file, output_file)
