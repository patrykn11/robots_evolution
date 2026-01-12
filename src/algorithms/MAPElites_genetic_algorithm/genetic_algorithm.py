import numpy as np
import random
import copy
import multiprocessing
from typing import List, Tuple, Dict
from structure import Structure
from algorithms.base_genetic_algorithm import BaseGeneticAlgorithm
from evogym import sample_robot
from evaluation import evaluate
import matplotlib.pyplot as plt 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
import os 
import zipfile
import json
import random


class MAPElitesAlgorithm(BaseGeneticAlgorithm):
    def __init__(
        self,
        experiment_name: str,
        robot_shape: Tuple[int, int] = (5, 5),
        voxel_types: List[int] = [0, 1, 2, 3, 4], 
        env_type: str = 'Walker-v0',
        generations=1000,           
        grid_size=20,              
        pop_size=100       
    ):
        super().__init__(
            experiment_name,
            pop_size=pop_size,
            generations=generations,
            robot_shape=robot_shape,
            voxel_types=voxel_types,
            env_type=env_type,
        )
        
        self.grid_size = grid_size
        self.archive: Dict[Tuple[int, int, int], Structure] = {}
        self.top_10_robots: List[Structure] = []
        self.top_robot = None
        self.non_change_score_counter: int = 0
        self.do_boost: bool = False
        
        
    def _get_robot_thumbnail(self, body_matrix):
        rows, cols = body_matrix.shape
        img = np.ones((rows, cols, 3)) 
        
        colors = {
            0: [1.0, 1.0, 1.0], 
            1: [0.0, 0.0, 0.0],  
            2: [0.5, 0.5, 0.5],  
            3: [1.0, 0.6, 0.0],  
            4: [0.0, 0.8, 1.0]   
        }
        
        for r in range(rows):
            for c in range(cols):
                val = body_matrix[r, c]
                if val in colors:
                    img[r, c] = colors[val]
                    
        return img
    
    def visualize_showcase(self, filename="showcase_map_PERCENT.png", num_samples=8):
        heatmap_data = np.full((self.grid_size, self.grid_size), np.nan)
        for (x, y, z), robot in self.archive.items():
            current_val = heatmap_data[y, x] 
            if np.isnan(current_val) or robot.fitness > current_val:
                heatmap_data[y, x] = robot.fitness
        
        if np.all(np.isnan(heatmap_data)):
            print("Archiwum puste.")
            return

        fig, ax = plt.subplots(figsize=(20, 20))
        
        ax.add_patch(patches.Rectangle((-0.5, -0.5), self.grid_size, self.grid_size, 
                                     facecolor='#e0e0e0', zorder=0))
        
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='white', lw=2, zorder=1)
            ax.axvline(i - 0.5, color='white', lw=2, zorder=1)

        im = ax.imshow(heatmap_data, cmap='jet', origin='lower', interpolation='nearest', 
                      aspect='equal', extent=[-0.5, self.grid_size-0.5, -0.5, self.grid_size-0.5], zorder=2)
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Fitness Score', fontsize=20, weight='bold')
        cbar.ax.tick_params(labelsize=16)
        
        ax.set_title(f'MAP-Elites Archive Showcase\n(Diversity: {len(self.archive)} solutions)', 
                     fontsize=26, pad=30, weight='bold')
        
        ax.axis('off')

        for p in range(0, 101, 10):
            t = (p / 100.0) * self.grid_size - 0.5 
            coord = (p / 100.0) * self.grid_size - 0.5

            ax.text(coord, -1.8, f"{p}%", ha='center', va='top', fontsize=16, weight='bold')
            
            ax.text(-1.8, coord, f"{p}%", ha='right', va='center', fontsize=16, weight='bold')

        ax.text(self.grid_size / 2.0 - 0.5, -4.5, "Descriptor X: Masa (Lekki -> Ciężki)", 
                ha='center', va='top', fontsize=24, weight='bold')
        
        ax.text(-4.5, self.grid_size / 2.0 - 0.5, "Descriptor Y: Mięśnie (Mało -> Dużo)", 
                ha='right', va='center', rotation=90, fontsize=24, weight='bold')

        rect = patches.Rectangle((-0.5, -0.5), self.grid_size, self.grid_size, 
                                 linewidth=5, edgecolor='black', facecolor='none', zorder=10)
        ax.add_patch(rect)

        all_robots = list(self.archive.items())
        all_robots.sort(key=lambda item: item[1].fitness, reverse=True)
        
        selected_robots = []
        occupied_spots = []
        
        for coords, robot in all_robots:
            if len(selected_robots) >= num_samples: break     
            cx, cy, cz = coords
            
            is_far_enough = True
            for (ox, oy, _) in occupied_spots:
                dist = np.sqrt((cx - ox)**2 + (cy - oy)**2)
                if dist < self.grid_size * 0.20: 
                    is_far_enough = False
                    break
            
            if is_far_enough:
                selected_robots.append({'coords': coords, 'robot': robot})
                occupied_spots.append(coords)

        center_x, center_y = (self.grid_size - 1) / 2.0, (self.grid_size - 1) / 2.0
        for item in selected_robots:
            gx, gy, _ = item['coords']
            item['angle'] = np.arctan2(gy - center_y, gx - center_x)

        selected_robots.sort(key=lambda x: x['angle'])

        min_angle_diff = 2 * np.pi / (num_samples + 1)
        for i in range(len(selected_robots) - 1):
            curr, nxt = selected_robots[i], selected_robots[i+1]
            diff = nxt['angle'] - curr['angle']
            if diff < min_angle_diff:
                nxt['angle'] += (min_angle_diff - diff) * 0.8

        display_radius = (self.grid_size / 2.0) * 1.6 

        for item in selected_robots:
            robot = item['robot']
            gx, gy, _ = item['coords']
            angle = item['angle']
            
            img_arr = self._get_robot_thumbnail(robot.body)
            
            imagebox = OffsetImage(img_arr, zoom=22.0, cmap='gray', interpolation='nearest') 
            imagebox.image.axes = ax

            box_x = center_x + display_radius * np.cos(angle)
            box_y = center_y + display_radius * np.sin(angle)

            ab = AnnotationBbox(
                imagebox,
                (gx, gy),
                xybox=(box_x, box_y),
                xycoords='data',
                boxcoords='data',
                pad=0.3,
                arrowprops=dict(arrowstyle="->", color="black", lw=2.5, connectionstyle="arc3,rad=0.2"),
                bboxprops=dict(edgecolor='black', linewidth=3, facecolor='white', alpha=1.0)
            )
            ax.add_artist(ab)

        margin = 12 
        ax.set_xlim(-margin, self.grid_size + margin - 1)
        ax.set_ylim(-margin, self.grid_size + margin - 1)

        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        
    def get_descriptors(self, structure: Structure) -> Tuple[int, int, int]:
        body = structure.body
        
        mass = np.count_nonzero(body)
        max_possible_mass = self.robot_shape[0] * self.robot_shape[1]     
        x_idx = int((mass / max_possible_mass) * (self.grid_size - 1))
        
        soft_voxels = np.count_nonzero((body == 2) | (body == 3) | (body == 4))
        if mass > 0:
            muscle_ratio = soft_voxels / mass
        else:
            muscle_ratio = 0
        y_idx = int(muscle_ratio * (self.grid_size - 1))
        
        rows, cols = np.nonzero(body)
        if len(rows) > 0:
            height = (np.max(rows) - np.min(rows)) + 1
            width = (np.max(cols) - np.min(cols)) + 1
            ratio = width / height
        else:
            ratio = 1.0

        clamped_ratio = np.clip(ratio, 0.2, 3.0)
        normalized_ratio = (clamped_ratio - 0.2) / (3.0 - 0.2) 
        z_idx = int(normalized_ratio * (self.grid_size - 1))

        return (x_idx, y_idx, z_idx)

    def _random_selection(self, keys: list[Tuple]) -> Structure:
        random_key = random.choice(keys)
        return self.archive[random_key]

    def _tournament_selection(self, keys: list[Tuple], tournament_size: int = 5) -> Structure:
        candidate_keys = random.sample(keys, min(len(keys), tournament_size))
        best_key = max(candidate_keys,  key = lambda k: self.archive[k].fitness)
        return self.archive[best_key]
    
    def _exponential_selection(self, keys: list[Tuple]) -> Structure:
        candidtaes = [self.archive[k] for k in keys]
        fitnesses = np.array([rob.fitness for rob in candidtaes])
        weights = np.exp(fitnesses- np.max(fitnesses))
        
        return random.choices(candidtaes, weights=weights, k=1)[0]
    
    def warm_up(self, n_samples=1000):
        random_population = []
        
        print("Warming up...")
        for _ in range(n_samples):
            body, _ = sample_robot((self.robot_shape[0], self.robot_shape[1]))
            structure = Structure(body)
            if structure.is_valid():
                random_population.append(structure)
        
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            inputs = zip(random_population, [self.env_type] * len(random_population))
            fitness_scores = pool.starmap(evaluate, inputs)
            
        for ind, fit in zip(random_population, fitness_scores):
            ind.fitness = fit
            self._add_to_archive(ind)
        print("Warming up ended:)")
            
    def _mutation_1(self, parent: Structure, min_mutations: int = 1, bonus_chance: float = 0.2) -> Structure:
        for _ in range(20): 
            new_body = parent.body.copy()
            mutations_left = min_mutations
            
            while mutations_left > 0:
                r, c = np.random.randint(0, self.robot_shape[0]), np.random.randint(0, self.robot_shape[1])
                mutation_happened = False
                
                if new_body[r, c] != 0: 
                    if random.random() < 0.5:
                        current = new_body[r, c]
                        new_body[r, c] = random.choice([2, 3, 4]) if current == 1 else 1
                    else:
                        if random.random() < 0.4: new_body[r, c] = 0
                    mutation_happened = True
                else:
                    if random.random() < 0.5:
                        new_body[r, c] = 1 if random.random() < 0.5 else random.choice([2, 3, 4])
                        mutation_happened = True
                
                if mutation_happened:
                    if random.random() > bonus_chance:
                        mutations_left -= 1
            
            child = Structure(new_body)
            if child.is_valid():
                return child
        return parent 
    
    """Cascade mutation with starting chance"""
    def _mutation_2(self, parent: Structure, starting_chance: float) -> Structure:
        chance = starting_chance
        for _ in range(20):
            new_body = parent.body.copy()
            while (random.random() < chance):
                r, c = np.random.randint(0, self.robot_shape[0]), np.random.randint(0, self.robot_shape[1])
                
                if new_body[r, c] == 0: new_body[r, c] = random.choice([2, 3, 4])
                else: new_body[r, c] = 0
                chance /= 2
            
            child = Structure(new_body)
            if child.is_valid():
                return child
        return parent
    
    """Simple mutation with provided chance"""
    def _mutation_3(self, parent: Structure, mutation_chance: float) -> Structure:
        for _ in range(20):
            new_body =  parent.body.copy()
            if random.random() < mutation_chance:
                r, c = np.random.randint(0, self.robot_shape[0]), np.random.randint(0, self.robot_shape[1])
                current = new_body[r, c]
                if current == 0: new_body[r, c] = random.choice([2, 3, 4]) if current == 1 else 1
                else: new_body[r, c] = 0
            
            child = Structure(new_body)
            if child.is_valid(): return child
        return parent        
    
    """Mutation with boost"""
    def _mutation_4(self, parent: Structure, boost_mutations: int = 5) -> Structure:
        for _ in range(20):
            new_body = parent.body.copy()
            if not self.do_boost: boost_mutations = 1
            for _ in range(boost_mutations):
                r, c = np.random.randint(0, self.robot_shape[0]), np.random.randint(0, self.robot_shape[1])
                current = new_body[r, c]
                if current == 0: new_body[r, c] = random.choice([2, 3, 4]) if current == 1 else 1
                else: new_body[r, c] = 0
            
            child = Structure(new_body)
            if child.is_valid(): return child
        return parent     
    
    def _mutation_5(self, parent: Structure) -> Structure:
        for _ in range(20):
            new_body = parent.body.copy()
            r1 = np.random.randint(0, self.robot_shape[0])
            r2 = np.random.randint(0, self.robot_shape[0])
            c1 = np.random.randint(0, self.robot_shape[1])
            c2 = np.random.randint(0, self.robot_shape[1])

            r_start, r_end = sorted((r1, r2))
            c_start, c_end = sorted((c1, c2))

            if r_start == r_end: r_end += 1
            if c_start == c_end: c_end += 1
            
            r_end = min(r_end, self.robot_shape[0])
            c_end = min(c_end, self.robot_shape[1])

            subgrid = new_body[r_start:r_end, c_start:c_end]
            flat_subgrid = subgrid.flatten()
            np.random.shuffle(flat_subgrid)
            new_body[r_start:r_end, c_start:c_end] = flat_subgrid.reshape(subgrid.shape)
            
            child = Structure(new_body)
            if child.is_valid(): return child
            
        return parent
              
    def mutate(self, parent: Structure, strategy: int, *args, **kwargs) -> Structure:
        if strategy == 1:
            mutant = self._mutation_1(parent, *args, **kwargs)
        elif strategy == 2:
            mutant = self._mutation_2(parent, *args, **kwargs)
        elif strategy == 3:
            mutant = self._mutation_3(parent, *args, **kwargs)
        elif strategy == 4:
            mutant = self._mutation_4(parent, *args, **kwargs)
        elif strategy == 5:
            mutant = self._mutation_5(parent, *args, **kwargs)
        else:
            raise ValueError("Incorrect mutation strategy provided")
            
        return mutant
    def run(self, mutation_strategy: int = 1, selection_strategy: str = "tournament", n_warm_up: int = 1000,  *args, **kwargs):
        history = {"best_fitness": [], "avg_fitness": [], "archive_size": []}
        
        self.warm_up(n_warm_up)
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            inputs = zip(self.population, [self.env_type] * len(self.population))
            fitness_scores = pool.starmap(evaluate, inputs)
            
            for ind, fit in zip(self.population, fitness_scores):
                ind.fitness = fit
                self._add_to_archive(ind)

        batch_size = self.num_workers * 4 
        current_iter = 0
        
        best_score = 0
        while current_iter < self.generations:
            keys = list(self.archive.keys())
            if not keys: break 
            
            offspring = []
            for _ in range(batch_size):
                parent = None
                child = None
                        
                if selection_strategy == "random_search":
                    parent = self._random_selection(keys)
                elif selection_strategy == "tournament":
                    parent = self._tournament_selection(keys, tournament_size=5)
                elif selection_strategy == "aggressive_bonus":
                    parent = self._exponential_selection(keys)
                else:
                    parent = self._random_selection(keys)
                child = self.mutate(parent, mutation_strategy, *args,  **kwargs)
                offspring.append(child)
                        
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                inputs = zip(offspring, [self.env_type] * len(offspring))
                fitness_scores = pool.starmap(evaluate, inputs)
            
            for child, fit in zip(offspring, fitness_scores):
                child.fitness = fit
                self._add_to_archive(child)
            
            current_iter += 1
            
            best_glob = self.top_10_robots[0].fitness if self.top_10_robots else 0
            if best_glob > best_score: 
                best_score = best_glob
                self.non_change_score_counter = 0
                self.do_boost = False
            elif round(best_glob, 2) == round(best_score, 2): 
                self.non_change_score_counter += 1
            
            if self.non_change_score_counter >= 2: self.do_boost =  True
            
            avg_fit = np.mean([r.fitness for r in self.archive.values()]) if self.archive else 0
            
            history["best_fitness"].append(float(best_glob))
            history["avg_fitness"].append(float(avg_fit))
            history["archive_size"].append(len(self.archive))
            
            print(f"Iter {current_iter} | Archive Size: {len(self.archive)} | Best: {best_glob:.4f} | Avg: {avg_fit:.4f} " +
                  f'Boost: {self.do_boost}')

            self.save_history(history)
            self.save_robot(current_iter, self.top_10_robots[0] if self.top_10_robots else None)
        
        experiment_dir = os.path.join("results", self.experiment_name)
        history_dir = os.path.join(experiment_dir, "history_snapshots")
        
        json_path = os.path.join(experiment_dir, "history.json")
        serializable_history = {
            k: [float(v) for v in vals] if isinstance(vals, list) else vals
            for k, vals in history.items()
        }
        with open(json_path, "w") as f:
            json.dump(serializable_history, f, indent=4)

        zip_path = os.path.join(experiment_dir, "evolution_data.zip")
        if os.path.exists(history_dir):
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(history_dir):
                    for file in files:
                        if file.endswith(".pkl"):
                            file_path = os.path.join(root, file)
                            zipf.write(file_path, arcname=file)
        
        self.zip_results()
        return self.top_10_robots[0] if self.top_10_robots else None
    
    def visualize_archive(self, filename="fitness_heatmap.png"):
        heatmap_data = np.full((self.grid_size, self.grid_size), np.nan)

        for (x, y, z), robot in self.archive.items():
            current_val = heatmap_data[x, y]
            if np.isnan(current_val) or robot.fitness > current_val:
                heatmap_data[x, y] = robot.fitness

        plt.figure(figsize=(10, 8))
        
        plt.imshow(heatmap_data.T, cmap='viridis', origin='lower', interpolation='nearest')
        
        cbar = plt.colorbar()
        cbar.set_label('Fitness Score')
        
        plt.xlabel('Descriptor X: Masa (0 = lekki, Max = ciężki)')
        plt.ylabel('Descriptor Y: Mięśnie (0 = mało, Max = dużo)')
        plt.title(f'MAP-Elites Archive Heatmap\nRobots found: {len(self.archive)}')

        plt.savefig(filename)
        
        plt.show()
    
    def visualize_advanced(self, filename="heatmap_advanced.png", slices=4):
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        axes = axes.flatten()
        z_step = self.grid_size // slices
        
        all_fitnesses = [r.fitness for r in self.archive.values()]
        if not all_fitnesses:
            return
        vmin, vmax = min(all_fitnesses), max(all_fitnesses)

        for i in range(slices):
            ax = axes[i]
            z_start = i * z_step
            z_end = (i + 1) * z_step
            
            ax.set_title(f"Z-Slice {i+1}: Aspect Ratio (Index {z_start}-{z_end-1})")
            slice_data = np.full((self.grid_size, self.grid_size), np.nan)
            robots_in_slice = []
            for (x, y, z), robot in self.archive.items():
                if z_start <= z < z_end:
                    slice_data[y, x] = robot.fitness
                    robots_in_slice.append(((x, y), robot))

            im = ax.imshow(slice_data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
            
            ax.set_xlabel('X: Masa (Lekki -> Ciężki)')
            ax.set_ylabel('Y: Mięśnie (Mało -> Dużo)')
            
            for (gx, gy), robot in robots_in_slice:
                img_arr = self._get_robot_thumbnail(robot.body)
                imagebox = OffsetImage(img_arr, zoom=0.6, cmap='gray') 
                imagebox.image.axes = ax
                
                ab = AnnotationBbox(
                    imagebox, 
                    (gx, gy),
                    frameon=False,
                    pad=0
                )
                ax.add_artist(ab)

        fig.colorbar(im, ax=axes.ravel().tolist(), label='Fitness')
        plt.suptitle(f'MAP-Elites 3D Archive Breakdown\nTotal Robots: {len(self.archive)}', fontsize=16)
        
        plt.savefig(filename)
        plt.close()
    
    def visualize_max_projection(self, filename="max_projection_heatmap.png", title_suffix=""):
        projection_map = np.full((self.grid_size, self.grid_size), np.nan)

        for (x, y, z), robot in self.archive.items():
            current_val = projection_map[y, x] 
            
            if np.isnan(current_val) or robot.fitness > current_val:
                projection_map[y, x] = robot.fitness

        plt.figure(figsize=(12, 10))
        mask = np.isnan(projection_map)
        
        ax = sns.heatmap(
            projection_map, 
            mask=mask,               
            annot=True,             
            fmt=".1f",
            cmap="viridis",          
            linewidths=.5,           
            cbar_kws={'label': 'Max Fitness Score'},
            square=True              
        )

        ax.invert_yaxis()
        
        plt.xlabel('Descriptor X: Masa (Index)')
        plt.ylabel('Descriptor Y: Mięśnie (Index)')
        plt.title(f'MAP-Elites Max-Projection (2D)\nArchives Size: {len(self.archive)} {title_suffix}')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close() 

    def _add_to_archive(self, robot: Structure):
        coords = self.get_descriptors(robot)
        
        if coords not in self.archive:
            self.archive[coords] = copy.deepcopy(robot)
        else:
            current_resident = self.archive[coords]
            if robot.fitness > current_resident.fitness:
                self.archive[coords] = copy.deepcopy(robot)
        
        self._update_top_10(robot)

    def _update_top_10(self, robot: Structure):
        self.top_10_robots.append(copy.deepcopy(robot))
        self.top_10_robots.sort(key=lambda x: x.fitness, reverse=True)
        self.top_10_robots = self.top_10_robots[:10]
