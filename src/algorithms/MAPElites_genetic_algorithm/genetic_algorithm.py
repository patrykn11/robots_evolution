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

    def crossover(self, parent1: Structure, parent2: Structure) -> Structure:
        for _ in range(20): 
            child_body = np.zeros(self.robot_shape, dtype=int)
            
            axis = random.choice([0, 1])
            split_idx = random.randint(1, self.robot_shape[axis] - 1)
            
            if axis == 0:
                child_body[:split_idx, :] = parent1.body[:split_idx, :]
                child_body[split_idx:, :] = parent2.body[split_idx:, :]
            else:
                child_body[:, :split_idx] = parent1.body[:, :split_idx]
                child_body[:, split_idx:] = parent2.body[:, split_idx:]
            
            child = Structure(child_body)
            if child.is_valid():
                return child
        
        return copy.deepcopy(parent1)

    def mutate(self, parent: Structure, min_mutations: int = 1, bonus_chance: float = 0.2) -> Structure:
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

    def run(self, strategy: str = "tournament", min_mutations: int = 1):
        history = {"best_fitness": [], "avg_fitness": [], "archive_size": []}
        
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            inputs = zip(self.population, [self.env_type] * len(self.population))
            fitness_scores = pool.starmap(evaluate, inputs)
            
            for ind, fit in zip(self.population, fitness_scores):
                ind.fitness = fit
                self._add_to_archive(ind)

        batch_size = self.num_workers * 4 
        current_iter = 0
        
        while current_iter < self.generations:
            keys = list(self.archive.keys())
            if not keys: break 
            
            offspring = []
            for _ in range(batch_size):
                parent = None
                child = None
                        
                match strategy:
                    case "random_search":
                        parent = self._random_selection(keys)
                        child = self.mutate(parent, min_mutations, bonus_chance=0.0)

                    case "tournament":
                        parent = self._tournament_selection(keys, tournament_size=5)
                        child = self.mutate(parent, min_mutations, bonus_chance=0.1)

                    case "aggressive_bonus":
                        parent = self._exponential_selection(keys)
                        child = self.mutate(parent, min_mutations, bonus_chance=0.2)

                    case _:
                        parent = self._select_random(keys)
                        child = self.mutate(parent)

                offspring.append(child)
                        
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                inputs = zip(offspring, [self.env_type] * len(offspring))
                fitness_scores = pool.starmap(evaluate, inputs)
            
            for child, fit in zip(offspring, fitness_scores):
                child.fitness = fit
                self._add_to_archive(child)
            
            current_iter += 1
            
            best_glob = self.top_10_robots[0].fitness if self.top_10_robots else 0
            avg_fit = np.mean([r.fitness for r in self.archive.values()]) if self.archive else 0
            
            history["best_fitness"].append(float(best_glob))
            history["avg_fitness"].append(float(avg_fit))
            history["archive_size"].append(len(self.archive))
            
            print(f"Iter {current_iter} | Archive Size: {len(self.archive)} | Best: {best_glob:.4f} | Avg: {avg_fit:.4f}")

            self.save_history(history)
            self.save_robot(current_iter, self.top_10_robots[0] if self.top_10_robots else None)
        
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
        print(f"Heatmapa zapisana jako {filename}")
        
        plt.show()

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
