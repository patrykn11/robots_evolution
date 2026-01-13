import numpy as np
from functools import partial
import multiprocessing
import os
import pickle
import json
import copy
from typing import List, Tuple, Any
from evogym import sample_robot
from structure import Structure
from evaluation import evaluate
from algorithms.saving_mixin import SavingMixin

class BaseGeneticAlgorithm(SavingMixin):
    def __init__(
        self,
        experiment_name: str,
        pop_size: int,
        generations: int,
        robot_shape: Tuple[int, int],
        voxel_types: List[int],
        env_type: str = 'Walker-v0',
    ) -> None:
        self.experiment_name = experiment_name
        self.save_path = os.path.join("results", self.experiment_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.pop_size = pop_size
        self.generations = generations
        self.robot_shape = robot_shape
        self.voxel_types = voxel_types
        self.env_type = env_type
        
        self.population = [self._create_random_structure() for _ in range(self.pop_size)]
        self.best_robot = None
        self.best_fit = -np.inf
        self.num_workers = multiprocessing.cpu_count()

    def _create_random_structure(self) -> Structure:
        body, _ = sample_robot(self.robot_shape)
        return Structure(body)

    def evaluate_population(self, pool: multiprocessing.Pool) -> None:
        evaluate_with_env = partial(evaluate, env_type=self.env_type)
        fitness_scores = pool.map(evaluate_with_env, self.population)
        for ind, fit in zip(self.population, fitness_scores):
            ind.fitness = fit
            if fit > self.best_fit:
                self.best_fit = fit
                self.best_robot = copy.deepcopy(ind)
    
    def prepare_generation(self) -> None:
        """Hook to be overridden by subclasses for generation-specific preparations (e.g., speciation)."""
        pass

    def selection(self) -> Any:
        raise NotImplementedError

    def crossover(self, parents: Any) -> List[Structure]:
        raise NotImplementedError

    def mutate(self, offspring: List[Structure]) -> List[Structure]:
        raise NotImplementedError

    def run(self) -> Structure | None:
        history = {"best_fitness": [], "avg_fitness": []}

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            for gen in range(self.generations):
                self.evaluate_population(pool)

                self.prepare_generation()

                avg_fit = np.mean([ind.fitness for ind in self.population])
                history["best_fitness"].append(float(self.best_fit))
                history["avg_fitness"].append(float(avg_fit))
                
                print(f"Gen {gen+1} | Best: {self.best_fit:.4f} | Avg: {avg_fit:.4f}")
                
                self.save_history(history)
                self.save_robot(gen + 1, self.best_robot)

                parents = self.selection()
                offspring = self.crossover(parents)
                self.population = self.mutate(offspring)
        
        self.zip_results()
        return self.best_robot
