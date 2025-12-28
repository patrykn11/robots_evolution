import numpy as np
import random
import copy
from typing import List, Tuple
from structure import Structure
from algorithms.base_genetic_algorithm import BaseGeneticAlgorithm

class GeneticAlgorithm(BaseGeneticAlgorithm):
    def __init__(
        self,
        experiment_name: str,
        mutation_rate: float = 0.25,
        crossover_probability: float = 0.8,
        robot_shape: Tuple[int, int] = (5, 5),
        voxel_types: List[int] = [0, 1, 2, 3, 4],
        env_type: str = 'Walker-v0',
        pop_size=80,
        generations=50,
    ):
        super().__init__(
            experiment_name,
            pop_size=pop_size,
            generations=generations,
            robot_shape=robot_shape,
            voxel_types=voxel_types,
            env_type=env_type,
        )
        self.mutation_rate = mutation_rate
        self.crossover_probability = crossover_probability

    def selection(self) -> List[Structure]:
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.population[0].fitness > self.best_fit:
            self.best_fit = self.population[0].fitness
            self.best_robot = copy.deepcopy(self.population[0])
        return self.population[:self.pop_size // 2]

    def crossover(self, parents: List[Structure]) -> List[Structure]:
        next_gen = [copy.deepcopy(self.best_robot)]
        
        while len(next_gen) < self.pop_size:
            p1, p2 = random.choice(parents), random.choice(parents)
            
            if random.random() < self.crossover_probability:
                child = self._do_crossover_body(p1, p2)
            else:
                child = copy.deepcopy(p1)
            
            next_gen.append(child)
        return next_gen

    def _do_crossover_body(self, p1: Structure, p2: Structure) -> Structure:
        for _ in range(20):
            child_body = p1.body.copy()
            axis = np.random.randint(0, 2)
            cut = np.random.randint(1, self.robot_shape[axis] - 1)
            if axis == 0:
                child_body[cut:, :] = p2.body[cut:, :]
            else:
                child_body[:, cut:] = p2.body[:, cut:]
            
            temp = Structure(child_body)
            if temp.is_valid():
                return temp
        return p1

    def mutate(self, next_gen: List[Structure]) -> List[Structure]:
        mutated_gen = []
        for ind in next_gen:
            if ind == self.best_robot:
                mutated_gen.append(ind)
                continue
                
            if random.random() < self.mutation_rate:
                new_ind = self._do_mutation_body(ind)
                mutated_gen.append(new_ind)
            else:
                mutated_gen.append(ind)
        return mutated_gen

    def _do_mutation_body(self, parent: Structure) -> Structure:
        for _ in range(20):
            new_body = parent.body.copy()
            r, c = np.random.randint(0, self.robot_shape[0]), np.random.randint(0, self.robot_shape[1])
            new_body[r, c] = np.random.choice(self.voxel_types)
            temp = Structure(new_body)
            if temp.is_valid():
                return temp
        return parent
