import numpy as np
import random
import copy
from typing import List, Tuple
from structure import Structure
from algorithms.species_genetic_algorithm.species import Species
from algorithms.base_genetic_algorithm import BaseGeneticAlgorithm


class SpeciesGeneticAlgorithm(BaseGeneticAlgorithm):
    def __init__(
        self,
        experiment_name: str,
        target_species: int = 6,
        mutation_rate: float = 0.5,
        crossover_probability: float = 0.7,
        robot_shape: Tuple[int, int] = (5, 5),
        voxel_types: List[int] = [0, 1, 2, 3, 4],
        pop_size=100, 
        generations=80, 
    ):
        super().__init__(
            experiment_name, 
            pop_size=pop_size, 
            generations=generations, 
            robot_shape=robot_shape,
            voxel_types=voxel_types,
        )
        self.target_species = target_species
        self.mutation_rate = mutation_rate
        self.crossover_probability = crossover_probability
        self.threshold = 0.2
        self.species = []

    def get_distance(self, ind1: Structure, ind2: Structure) -> float:
        diff = np.sum(ind1.body != ind2.body)
        return diff / 25.0

    def prepare_generation(self):
        self._speciate()

    def _speciate(self):
        for s in self.species:
            if len(s.members) > 0:
                s.representative = copy.deepcopy(random.choice(s.members))
            s.members = []

        for ind in self.population:
            found = False
            for s in self.species:
                if self.get_distance(ind, s.representative) < self.threshold:
                    s.add_member(ind)
                    found = True
                    break
            if not found:
                new_s = Species(ind)
                new_s.add_member(ind)
                self.species.append(new_s)

        self.species = [s for s in self.species if len(s.members) > 0]

        if len(self.species) < self.target_species:
            self.threshold -= 0.01
        elif len(self.species) > self.target_species:
            self.threshold += 0.01
        self.threshold = max(0.05, self.threshold)

    def selection(self) -> List[Species]:
        for s in self.species:
            n_j = len(s.members)
            for ind in s.members:
                ind.adjusted_fitness = ind.fitness / n_j
            s.avg_fitness = np.mean([ind.adjusted_fitness for ind in s.members])

        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.population[0].fitness > self.best_fit:
            self.best_fit = self.population[0].fitness
            self.best_robot = copy.deepcopy(self.population[0])
        return self.species

    def crossover(self, species_list: List[Species]) -> List[Structure]:
        new_population = [copy.deepcopy(self.best_robot)]
        total_avg_fit = sum(s.avg_fitness for s in species_list) or 1.0

        for s in species_list:
            num_offspring = int((s.avg_fitness / total_avg_fit) * self.pop_size)
            s.members.sort(key=lambda x: x.fitness, reverse=True)
            pool = s.members[:max(1, len(s.members)//2)]
            for _ in range(num_offspring):
                if len(new_population) < self.pop_size:
                    if len(pool) > 1 and random.random() < self.crossover_probability:
                        p1, p2 = random.sample(pool, 2)
                        child = self._do_crossover_body(p1, p2)
                    else:
                        child = copy.deepcopy(random.choice(pool))
                    new_population.append(child)

        while len(new_population) < self.pop_size:
            new_population.append(copy.deepcopy(self.best_robot))
        return new_population

    def _do_crossover_body(self, p1: Structure, p2: Structure) -> Structure:
        child_body = p1.body.copy()
        axis = np.random.randint(0, 2)
        cut = np.random.randint(1, self.robot_shape[axis])
        if axis == 0:
            child_body[cut:, :] = p2.body[cut:, :]
        else:
            child_body[:, cut:] = p2.body[:, cut:]
        temp = Structure(child_body)
        return temp if temp.is_valid() else copy.deepcopy(p1)

    def mutate(self, next_gen: List[Structure]) -> List[Structure]:
        for ind in next_gen:
            if ind == self.best_robot:
                continue
            if random.random() < self.mutation_rate:
                for _ in range(random.randint(1, 4)):
                    r, c = np.random.randint(0, 5), np.random.randint(0, 5)
                    old_val = ind.body[r, c]
                    ind.body[r, c] = np.random.choice(self.voxel_types)
                    if not ind.is_valid():
                        ind.body[r, c] = old_val
        return next_gen
