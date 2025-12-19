import numpy as np
import random
import copy
import multiprocessing
import gymnasium as gym

from evogym import sample_robot
from structure import Structure
from controller import Controller
from evaluation import evaluate 

class GeneticAlgorithm:
    def __init__(self, pop_size=80, generations=50, mutation_rate=0.25, crossover_probability=0.8, 
                 robot_shape=(5, 5), voxel_types=[0, 1, 2, 3, 4],
                 selection_func=None, crossover_func=None, mutation_func=None):
        
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_probability = crossover_probability
        self.robot_shape = robot_shape
        self.voxel_types = voxel_types

        self.population = [self._create_random_structure() for _ in range(self.pop_size)]
        self.best_robot = None
        self.best_fit = -np.inf
        self.num_workers = multiprocessing.cpu_count()

        self.selection_strategy = selection_func if selection_func else self.selection
        self.crossover_strategy = crossover_func if crossover_func else self.crossover
        self.mutation_strategy = mutation_func if mutation_func else self.mutate

    def _create_random_structure(self):
        body, connections = sample_robot(self.robot_shape)
        return Structure(body)

    def selection(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.population[0].fitness > self.best_fit:
            self.best_fit = self.population[0].fitness
            self.best_robot = copy.deepcopy(self.population[0])
        
        return self.population[:self.pop_size // 2]

    def crossover(self, parents):
        next_gen = [copy.deepcopy(self.best_robot)]
        
        while len(next_gen) < self.pop_size:
            p1, p2 = random.choice(parents), random.choice(parents)
            
            if random.random() < self.crossover_probability:
                child = self._do_crossover_body(p1, p2)
            else:
                child = copy.deepcopy(p1)
            
            next_gen.append(child)
        return next_gen

    def _do_crossover_body(self, p1, p2):
        for _ in range(20):
            child_body = p1.body.copy()
            axis = np.random.randint(0, 2)
            cut = np.random.randint(1, self.robot_shape[axis] - 1)
            if axis == 0: child_body[cut:, :] = p2.body[cut:, :]
            else: child_body[:, cut:] = p2.body[:, cut:]
            
            temp = Structure(child_body)
            if temp.is_valid(): return temp
        return p1

    def mutate(self, next_gen):
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

    def _do_mutation_body(self, parent):
        for _ in range(20):
            new_body = parent.body.copy()
            r, c = np.random.randint(0, self.robot_shape[0]), np.random.randint(0, self.robot_shape[1])
            new_body[r, c] = np.random.choice(self.voxel_types)
            temp = Structure(new_body)
            if temp.is_valid(): return temp
        return parent

    def evaluate_population(self, pool):
        fitness_scores = pool.map(evaluate, self.population)
        for ind, fit in zip(self.population, fitness_scores):
            ind.fitness = fit

    def run(self):
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            for gen in range(self.generations):
                self.evaluate_population(pool)
                
                parents = self.selection_strategy()
                offspring = self.crossover_strategy(parents)
                self.population = self.mutation_strategy(offspring)
                
                print(f"Gen {gen+1} | Best Fit: {self.best_fit}")
        
        return self.best_robot