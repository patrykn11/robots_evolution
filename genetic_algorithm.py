import numpy as np
import random
import copy
import multiprocessing
import gymnasium as gym

from evogym import sample_robot
from structure import Structure
from controller import Controller
from evaluation import evaluate as eval_function


class GeneticAlgorithm:
    def __init__(self,
                 pop_size=80,
                 generations=3,
                 mutation_rate=0.25,
                 robot_shape=(5, 5)):
        
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.robot_shape = robot_shape
        
        self.population = [self._create_random_structure() for _ in range(self.pop_size)]
        self.best_robot = None
        self.best_fit = -np.inf
        self.processes = multiprocessing.cpu_count()

    def _create_random_structure(self):
        body, connections = sample_robot(self.robot_shape)
        return Structure(body)

    def mutate(self, parent):
        for _ in range(20):
            new_body = parent.body.copy()
            r = np.random.randint(0, self.robot_shape[0])
            c = np.random.randint(0, self.robot_shape[1])
            new_body[r, c] = np.random.choice(parent.get_1D_body())
            temp_structure = Structure(new_body)
            if temp_structure.is_valid():
                return temp_structure
        return parent

    def crossover(self, p1, p2):
        for _ in range(20):
            child_body = p1.body.copy()
            axis = np.random.randint(0, 2)
            cut = np.random.randint(1, self.robot_shape[axis] - 1)
            
            if axis == 0:
                child_body[cut:, :] = p2.body[cut:, :]
            else:
                child_body[:, cut:] = p2.body[:, cut:]
            
            _, child_connections = sample_robot(self.robot_shape)
            temp_structure = Structure(child_body)
            if temp_structure.is_valid():
                return temp_structure
        return p1

    def evaluate(self, pool):
        fitness_scores = pool.map(eval_function, self.population)
        
        for ind, fit in zip(self.population, fitness_scores):
            ind.fitness = fit

    def selection(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        current_best = self.population[0]

        if current_best.fitness > self.best_fit:
            self.best_fit = current_best.fitness
            self.best_robot = copy.deepcopy(current_best)

        next_gen = [copy.deepcopy(current_best)]
        parents = self.population[:self.pop_size // 2]

        while len(next_gen) < self.pop_size:
            p1, p2 = random.choice(parents), random.choice(parents)
            if random.random() < 0.8:
                child = self.crossover(p1, p2)
            else:
                child = copy.deepcopy(p1)

            if random.random() < self.mutation_rate:
                child = self.mutate(child)

            next_gen.append(child)
        self.population = next_gen

    def run(self):
        with multiprocessing.Pool(processes=self.processes) as pool:
            for gen in range(self.generations):
                print(f"{gen+1} {self.best_fit} \n")       
                self.evaluate(pool)
                self.selection()

        return self.best_robot
