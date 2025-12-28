import time
import itertools
from typing import List, Dict, Any, Type
from algorithms.base_genetic_algorithm import BaseGeneticAlgorithm


class ExperimentsRunner:
    """
    Class to run and manage experiments.
    Allows running multiple genetic algorithm configurations sequentially.
    """
    
    def __init__(self):
        self.experiments = []

    def add_experiment(self, algo_class, params):
        """
        Registers a single experiment to be run.
        
        Args:
            algo_class: The class of the algorithm (e.g. SpeciesGeneticAlgorithm).
            params: Dictionary of parameters for the algorithm's __init__.
                    Must include 'experiment_name'.
        """
        if "experiment_name" not in params:
            raise ValueError("Parameters must include 'experiment_name' to identify the run.")
            
        self.experiments.append({
            "class": algo_class,
            "params": params
        })

    def run_all(self):
        """
        Executes all registered experiments sequentially.
        """
        total_exps = len(self.experiments)
        print(f"--- Starting Batch Execution of {total_exps} Experiments ---")
        
        for i, exp in enumerate(self.experiments):
            algo_class = exp["class"]
            params = exp["params"]
            name = params["experiment_name"]
            
            print(f"\n[{i+1}/{total_exps}] Running: {name}")
            print(f"Algorithm: {algo_class.__name__}")
            print(f"Parameters: {params}")
            
            start_time = time.time()
            algorithm = algo_class(**params)
            winner = algorithm.run()
            print(f"Success! Best Fitness: {algorithm.best_fit:.4f}")
            
            duration = time.time() - start_time
            print(f"Duration: {duration:.2f}s")
            
        print("\n--- All Experiments Completed ---")