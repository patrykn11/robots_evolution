from experiments_runner import ExperimentsRunner
from algorithms.species_genetic_algorithm.genetic_algorithm import SpeciesGeneticAlgorithm
from algorithms.default_genetic_algorithm.genetic_algorithm import GeneticAlgorithm

def run_batch():
    runner = ExperimentsRunner()

    # runner.add_experiment(
    #     algo_class=SpeciesGeneticAlgorithm,
    #     params={
    #         "experiment_name": "baseline_species",
    #         "generations": 5,
    #         "pop_size": 100,
    #         "target_species": 6
    #     }
    # )

    runner.add_experiment(
        algo_class=GeneticAlgorithm,
        params={
            "experiment_name": "baseline_genetic",
            "generations": 50,
            "pop_size": 80,
        }
    )
    
    runner.run_all()

if __name__ == "__main__":
    run_batch()