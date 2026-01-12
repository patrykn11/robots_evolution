from experiments_runner import ExperimentsRunner
from algorithms.species_genetic_algorithm.genetic_algorithm import SpeciesGeneticAlgorithm
from algorithms.default_genetic_algorithm.genetic_algorithm import GeneticAlgorithm
# from algorithms.MAPElites_genetic_algorithm.genetic_algorithm import MAPElitesAlgorithm


def run_all_algorithms():
    """
    Here we define and run basic experiments for different genetic algorithms.
    """
    runner = ExperimentsRunner()

    runner.add_experiment(
        algo_class=GeneticAlgorithm,
        params={
            "experiment_name": "genetic_experiment",
            "generations": 120,
            "pop_size": 120,
        }
    )

    runner.add_experiment(
        algo_class=SpeciesGeneticAlgorithm,
        params={
            "experiment_name": "species_genetic_experiment",
            "generations": 120,
            "pop_size": 120,
        }
    )

    # runner.add_experiment(
    #     algo_class=MAPElitesAlgorithm,
    #     params={
    #         "experiment_name": "map_elites_experiment",
    #         "generations": 120,
    #         "pop_size": 120,
    #     }
    # )
    
    runner.run_all()

if __name__ == "__main__":
    run_all_algorithms()