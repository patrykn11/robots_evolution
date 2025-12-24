class Species:
    def __init__(self, representative):
        self.representative = copy.deepcopy(representative)
        self.members = []
        self.avg_fitness = 0
        self.best_fitness = -np.inf
        self.staleness = 0

    def add_member(self, ind):
        self.members.append(ind)
        if ind.fitness > self.best_fitness:
            self.best_fitness = ind.fitness
            self.staleness = 0
        else:
            self.staleness += 1