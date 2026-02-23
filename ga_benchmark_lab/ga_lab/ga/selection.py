import numpy as np

def costs_to_fitness(costs: np.ndarray) -> np.ndarray:
    # minimization -> higher fitness for smaller cost, guaranteed positive
    cmin = float(np.min(costs))
    adj = costs - cmin
    fitness = 1.0 / (1.0 + adj)
    return fitness

def roulette_select(fitness: np.ndarray, rng: np.random.Generator, k: int) -> np.ndarray:
    total = float(np.sum(fitness))
    if total <= 0:
        probs = np.ones_like(fitness) / fitness.size
    else:
        probs = fitness / total
    return rng.choice(fitness.size, size=k, replace=True, p=probs)