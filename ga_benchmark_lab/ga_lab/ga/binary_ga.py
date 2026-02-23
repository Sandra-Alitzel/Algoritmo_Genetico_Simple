import numpy as np
from typing import Callable, List, Tuple, Dict, Any

from .config import GAConfig
from .encoding import build_bit_layout, decode_genome
from .selection import costs_to_fitness, roulette_select
from .operators import one_point_crossover, mutate_bitflip

Objective = Callable[[np.ndarray], float]

def run_binary_ga(bounds: List[Tuple[float, float]], objective_cost: Objective, config: GAConfig) -> Dict[str, Any]:
    rng = np.random.default_rng(config.seed)

    bits_per_dim = build_bit_layout(bounds, config.decimals)
    genome_len = int(sum(bits_per_dim))

    if config.pm_mode == "1/L":
        pm = 1.0 / genome_len
    else:
        pm = float(config.pm_value)

    # 1) init population
    pop = rng.integers(0, 2, size=(config.pop_size, genome_len), dtype=np.int8)

    history_best_cost = np.zeros(config.generations, dtype=float)
    history_best_x = np.zeros((config.generations, len(bounds)), dtype=float)
    
    history_mean_cost = np.zeros(config.generations, dtype=float)
    history_std_cost = np.zeros(config.generations, dtype=float)
    history_diversity = np.zeros(config.generations, dtype=float)
    #history_best_idx = np.zeros(config.generations, dtype=int)

    best_cost = float("inf")
    best_x = None

    for g in range(config.generations):
        # 2) evaluation
        xs = np.array([decode_genome(ind, bounds, bits_per_dim, config.decimals) for ind in pop])
        costs = np.array([objective_cost(x) for x in xs], dtype=float)
        
        history_mean_cost[g] = float(np.mean(costs))
        history_std_cost[g] = float(np.std(costs))

        # Diversity para genoma binario: promedio de 2 p (1-p) por bit
        p = np.mean(pop, axis=0)  # proporción de 1s por posición
        history_diversity[g] = float(np.mean(2.0 * p * (1.0 - p)))

        # track best-so-far
        i_best = int(np.argmin(costs))
        if float(costs[i_best]) < best_cost:
            best_cost = float(costs[i_best])
            best_x = xs[i_best].copy()

        history_best_cost[g] = best_cost
        history_best_x[g] = best_x

        # 3) selection (roulette on transformed fitness)
        fitness = costs_to_fitness(costs)

        # elitism: keep k best genomes
        k = int(config.elitism)
        elite_idx = np.argsort(costs)[:k] if k > 0 else np.array([], dtype=int)
        elites = pop[elite_idx].copy() if k > 0 else np.empty((0, genome_len), dtype=np.int8)

        # 4) create new population via selection + crossover + mutation
        new_pop = []
        if k > 0:
            for e in elites:
                new_pop.append(e)

        while len(new_pop) < config.pop_size:
            parents_idx = roulette_select(fitness, rng, k=2)
            p1, p2 = pop[parents_idx[0]], pop[parents_idx[1]]

            if rng.random() < config.pc:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate_bitflip(c1, pm, rng)
            new_pop.append(c1)
            if len(new_pop) < config.pop_size:
                c2 = mutate_bitflip(c2, pm, rng)
                new_pop.append(c2)

        # 5) replacement
        pop = np.array(new_pop, dtype=np.int8)

    return {
        "best_cost": best_cost,
        "best_x": best_x,
        "history_best_cost": history_best_cost,
        "history_best_x": history_best_x,
        "genome_len": genome_len,
        "bits_per_dim": bits_per_dim,
        "pm_used": pm,
        "config": config,
        "history_mean_cost": history_mean_cost,
        "history_std_cost": history_std_cost,
        "history_diversity": history_diversity,
    }