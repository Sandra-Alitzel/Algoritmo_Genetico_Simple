import numpy as np

def one_point_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator):
    L = p1.size
    if L < 2:
        return p1.copy(), p2.copy()
    cx = rng.integers(1, L)  # cut in [1, L-1]
    c1 = np.concatenate([p1[:cx], p2[cx:]])
    c2 = np.concatenate([p2[:cx], p1[cx:]])
    return c1, c2

def mutate_bitflip(child: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
    if pm <= 0:
        return child
    mask = rng.random(child.size) < pm
    child = child.copy()
    child[mask] = 1 - child[mask]
    return child