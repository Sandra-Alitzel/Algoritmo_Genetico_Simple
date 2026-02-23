import math
import numpy as np

def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    n = x.size
    return A * n + float(np.sum(x**2 - A * np.cos(2 * math.pi * x)))

def bukin_n6(x: np.ndarray) -> float:
    X, Y = float(x[0]), float(x[1])
    return 100.0 * math.sqrt(abs(Y - 0.01 * X * X)) + 0.01 * abs(X + 10.0)

def himmelblau(x: np.ndarray) -> float:
    X, Y = float(x[0]), float(x[1])
    return (X*X + Y - 11.0)**2 + (X + Y*Y - 7.0)**2

def eggholder(x: np.ndarray) -> float:
    X, Y = float(x[0]), float(x[1])
    term1 = -(Y + 47.0) * math.sin(math.sqrt(abs(X/2.0 + (Y + 47.0))))
    term2 = -X * math.sin(math.sqrt(abs(X - (Y + 47.0))))
    return term1 + term2