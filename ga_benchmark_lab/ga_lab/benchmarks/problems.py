from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import numpy as np

from .functions import rastrigin, bukin_n6, himmelblau, eggholder

@dataclass(frozen=True)
class Problem:
    key: str
    name: str
    func: Callable[[np.ndarray], float]
    bounds: List[Tuple[float, float]]
    dims: int

def get_problems() -> Dict[str, Problem]:
    return {
        "rastrigin_2": Problem("rastrigin_2", "Rastrigin (n=2)", rastrigin, [(-5.12, 5.12)]*2, 2),
        "rastrigin_5": Problem("rastrigin_5", "Rastrigin (n=5)", rastrigin, [(-5.12, 5.12)]*5, 5),
        "bukin_n6": Problem("bukin_n6", "Bukin N.6", bukin_n6, [(-15.0, -5.0), (-3.0, 3.0)], 2),
        "himmelblau": Problem("himmelblau", "Himmelblau", himmelblau, [(-5.0, 5.0), (-5.0, 5.0)], 2),
        "eggholder": Problem("eggholder", "Eggholder", eggholder, [(-512.0, 512.0), (-512.0, 512.0)], 2),
    }