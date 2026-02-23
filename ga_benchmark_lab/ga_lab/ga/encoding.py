import numpy as np
import math
from typing import List, Tuple

def bits_for_precision(a: float, b: float, decimals: int) -> int:
    step = 10 ** (-decimals)
    states = (b - a) / step
    L = int(math.ceil(math.log2(states + 1)))
    return max(L, 1)

def decode_bits_to_unit_int(bits: np.ndarray) -> int:
    # bits: shape (L,), MSB first
    u = 0
    for bit in bits:
        u = (u << 1) | int(bit)
    return u

def decode_bits_to_real(bits: np.ndarray, a: float, b: float, decimals: int) -> float:
    L = bits.size
    u = decode_bits_to_unit_int(bits)
    denom = (2**L - 1)
    x = a + (u / denom) * (b - a)
    return float(np.round(x, decimals))

def build_bit_layout(bounds: List[Tuple[float, float]], decimals: int) -> List[int]:
    return [bits_for_precision(a, b, decimals) for (a, b) in bounds]

def decode_genome(genome: np.ndarray, bounds: List[Tuple[float, float]], bits_per_dim: List[int], decimals: int) -> np.ndarray:
    xs = []
    idx = 0
    for (a, b), L in zip(bounds, bits_per_dim):
        chunk = genome[idx: idx + L]
        xs.append(decode_bits_to_real(chunk, a, b, decimals))
        idx += L
    return np.array(xs, dtype=float)