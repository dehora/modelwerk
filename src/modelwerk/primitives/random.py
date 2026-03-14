"""Level 0: Random number generation.

Seeded RNG wrappers for reproducible weight initialization.
All randomness in the project flows through this module.
"""

import math
import random as _random

Vector = list[float]
Matrix = list[list[float]]


def create_rng(seed: int = 42) -> _random.Random:
    return _random.Random(seed)


def uniform(rng: _random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def random_vector(rng: _random.Random, n: int, lo: float = -1.0, hi: float = 1.0) -> Vector:
    return [rng.uniform(lo, hi) for _ in range(n)]


def random_matrix(rng: _random.Random, rows: int, cols: int, lo: float = -1.0, hi: float = 1.0) -> Matrix:
    return [random_vector(rng, cols, lo, hi) for _ in range(rows)]


def xavier_init(rng: _random.Random, fan_in: int, fan_out: int) -> Matrix:
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return random_matrix(rng, fan_out, fan_in, -limit, limit)
