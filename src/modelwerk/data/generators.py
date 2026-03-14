"""Synthetic dataset generators.

Logic gates (AND, OR, XOR), spirals, circles — small datasets
for testing and visualizing model behavior.
"""

import math

from modelwerk.primitives.random import uniform

Vector = list[float]
Dataset = tuple[list[Vector], list[float]]


def and_gate() -> Dataset:
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    labels = [0.0, 0.0, 0.0, 1.0]
    return inputs, labels


def or_gate() -> Dataset:
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    labels = [0.0, 1.0, 1.0, 1.0]
    return inputs, labels


def nand_gate() -> Dataset:
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    labels = [1.0, 1.0, 1.0, 0.0]
    return inputs, labels


def xor_gate() -> Dataset:
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    labels = [0.0, 1.0, 1.0, 0.0]
    return inputs, labels


def circles(rng, n_samples: int = 100, noise: float = 0.1) -> Dataset:
    """Two concentric circles — inner ring is class 1, outer is class 0.

    A non-linear classification problem that requires at least one
    hidden layer. No straight line can separate inside from outside.
    """
    inputs: list[Vector] = []
    labels: list[float] = []

    for i in range(n_samples):
        theta = uniform(rng, 0.0, 2 * math.pi)

        if i % 2 == 0:
            # Inner circle (class 1)
            r = uniform(rng, 0.0, 0.4)
            x = r * math.cos(theta) + uniform(rng, -noise, noise)
            y = r * math.sin(theta) + uniform(rng, -noise, noise)
            labels.append(1.0)
        else:
            # Outer ring (class 0)
            r = uniform(rng, 0.7, 1.0)
            x = r * math.cos(theta) + uniform(rng, -noise, noise)
            y = r * math.sin(theta) + uniform(rng, -noise, noise)
            labels.append(0.0)

        inputs.append([x, y])

    return inputs, labels
