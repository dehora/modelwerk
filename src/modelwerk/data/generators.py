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


def parity(rng, seq_len: int = 16, n_samples: int = 100) -> tuple[list[list[float]], list[list[float]]]:
    """Random ±1 sequences with cumulative parity targets.

    Each input is a sequence of +1.0 or -1.0 values.
    Each target is the cumulative parity at each position:
    +1 if the product of values up to that position is positive, 0 otherwise.
    """
    inputs: list[list[float]] = []
    targets: list[list[float]] = []

    for _ in range(n_samples):
        seq = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(seq_len)]
        # Cumulative parity: product of signs up to each position
        target = []
        product = 1.0
        for val in seq:
            product *= val
            target.append(1.0 if product > 0 else 0.0)
        inputs.append(seq)
        targets.append(target)

    return inputs, targets


def selective_copying(
    rng, seq_len: int = 32, n_copy: int = 4, vocab_size: int = 8, n_samples: int = 100,
) -> tuple[list[list[int]], list[list[int]]]:
    """Random-spacing selective copying task from the Mamba paper.

    Vocabulary: {0: BLANK, 1: COPY_MARKER, 2..vocab_size-1: data tokens}

    Each input has n_copy data tokens placed at random positions in the first
    half, followed by a COPY_MARKER, then blanks. The target is all blanks
    until the marker, then the data tokens in order.

    Returns (inputs, targets) where each is a list of integer token sequences.
    """
    inputs: list[list[int]] = []
    targets: list[list[int]] = []

    # Region for data tokens: positions 0..marker_pos-1
    # marker_pos is chosen so there's room for n_copy output slots after it
    marker_pos = seq_len - n_copy - 1

    for _ in range(n_samples):
        inp = [0] * seq_len
        tgt = [0] * seq_len

        # Choose n_copy random positions in [0, marker_pos) for data tokens
        positions = []
        available = list(range(marker_pos))
        for i in range(n_copy):
            idx = int(rng.random() * len(available))
            positions.append(available[idx])
            available[idx] = available[-1]
            available.pop()
        positions.sort()

        # Place data tokens at those positions
        data_tokens = []
        for pos in positions:
            token = int(rng.random() * (vocab_size - 2)) + 2  # range [2, vocab_size)
            inp[pos] = token
            data_tokens.append(token)

        # Place copy marker
        inp[marker_pos] = 1

        # Target: blanks until marker, then data tokens in order
        for i, token in enumerate(data_tokens):
            tgt[marker_pos + 1 + i] = token

        inputs.append(inp)
        targets.append(tgt)

    return inputs, targets


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
