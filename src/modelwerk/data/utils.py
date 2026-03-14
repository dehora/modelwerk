"""Data utilities.

Batching, shuffling, train/test splitting, one-hot encoding,
and normalization.
"""

Vector = list[float]


def one_hot(label: int, num_classes: int = 10) -> Vector:
    """Convert an integer label to a one-hot vector."""
    v = [0.0] * num_classes
    v[label] = 1.0
    return v


def subsample(rng, data: list, labels: list, n: int) -> tuple[list, list]:
    """Select n random samples using seeded RNG."""
    indices = list(range(len(data)))
    # Fisher-Yates shuffle on indices, then take first n
    for i in range(len(indices) - 1, 0, -1):
        j = int(rng.uniform(0, i + 1))
        if j > i:
            j = i
        indices[i], indices[j] = indices[j], indices[i]
    indices = indices[:n]
    return [data[i] for i in indices], [labels[i] for i in indices]


def shuffle_together(rng, data: list, labels: list) -> tuple[list, list]:
    """Shuffle data and labels in sync using seeded RNG."""
    return subsample(rng, data, labels, len(data))
