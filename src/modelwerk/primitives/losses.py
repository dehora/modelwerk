"""Level 2: Loss functions.

Measure how far predictions are from targets — MSE, cross-entropy.
Each includes its derivative for backprop.
Built from scalar and vector operations.
"""

from modelwerk.primitives import scalar, vector

Vector = list[float]


def mse(predicted: Vector, actual: Vector) -> float:
    """Mean squared error between predicted and actual vectors."""
    diffs = vector.subtract(predicted, actual)
    squared = [scalar.multiply(d, d) for d in diffs]
    return scalar.multiply(vector.sum_all(squared), scalar.inverse(len(squared)))


def mse_derivative(predicted: Vector, actual: Vector) -> Vector:
    """Gradient of MSE with respect to each predicted value."""
    n = len(predicted)
    factor = 2.0 / n
    return [scalar.multiply(factor, scalar.subtract(p, a))
            for p, a in zip(predicted, actual)]


def cross_entropy(predicted: Vector, actual: Vector) -> float:
    """Cross-entropy loss — measures divergence between predicted and actual distributions."""
    terms = [scalar.multiply(a, scalar.log(p))
             for p, a in zip(predicted, actual)]
    return scalar.negate(vector.sum_all(terms))


def cross_entropy_derivative(predicted: Vector, actual: Vector) -> Vector:
    """Gradient of cross-entropy with respect to each predicted value."""
    return [scalar.negate(scalar.multiply(a, scalar.inverse(p)))
            for p, a in zip(predicted, actual)]
