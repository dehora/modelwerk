"""Level 2: Activation functions.

Non-linear functions applied to neuron outputs — step, sigmoid,
tanh, relu, softmax. Each includes its derivative for backprop.
Built from scalar and vector operations.
"""

import math

from modelwerk.primitives import scalar, vector

Vector = list[float]


def step(x: float) -> float:
    """Return 1 if x >= 0, else 0 — the binary threshold activation."""
    return 1.0 if x >= 0 else 0.0


def sigmoid(x: float) -> float:
    """Squash x into (0, 1) — smooth alternative to step."""
    return scalar.inverse(scalar.add(1.0, scalar.exp(scalar.negate(x))))


def sigmoid_derivative(x: float) -> float:
    """Derivative of sigmoid: σ(x) * (1 - σ(x))."""
    s = sigmoid(x)
    return scalar.multiply(s, scalar.subtract(1.0, s))


def tanh_(x: float) -> float:
    """Squash x into (-1, 1). Trailing underscore avoids shadowing math.tanh."""
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    """Derivative of tanh: 1 - tanh(x)²."""
    t = tanh_(x)
    return scalar.subtract(1.0, scalar.multiply(t, t))


def relu(x: float) -> float:
    """Return x if positive, else 0 — the rectified linear unit."""
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    """Derivative of ReLU: 1 if x > 0, else 0."""
    return 1.0 if x > 0 else 0.0


def softmax(v: Vector) -> Vector:
    """Convert a vector of scores into probabilities that sum to 1."""
    m = vector.max_val(v)
    exps = [scalar.exp(scalar.subtract(x, m)) for x in v]
    total = vector.sum_all(exps)
    return [scalar.multiply(e, scalar.inverse(total)) for e in exps]
