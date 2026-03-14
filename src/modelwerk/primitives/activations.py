"""Level 2: Activation functions.

Non-linear functions applied to neuron outputs — step, sigmoid,
tanh, relu, softmax. Each includes its derivative for backprop.
Built from scalar and vector operations.
"""

import math

from modelwerk.primitives import scalar, vector

Vector = list[float]


def step(x: float) -> float:
    return 1.0 if x >= 0 else 0.0


def sigmoid(x: float) -> float:
    return scalar.inverse(scalar.add(1.0, scalar.exp(scalar.negate(x))))


def sigmoid_derivative(x: float) -> float:
    s = sigmoid(x)
    return scalar.multiply(s, scalar.subtract(1.0, s))


def tanh_(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    t = tanh_(x)
    return scalar.subtract(1.0, scalar.multiply(t, t))


def relu(x: float) -> float:
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def softmax(v: Vector) -> Vector:
    m = vector.max_val(v)
    exps = [scalar.exp(scalar.subtract(x, m)) for x in v]
    total = vector.sum_all(exps)
    return [scalar.multiply(e, scalar.inverse(total)) for e in exps]
