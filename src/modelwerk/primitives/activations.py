"""Level 2: Activation functions and normalization.

Non-linear functions applied to neuron outputs — step, sigmoid,
tanh, relu, softmax. Each includes its derivative for backprop.
Also includes layer normalization (zero mean, unit variance) used
by the transformer. Built from scalar and vector operations.
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


def silu(x: float) -> float:
    """SiLU (Sigmoid Linear Unit): x * sigmoid(x). Smooth, non-monotonic activation."""
    return scalar.multiply(x, sigmoid(x))


def silu_derivative(x: float) -> float:
    """Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))."""
    s = sigmoid(x)
    return scalar.add(s, scalar.multiply(x, scalar.multiply(s, scalar.subtract(1.0, s))))


def softplus(x: float) -> float:
    """log(1 + exp(x)), clamped for large x. Keeps Δ positive in Mamba."""
    if x > 20.0:
        return x
    return scalar.log(scalar.add(1.0, scalar.exp(x)))


def softplus_derivative(x: float) -> float:
    """Derivative of softplus: sigmoid(x)."""
    return sigmoid(x)


def identity(x: float) -> float:
    """Pass-through activation — no transformation."""
    return x


def identity_derivative(x: float) -> float:
    """Derivative of identity: always 1."""
    return 1.0


def layer_norm(v: Vector) -> Vector:
    """Normalize a vector to zero mean and unit variance.

    Simplified: no learnable gamma/beta parameters.
    The following dense layer absorbs any needed scale/shift.
    """
    n = len(v)
    mean = vector.sum_all(v) / n
    centered = [scalar.subtract(x, mean) for x in v]
    variance = vector.sum_all([scalar.multiply(val, val) for val in centered]) / n
    eps = 1e-5  # prevents division by zero when variance is tiny
    std_inv = scalar.inverse(scalar.power(scalar.add(variance, eps), 0.5))
    return [scalar.multiply(val, std_inv) for val in centered]


def layer_norm_backward(grad_out: Vector, normed_input: Vector, original_input: Vector) -> Vector:
    """Backward pass for layer norm.

    Args:
        grad_out: gradient flowing back through layer norm
        normed_input: the output of layer_norm (the normalized values)
        original_input: the input that was passed to layer_norm
    """
    # Layer norm has three steps: center (subtract mean), compute variance,
    # scale by 1/std. The backward pass reverses these, accumulating how
    # the gradient flows through each step. The d_ prefix means "gradient
    # with respect to" — e.g. d_variance is how much the loss changes
    # when variance changes.
    n = len(grad_out)
    mean = vector.sum_all(original_input) / n
    centered = [scalar.subtract(x, mean) for x in original_input]
    variance = vector.sum_all([scalar.multiply(val, val) for val in centered]) / n
    eps = 1e-5  # prevents division by zero when variance is tiny
    std_inv = scalar.inverse(scalar.power(scalar.add(variance, eps), 0.5))

    # d_centered = grad_out * std_inv
    d_centered = [scalar.multiply(g, std_inv) for g in grad_out]

    # d_variance = sum(grad_out * centered * -0.5 * (var + eps)^(-3/2))
    d_variance = 0.0
    var_factor = scalar.multiply(-0.5, scalar.power(scalar.add(variance, eps), -1.5))
    for dim in range(n):
        d_variance = scalar.add(d_variance,
                                scalar.multiply(grad_out[dim],
                                                scalar.multiply(centered[dim], var_factor)))

    # d_mean = sum(-d_centered) + d_variance * sum(-2 * centered) / n
    d_mean = 0.0
    for dim in range(n):
        d_mean = scalar.subtract(d_mean, d_centered[dim])
    sum_centered = vector.sum_all(centered)
    d_mean = scalar.add(d_mean,
                        scalar.multiply(d_variance, scalar.multiply(-2.0 / n, sum_centered)))

    # d_input = d_centered + d_variance * 2 * centered / n + d_mean / n
    d_input = []
    for dim in range(n):
        grad = scalar.add(d_centered[dim],
                          scalar.add(scalar.multiply(d_variance,
                                                     scalar.multiply(2.0 / n, centered[dim])),
                                     d_mean / n))
        d_input.append(grad)
    return d_input


def softmax(v: Vector) -> Vector:
    """Convert a vector of scores into probabilities that sum to 1."""
    # Subtract max for numerical stability — prevents exp() overflow
    # while producing identical probabilities (the math cancels out).
    max_score = vector.max_val(v)
    exps = [scalar.exp(scalar.subtract(x, max_score)) for x in v]
    total = vector.sum_all(exps)
    return [scalar.multiply(e, scalar.inverse(total)) for e in exps]
