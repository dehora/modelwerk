"""Level 4: Fully-connected (dense) layer.

A collection of neurons represented as a weight matrix and bias vector.
Forward pass returns output and cache for backprop.

A dense layer with n inputs and m outputs is equivalent to m neurons,
each receiving the same n inputs. Instead of looping over neurons, we
use a single matrix multiply: z = W @ x + b, then apply the activation
function element-wise.

    W: (m, n) matrix — row i holds the weights for output i
    b: (m,) vector  — one bias per output
    x: (n,) vector  — the input

This is the building block of the MLP: stack dense layers, and you get
a network that can learn non-linear decision boundaries.
"""

from dataclasses import dataclass, field

from modelwerk.primitives import vector, matrix
from modelwerk.primitives.random import xavier_init, random_vector

Vector = list[float]
Matrix = list[list[float]]


@dataclass
class DenseLayer:
    weights: Matrix  # (num_outputs, num_inputs)
    biases: Vector   # (num_outputs,)


@dataclass
class DenseCache:
    """Values saved during forward pass, needed for backprop."""
    inputs: Vector   # the input to this layer
    z: Vector        # pre-activation: W @ inputs + biases
    a: Vector        # post-activation: f(z)


def create_dense(rng, num_inputs: int, num_outputs: int) -> DenseLayer:
    """Create a dense layer with Xavier-initialized weights and zero biases."""
    weights = xavier_init(rng, num_inputs, num_outputs)
    biases = vector.zeros(num_outputs)
    return DenseLayer(weights=weights, biases=biases)


def dense_forward(layer: DenseLayer, inputs: Vector, activation_fn) -> tuple[Vector, DenseCache]:
    """Compute output = activation(W @ inputs + biases).

    Returns both the output and a cache of intermediate values
    needed by backpropagation.
    """
    z = vector.add(matrix.mat_vec(layer.weights, inputs), layer.biases)
    a = vector.apply(activation_fn, z)
    cache = DenseCache(inputs=inputs, z=z, a=a)
    return a, cache
