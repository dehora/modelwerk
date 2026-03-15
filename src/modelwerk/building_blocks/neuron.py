"""Level 3: A single neuron.

The fundamental unit of a neural network: activation(dot(weights, inputs) + bias).
Composes vector operations with an activation function.
"""

from dataclasses import dataclass

from modelwerk.primitives import scalar, vector
from modelwerk.primitives.random import create_rng, random_vector

Vector = list[float]


@dataclass
class Neuron:
    weights: Vector
    bias: float


def create_neuron(weight_init, num_inputs: int) -> Neuron:
    weights = random_vector(weight_init, num_inputs, -1.0, 1.0)
    bias = 0.0
    return Neuron(weights=weights, bias=bias)


def forward(neuron: Neuron, inputs: Vector, activation_fn) -> float:
    z = scalar.add(vector.dot(neuron.weights, inputs), neuron.bias)  # pre-activation
    return activation_fn(z)
