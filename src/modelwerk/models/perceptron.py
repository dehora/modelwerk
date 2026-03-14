"""Perceptron (Rosenblatt, 1958).

The perceptron is a single neuron that classifies inputs into two
categories — true (1) or false (0). It was the first model that
could *learn* from data rather than being explicitly programmed.

How it works:

    x₁ * w₁ ──▶ o₁ ──┐
                      ├──▶ sum(o₁, o₂) + bias ──▶ step() ──▶ prediction
    x₂ * w₂ ──▶ o₂ ──┘

Each input is multiplied by a weight, the results are summed with
a bias term, and the total is passed through a step function: if
it exceeds a threshold, output 1; otherwise output 0.

The learning rule is direct — no backpropagation needed:

    error = target - prediction
    wᵢ    = wᵢ + learning_rate × error × xᵢ
    bias  = bias + learning_rate × error

When the prediction is wrong, nudge the weights in the direction
that would have produced the right answer. When it's right, the
error is 0 and nothing changes.

Limitation: the perceptron can only learn linearly separable
patterns — problems where a single straight line can divide the
two classes. AND, OR, and NAND are linearly separable. XOR is not.
This was shown by Minsky & Papert (1969) and is addressed by the
multi-layer perceptron (MLP) in lesson 2.
"""

from modelwerk.primitives import scalar, vector
from modelwerk.primitives.activations import step
from modelwerk.building_blocks.neuron import Neuron, create_neuron

Vector = list[float]


def create_perceptron(weight_init, num_inputs: int) -> Neuron:
    """Create a perceptron with random initial weights and zero bias."""
    return create_neuron(weight_init, num_inputs)


def predict(neuron: Neuron, inputs: Vector) -> int:
    """Compute the perceptron's prediction for a single input.

    This is the forward pass:
      z = dot(weights, inputs) + bias
      prediction = step(z)   # 1 if z >= 0, else 0
    """
    z = scalar.add(vector.dot(neuron.weights, inputs), neuron.bias)
    return int(step(z))


def train(
    neuron: Neuron,
    data: list[Vector],
    labels: list[float],
    learning_rate: float = 0.1,
    epochs: int = 100,
) -> list[float]:
    """Train with the perceptron learning rule.

    For each data point, compare the prediction to the target. If wrong,
    adjust weights proportionally to the input and the error. The learning
    rate (lr) controls how large each adjustment is.

    Returns the total error per epoch — a flat line at 0 means the
    perceptron has learned the pattern. A line that never settles (like
    XOR) means the pattern isn't linearly separable.
    """
    error_history: list[float] = []

    for _ in range(epochs):
        total_error = 0.0
        for inputs, target in zip(data, labels):
            prediction = predict(neuron, inputs)
            error = target - prediction

            # Perceptron learning rule:
            # - If prediction is correct, error = 0 → no update
            # - If prediction is too low (0 when should be 1), error = +1
            #   → weights increase, making this input more likely to fire
            # - If prediction is too high (1 when should be 0), error = -1
            #   → weights decrease, making this input less likely to fire
            neuron.weights = vector.add(
                neuron.weights,
                vector.scale(scalar.multiply(learning_rate, error), inputs),
            )
            neuron.bias = scalar.add(neuron.bias, scalar.multiply(learning_rate, error))

            total_error = scalar.add(total_error, abs(error))

        error_history.append(total_error)

    return error_history
