"""Level 6: Optimizers.

Parameter update rules — SGD and SGD with momentum.
Takes gradients and adjusts weights to minimize loss.

SGD (stochastic gradient descent) is the simplest optimizer:
move each weight in the direction that reduces the loss, scaled
by the learning rate.

    w = w - lr * dL/dw

Momentum adds a "velocity" that accumulates past gradients,
smoothing out noisy updates and helping escape shallow local minima:

    v = momentum * v - lr * dL/dw
    w = w + v
"""

from modelwerk.primitives import vector, matrix
from modelwerk.building_blocks.grad import LayerGradients

Vector = list[float]
Matrix = list[list[float]]


def sgd_update(network, gradients: list[LayerGradients], learning_rate: float):
    """Update weights and biases using vanilla SGD.

    Modifies the network in place.
    """
    for layer, grads in zip(network.layers, gradients):
        # W = W - lr * dW
        layer.weights = matrix.add(
            layer.weights,
            matrix.scale(-learning_rate, grads.weight_grads),
        )
        # b = b - lr * db
        layer.biases = vector.add(
            layer.biases,
            vector.scale(-learning_rate, grads.bias_grads),
        )


def sgd_momentum_update(
    network,
    gradients: list[LayerGradients],
    velocities: list[LayerGradients],
    learning_rate: float,
    momentum: float = 0.9,
) -> list[LayerGradients]:
    """Update weights using SGD with momentum.

    Modifies the network in place. Returns updated velocities
    for the next step.
    """
    new_velocities = []

    for layer, grads, vel in zip(network.layers, gradients, velocities):
        # v_w = momentum * v_w - lr * dW
        new_weight_vel = matrix.add(
            matrix.scale(momentum, vel.weight_grads),
            matrix.scale(-learning_rate, grads.weight_grads),
        )
        # v_b = momentum * v_b - lr * db
        new_bias_vel = vector.add(
            vector.scale(momentum, vel.bias_grads),
            vector.scale(-learning_rate, grads.bias_grads),
        )

        # W = W + v_w
        layer.weights = matrix.add(layer.weights, new_weight_vel)
        # b = b + v_b
        layer.biases = vector.add(layer.biases, new_bias_vel)

        new_velocities.append(LayerGradients(
            weight_grads=new_weight_vel,
            bias_grads=new_bias_vel,
        ))

    return new_velocities
