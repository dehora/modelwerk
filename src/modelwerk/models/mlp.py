"""Multi-Layer Perceptron (Rumelhart, Hinton & Williams, 1986).

Multiple layers of neurons trained with backpropagation.
The architecture that proved hidden layers could learn
non-linear decision boundaries.

A perceptron draws one straight line. An MLP stacks layers:

    input → [dense + activation] → [dense + activation] → output

Each hidden layer transforms the input into a new representation.
The first hidden layer might detect simple features; deeper layers
combine them. Backpropagation — the chain rule applied layer by
layer — tells each weight how much it contributed to the error,
so all layers learn simultaneously.

For XOR, one hidden layer with two neurons is enough:
    input (2) → hidden (2, sigmoid) → output (1, sigmoid)
The hidden layer learns two linear boundaries; the output layer
combines them into the non-linear XOR boundary.
"""

from modelwerk.primitives.activations import sigmoid
from modelwerk.primitives.losses import mse, mse_derivative
from modelwerk.building_blocks.network import Network, create_network, network_forward
from modelwerk.building_blocks.grad import backward, _DERIVATIVES
from modelwerk.building_blocks.optimizers import sgd_update

Vector = list[float]


def create_mlp(
    rng,
    layer_sizes: list[int],
    activation_fn=sigmoid,
    output_activation_fn=sigmoid,
) -> Network:
    """Create an MLP with the given layer sizes.

    layer_sizes: [input_dim, hidden1, ..., output_dim]
    activation_fn: activation for hidden layers (default: sigmoid)
    output_activation_fn: activation for the output layer (default: sigmoid)
    """
    num_layers = len(layer_sizes) - 1
    activations = [activation_fn] * (num_layers - 1) + [output_activation_fn]
    return create_network(rng, layer_sizes, activations)


def predict(network: Network, inputs: Vector) -> Vector:
    """Forward pass — return the network's output."""
    output, _ = network_forward(network, inputs)
    return output


def train(
    network: Network,
    data: list[Vector],
    labels: list[Vector],
    learning_rate: float = 0.5,
    epochs: int = 1000,
    loss_fn=mse,
) -> list[float]:
    """Train with backpropagation and SGD.

    This is the algorithm from the 1986 paper:
      1. Forward pass: compute output and cache intermediates
      2. Compute loss and its gradient
      3. Backward pass: propagate gradients through every layer
      4. Update: adjust weights in the direction that reduces loss

    Returns loss per epoch.
    """
    loss_deriv = _get_loss_derivative(loss_fn)
    loss_history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for inputs, targets in zip(data, labels):
            # 1. Forward
            output, cache = network_forward(network, inputs)

            # 2. Loss
            epoch_loss += loss_fn(output, targets)

            # 3. Backward
            loss_grad = loss_deriv(output, targets)
            gradients = backward(network, cache, loss_grad)

            # 4. Update
            sgd_update(network, gradients, learning_rate)

        loss_history.append(epoch_loss / len(data))

    return loss_history


def _get_loss_derivative(loss_fn):
    """Look up the derivative for a loss function."""
    if loss_fn is mse:
        return mse_derivative
    from modelwerk.primitives.losses import cross_entropy, cross_entropy_derivative
    if loss_fn is cross_entropy:
        return cross_entropy_derivative
    raise ValueError(f"Unknown loss function: {loss_fn}")
