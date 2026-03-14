"""Level 6: Gradient computation.

Backpropagation logic for each layer type, plus numerical gradient
checking via finite differences.

Backpropagation is the chain rule applied layer by layer. Given a
loss function L and a network with layers 1..N:

    Forward:  a₀ → [W₁,b₁] → z₁ → f₁ → a₁ → ... → aₙ → L
    Backward: dL/daₙ → delta_N → ... → delta_1

For each layer l:
    delta_l  = dL/da_l * f'(z_l)          (element-wise)
    dL/dW_l  = outer(delta_l, a_{l-1})    (how much each weight contributed)
    dL/db_l  = delta_l                    (bias gradient = delta directly)
    dL/da_{l-1} = W_l^T @ delta_l         (propagate error to previous layer)

The key insight from the 1986 paper: by caching z and a during the
forward pass, we can compute all gradients in a single backward pass
— no need to retrain each layer independently.
"""

from dataclasses import dataclass

from modelwerk.primitives import scalar, vector, matrix
from modelwerk.primitives.activations import (
    sigmoid, sigmoid_derivative,
    tanh_, tanh_derivative,
    relu, relu_derivative,
)

Vector = list[float]
Matrix = list[list[float]]


# Map activation functions to their derivatives.
_DERIVATIVES = {
    sigmoid: sigmoid_derivative,
    tanh_: tanh_derivative,
    relu: relu_derivative,
}


@dataclass
class LayerGradients:
    """Gradients for a single dense layer."""
    weight_grads: Matrix  # same shape as layer.weights
    bias_grads: Vector    # same shape as layer.biases


def backward(network, cache, loss_grad: Vector) -> list[LayerGradients]:
    """Compute gradients for every layer via backpropagation.

    Args:
        network: the Network (layers + activation_fns)
        cache: NetworkCache from the forward pass
        loss_grad: dL/da for the final output (from loss derivative)

    Returns:
        list of LayerGradients, one per layer, in layer order.
    """
    gradients = []
    delta = loss_grad

    # Walk backwards through layers
    for i in range(len(network.layers) - 1, -1, -1):
        layer = network.layers[i]
        layer_cache = cache.layer_caches[i]
        activation_fn = network.activation_fns[i]
        deriv_fn = _DERIVATIVES[activation_fn]

        # delta = dL/da * f'(z), element-wise
        f_prime = vector.apply(deriv_fn, layer_cache.z)
        delta = vector.elementwise(scalar.multiply, delta, f_prime)

        # Weight gradients: outer(delta, inputs)
        weight_grads = matrix.outer(delta, layer_cache.inputs)
        bias_grads = list(delta)

        gradients.append(LayerGradients(
            weight_grads=weight_grads,
            bias_grads=bias_grads,
        ))

        # Propagate error to previous layer: W^T @ delta
        delta = matrix.mat_vec(matrix.transpose(layer.weights), delta)

    # Reverse so gradients[i] matches network.layers[i]
    gradients.reverse()
    return gradients


def numerical_gradient_check(
    network,
    inputs: Vector,
    targets: Vector,
    loss_fn,
    epsilon: float = 1e-5,
) -> float:
    """Check analytical gradients against numerical (finite-difference) gradients.

    Perturb each weight by +/- epsilon, measure the change in loss,
    and compare to the analytical gradient. Returns the maximum
    relative error across all weights.

    This is slow (two forward passes per weight) but catches bugs
    in the backward pass. Use on small networks only.
    """
    from modelwerk.building_blocks.network import network_forward

    # Get analytical gradients
    loss_derivative = _get_loss_derivative(loss_fn)
    output, cache = network_forward(network, inputs)
    loss_grad = loss_derivative(output, targets)
    analytical = backward(network, cache, loss_grad)

    max_error = 0.0

    for layer_idx in range(len(network.layers)):
        layer = network.layers[layer_idx]
        grads = analytical[layer_idx]

        # Check weight gradients
        for r in range(len(layer.weights)):
            for c in range(len(layer.weights[0])):
                original = layer.weights[r][c]

                layer.weights[r][c] = original + epsilon
                out_plus, _ = network_forward(network, inputs)
                loss_plus = loss_fn(out_plus, targets)

                layer.weights[r][c] = original - epsilon
                out_minus, _ = network_forward(network, inputs)
                loss_minus = loss_fn(out_minus, targets)

                layer.weights[r][c] = original

                numerical = (loss_plus - loss_minus) / (2 * epsilon)
                anal = grads.weight_grads[r][c]

                denom = max(abs(numerical), abs(anal), 1e-8)
                error = abs(numerical - anal) / denom
                if error > max_error:
                    max_error = error

        # Check bias gradients
        for j in range(len(layer.biases)):
            original = layer.biases[j]

            layer.biases[j] = original + epsilon
            out_plus, _ = network_forward(network, inputs)
            loss_plus = loss_fn(out_plus, targets)

            layer.biases[j] = original - epsilon
            out_minus, _ = network_forward(network, inputs)
            loss_minus = loss_fn(out_minus, targets)

            layer.biases[j] = original

            numerical = (loss_plus - loss_minus) / (2 * epsilon)
            anal = grads.bias_grads[j]

            denom = max(abs(numerical), abs(anal), 1e-8)
            error = abs(numerical - anal) / denom
            if error > max_error:
                max_error = error

    return max_error


def _get_loss_derivative(loss_fn):
    """Look up the derivative for a loss function."""
    from modelwerk.primitives.losses import mse, mse_derivative, cross_entropy, cross_entropy_derivative
    if loss_fn is mse:
        return mse_derivative
    if loss_fn is cross_entropy:
        return cross_entropy_derivative
    raise ValueError(f"Unknown loss function: {loss_fn}")
