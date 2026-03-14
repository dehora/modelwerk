"""LeNet-5 (LeCun et al., 1998).

Convolutional neural network for handwritten digit recognition.
Introduces convolutions, pooling, and weight sharing.

The original LeNet-5 used 6 and 16 filters with 120+84 dense neurons.
This is a simplified version (3+6 filters, 32 dense neurons) that
trains in minutes on pure Python while preserving the architecture:

    Input:   1x28x28
    C1:      conv 5x5, 3 filters, tanh  -> 3x24x24
    S2:      avg pool 2x2               -> 3x12x12
    C3:      conv 5x5, 6 filters, tanh  -> 6x8x8
    S4:      avg pool 2x2               -> 6x4x4
    Flatten:                             -> 96
    F5:      dense 96->32, tanh
    F6:      dense 32->10, softmax
"""

from dataclasses import dataclass

from modelwerk.primitives import scalar, vector, matrix
from modelwerk.primitives.activations import tanh_, tanh_derivative, identity, softmax
from modelwerk.primitives.losses import cross_entropy
from modelwerk.primitives.matrix import tensor3d_flatten, tensor3d_reshape, tensor3d_zeros
from modelwerk.building_blocks.conv import (
    ConvLayer, ConvCache, create_conv, conv_forward, conv_backward,
)
from modelwerk.building_blocks.pool import (
    PoolCache, avg_pool_forward, avg_pool_backward,
)
from modelwerk.building_blocks.dense import (
    DenseLayer, DenseCache, create_dense, dense_forward,
)
from modelwerk.data.utils import one_hot
from modelwerk.primitives.progress import progress_bar, progress_done

Tensor3D = list[list[list[float]]]
Vector = list[float]


@dataclass
class LeNet5:
    conv1: ConvLayer
    conv2: ConvLayer
    dense1: DenseLayer
    dense2: DenseLayer


@dataclass
class LeNet5Cache:
    conv1_cache: ConvCache
    pool1_out: Tensor3D
    pool1_cache: PoolCache
    conv2_cache: ConvCache
    pool2_out: Tensor3D
    pool2_cache: PoolCache
    flat: Vector
    dense1_cache: DenseCache
    dense2_cache: DenseCache
    probs: Vector


def create_lenet5(rng) -> LeNet5:
    """Create a simplified LeNet-5."""
    conv1 = create_conv(rng, in_channels=1, out_channels=3, kernel_size=5)
    conv2 = create_conv(rng, in_channels=3, out_channels=6, kernel_size=5)
    dense1 = create_dense(rng, 96, 32)   # 6*4*4 = 96
    dense2 = create_dense(rng, 32, 10)
    return LeNet5(conv1=conv1, conv2=conv2, dense1=dense1, dense2=dense2)


def lenet5_forward(model: LeNet5, image: Tensor3D) -> tuple[Vector, LeNet5Cache]:
    """Forward pass through LeNet-5.

    Returns (probabilities, cache).
    """
    # C1: conv + tanh -> 3x24x24
    c1_out, c1_cache = conv_forward(model.conv1, image, tanh_)

    # S2: avg pool -> 3x12x12
    p1_out, p1_cache = avg_pool_forward(c1_out)

    # C3: conv + tanh -> 6x8x8
    c2_out, c2_cache = conv_forward(model.conv2, p1_out, tanh_)

    # S4: avg pool -> 6x4x4
    p2_out, p2_cache = avg_pool_forward(c2_out)

    # Flatten -> 96
    flat = tensor3d_flatten(p2_out)

    # F5: dense + tanh -> 32
    d1_out, d1_cache = dense_forward(model.dense1, flat, tanh_)

    # F6: dense (identity) -> 10 raw logits, then softmax
    logits, d2_cache = dense_forward(model.dense2, d1_out, identity)
    probs = softmax(logits)

    cache = LeNet5Cache(
        conv1_cache=c1_cache,
        pool1_out=p1_out,
        pool1_cache=p1_cache,
        conv2_cache=c2_cache,
        pool2_out=p2_out,
        pool2_cache=p2_cache,
        flat=flat,
        dense1_cache=d1_cache,
        dense2_cache=d2_cache,
        probs=probs,
    )
    return probs, cache


def lenet5_backward(model: LeNet5, cache: LeNet5Cache, target: Vector) -> dict:
    """Backward pass through LeNet-5.

    Uses combined softmax + cross-entropy gradient: probs - target.
    Returns dict of all gradients.
    """
    # Combined softmax + cross-entropy derivative: dL/d_logits = probs - target
    d_logits = vector.subtract(cache.probs, target)

    # F6 backward (identity activation, derivative = 1)
    d2_delta = d_logits  # identity_derivative is 1, so delta = d_logits
    d2_weight_grads = matrix.outer(d2_delta, cache.dense1_cache.a)
    d2_bias_grads = list(d2_delta)
    d_dense1_out = matrix.mat_vec(matrix.transpose(model.dense2.weights), d2_delta)

    # F5 backward (tanh activation)
    f_prime = vector.apply(tanh_derivative, cache.dense1_cache.z)
    d1_delta = vector.elementwise(scalar.multiply, d_dense1_out, f_prime)
    d1_weight_grads = matrix.outer(d1_delta, cache.dense1_cache.inputs)
    d1_bias_grads = list(d1_delta)
    d_flat = matrix.mat_vec(matrix.transpose(model.dense1.weights), d1_delta)

    # Reshape flat gradient back to 6x4x4
    d_pool2 = tensor3d_reshape(d_flat, 6, 4, 4)

    # S4 backward
    d_conv2_out = avg_pool_backward(d_pool2, cache.pool2_cache)

    # C3 backward
    d_pool1, c2_filter_grads, c2_bias_grads = conv_backward(
        model.conv2, cache.conv2_cache, d_conv2_out, tanh_derivative
    )

    # S2 backward
    d_conv1_out = avg_pool_backward(d_pool1, cache.pool1_cache)

    # C1 backward
    _, c1_filter_grads, c1_bias_grads = conv_backward(
        model.conv1, cache.conv1_cache, d_conv1_out, tanh_derivative
    )

    return {
        "conv1_filter_grads": c1_filter_grads,
        "conv1_bias_grads": c1_bias_grads,
        "conv2_filter_grads": c2_filter_grads,
        "conv2_bias_grads": c2_bias_grads,
        "dense1_weight_grads": d1_weight_grads,
        "dense1_bias_grads": d1_bias_grads,
        "dense2_weight_grads": d2_weight_grads,
        "dense2_bias_grads": d2_bias_grads,
    }


def _update_conv(layer: ConvLayer, filter_grads: list[Tensor3D], bias_grads: Vector, lr: float):
    """Update conv layer parameters with SGD."""
    for f in range(len(layer.filters)):
        layer.biases[f] = scalar.subtract(
            layer.biases[f], scalar.multiply(lr, bias_grads[f])
        )
        for c in range(len(layer.filters[f])):
            for kh in range(len(layer.filters[f][c])):
                for kw in range(len(layer.filters[f][c][kh])):
                    layer.filters[f][c][kh][kw] = scalar.subtract(
                        layer.filters[f][c][kh][kw],
                        scalar.multiply(lr, filter_grads[f][c][kh][kw]),
                    )


def lenet5_sgd_update(model: LeNet5, grads: dict, learning_rate: float):
    """Update all parameters using SGD. Modifies model in place."""
    _update_conv(model.conv1, grads["conv1_filter_grads"], grads["conv1_bias_grads"], learning_rate)
    _update_conv(model.conv2, grads["conv2_filter_grads"], grads["conv2_bias_grads"], learning_rate)

    # Dense layers: W = W - lr * dW, b = b - lr * db
    model.dense1.weights = matrix.add(
        model.dense1.weights, matrix.scale(-learning_rate, grads["dense1_weight_grads"])
    )
    model.dense1.biases = vector.add(
        model.dense1.biases, vector.scale(-learning_rate, grads["dense1_bias_grads"])
    )
    model.dense2.weights = matrix.add(
        model.dense2.weights, matrix.scale(-learning_rate, grads["dense2_weight_grads"])
    )
    model.dense2.biases = vector.add(
        model.dense2.biases, vector.scale(-learning_rate, grads["dense2_bias_grads"])
    )


def predict(model: LeNet5, image: Tensor3D) -> int:
    """Return predicted digit (0-9)."""
    probs, _ = lenet5_forward(model, image)
    return max(range(len(probs)), key=lambda i: probs[i])


def train(
    model: LeNet5,
    images: list[Tensor3D],
    labels: list[int],
    learning_rate: float = 0.01,
    epochs: int = 3,
) -> tuple[list[float], list[float]]:
    """Train LeNet-5 on MNIST images.

    Returns (loss_history, accuracy_history) per epoch.
    """
    loss_history: list[float] = []
    accuracy_history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0

        for image, label in zip(images, labels):
            target = one_hot(label)

            # Forward
            probs, cache = lenet5_forward(model, image)

            # Loss
            epoch_loss += cross_entropy(probs, target)

            # Accuracy
            pred = max(range(len(probs)), key=lambda i: probs[i])
            if pred == label:
                correct += 1

            # Backward
            grads = lenet5_backward(model, cache, target)

            # Update
            lenet5_sgd_update(model, grads, learning_rate)

        avg_loss = epoch_loss / len(images)
        accuracy = correct / len(images)
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        progress_bar(epoch + 1, epochs, avg_loss)

    progress_done()
    return loss_history, accuracy_history
