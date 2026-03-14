"""Level 4: Convolutional layer.

2D cross-correlation with learnable filters.
Used in LeNet-5 for spatial feature extraction.

A conv layer slides small filters across the input, computing a dot
product at each position. Each filter detects a spatial pattern (edge,
corner, curve) regardless of where it appears — this is weight sharing.

    filters: (num_filters, in_channels, kH, kW) — learnable
    biases:  (num_filters,) — one per filter
    input:   (in_channels, H, W)
    output:  (num_filters, H-kH+1, W-kW+1)
"""

from dataclasses import dataclass

from modelwerk.primitives import scalar
from modelwerk.primitives.random import create_rng
from modelwerk.primitives.matrix import tensor3d_zeros

Tensor3D = list[list[list[float]]]
Vector = list[float]


@dataclass
class ConvLayer:
    filters: list[Tensor3D]  # num_filters x in_channels x kH x kW
    biases: Vector            # num_filters


@dataclass
class ConvCache:
    """Values saved during forward pass, needed for backprop."""
    inputs: Tensor3D          # input to this layer
    z: Tensor3D               # pre-activation output
    a: Tensor3D               # post-activation output


def create_conv(rng, in_channels: int, out_channels: int, kernel_size: int = 5) -> ConvLayer:
    """Create a conv layer with Xavier-initialized filters and zero biases."""
    fan_in = in_channels * kernel_size * kernel_size
    fan_out = out_channels * kernel_size * kernel_size
    limit = (6.0 / (fan_in + fan_out)) ** 0.5

    filters = []
    for _ in range(out_channels):
        f = []
        for _ in range(in_channels):
            channel = []
            for _ in range(kernel_size):
                row = [rng.uniform(-limit, limit) for _ in range(kernel_size)]
                channel.append(row)
            f.append(channel)
        filters.append(f)

    biases = [0.0] * out_channels
    return ConvLayer(filters=filters, biases=biases)


def conv_forward(
    layer: ConvLayer, inputs: Tensor3D, activation_fn
) -> tuple[Tensor3D, ConvCache]:
    """2D cross-correlation forward pass.

    For each filter f, at each output position (i, j):
        z[f][i][j] = sum over (c, kh, kw) of
            filter[f][c][kh][kw] * input[c][i+kh][j+kw] + bias[f]
        a[f][i][j] = activation_fn(z[f][i][j])
    """
    in_channels = len(inputs)
    in_h = len(inputs[0])
    in_w = len(inputs[0][0])
    num_filters = len(layer.filters)
    k = len(layer.filters[0][0])  # kernel size
    out_h = in_h - k + 1
    out_w = in_w - k + 1

    z = tensor3d_zeros(num_filters, out_h, out_w)
    a = tensor3d_zeros(num_filters, out_h, out_w)

    for f in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                val = layer.biases[f]
                for c in range(in_channels):
                    for kh in range(k):
                        for kw in range(k):
                            val = scalar.add(
                                val,
                                scalar.multiply(
                                    layer.filters[f][c][kh][kw],
                                    inputs[c][i + kh][j + kw],
                                ),
                            )
                z[f][i][j] = val
                a[f][i][j] = activation_fn(val)

    cache = ConvCache(inputs=inputs, z=z, a=a)
    return a, cache


def conv_backward(
    layer: ConvLayer,
    cache: ConvCache,
    output_grad: Tensor3D,
    deriv_fn,
) -> tuple[Tensor3D, list[Tensor3D], Vector]:
    """Backward pass for conv layer.

    Returns (input_grad, filter_grads, bias_grads).
    """
    num_filters = len(layer.filters)
    in_channels = len(cache.inputs)
    in_h = len(cache.inputs[0])
    in_w = len(cache.inputs[0][0])
    k = len(layer.filters[0][0])
    out_h = len(output_grad[0])
    out_w = len(output_grad[0][0])

    # Compute delta = output_grad * activation_derivative(z)
    delta = tensor3d_zeros(num_filters, out_h, out_w)
    for f in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                delta[f][i][j] = scalar.multiply(
                    output_grad[f][i][j], deriv_fn(cache.z[f][i][j])
                )

    # Bias gradients: sum of delta over spatial positions
    bias_grads = [0.0] * num_filters
    for f in range(num_filters):
        for i in range(out_h):
            for j in range(out_w):
                bias_grads[f] = scalar.add(bias_grads[f], delta[f][i][j])

    # Filter gradients: correlate delta with inputs
    filter_grads = []
    for f in range(num_filters):
        fg = []
        for c in range(in_channels):
            channel = []
            for kh in range(k):
                row = []
                for kw in range(k):
                    grad = 0.0
                    for i in range(out_h):
                        for j in range(out_w):
                            grad = scalar.add(
                                grad,
                                scalar.multiply(
                                    delta[f][i][j],
                                    cache.inputs[c][i + kh][j + kw],
                                ),
                            )
                    row.append(grad)
                channel.append(row)
            fg.append(channel)
        filter_grads.append(fg)

    # Input gradients: "full" convolution of delta with rotated filters
    input_grad = tensor3d_zeros(in_channels, in_h, in_w)
    for c in range(in_channels):
        for i in range(in_h):
            for j in range(in_w):
                val = 0.0
                for f in range(num_filters):
                    for kh in range(k):
                        for kw in range(k):
                            # delta position = (i - kh, j - kw)
                            di = i - kh
                            dj = j - kw
                            if 0 <= di < out_h and 0 <= dj < out_w:
                                val = scalar.add(
                                    val,
                                    scalar.multiply(
                                        delta[f][di][dj],
                                        layer.filters[f][c][kh][kw],
                                    ),
                                )
                input_grad[c][i][j] = val

    return input_grad, filter_grads, bias_grads
