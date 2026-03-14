"""Level 4: Pooling layers.

Max and average pooling for spatial downsampling.
Reduces dimensionality while preserving important features.

Average pooling replaces each pool_size x pool_size window with its
mean. This shrinks the spatial dimensions while keeping the channel
count unchanged. LeNet-5 used average pooling; later networks prefer
max pooling.
"""

from dataclasses import dataclass

from modelwerk.primitives import scalar
from modelwerk.primitives.matrix import tensor3d_zeros

Tensor3D = list[list[list[float]]]


@dataclass
class PoolCache:
    """Values saved during forward pass, needed for backprop."""
    in_channels: int
    in_height: int
    in_width: int


def avg_pool_forward(
    inputs: Tensor3D, pool_size: int = 2, stride: int = 2
) -> tuple[Tensor3D, PoolCache]:
    """Average pooling forward pass.

    Each output position is the mean of a pool_size x pool_size window.
    """
    channels = len(inputs)
    in_h = len(inputs[0])
    in_w = len(inputs[0][0])
    out_h = (in_h - pool_size) // stride + 1
    out_w = (in_w - pool_size) // stride + 1
    area = float(pool_size * pool_size)

    output = tensor3d_zeros(channels, out_h, out_w)

    for c in range(channels):
        for i in range(out_h):
            for j in range(out_w):
                total = 0.0
                for ph in range(pool_size):
                    for pw in range(pool_size):
                        total = scalar.add(
                            total, inputs[c][i * stride + ph][j * stride + pw]
                        )
                output[c][i][j] = scalar.multiply(total, scalar.inverse(area))

    cache = PoolCache(in_channels=channels, in_height=in_h, in_width=in_w)
    return output, cache


def avg_pool_backward(
    output_grad: Tensor3D,
    cache: PoolCache,
    pool_size: int = 2,
    stride: int = 2,
) -> Tensor3D:
    """Average pooling backward pass.

    Each gradient is distributed equally to all positions in the window.
    """
    channels = cache.in_channels
    area = float(pool_size * pool_size)
    out_h = len(output_grad[0])
    out_w = len(output_grad[0][0])

    input_grad = tensor3d_zeros(channels, cache.in_height, cache.in_width)

    for c in range(channels):
        for i in range(out_h):
            for j in range(out_w):
                distributed = scalar.multiply(
                    output_grad[c][i][j], scalar.inverse(area)
                )
                for ph in range(pool_size):
                    for pw in range(pool_size):
                        row = i * stride + ph
                        col = j * stride + pw
                        input_grad[c][row][col] = scalar.add(
                            input_grad[c][row][col], distributed
                        )

    return input_grad
