"""Tests for building blocks (neuron, layers, network, gradients, conv, pool)."""

from modelwerk.primitives.activations import step, sigmoid, relu, tanh_
from modelwerk.primitives.losses import mse
from modelwerk.primitives.random import create_rng
from modelwerk.primitives.matrix import tensor3d_zeros
from modelwerk.building_blocks.neuron import Neuron, create_neuron, forward
from modelwerk.building_blocks.dense import DenseLayer, create_dense, dense_forward
from modelwerk.building_blocks.network import create_network, network_forward
from modelwerk.building_blocks.grad import backward, numerical_gradient_check
from modelwerk.building_blocks.optimizers import sgd_update
from modelwerk.building_blocks.conv import create_conv, conv_forward, conv_backward, ConvLayer
from modelwerk.building_blocks.pool import avg_pool_forward, avg_pool_backward


class TestNeuron:
    def test_create_neuron(self):
        rng = create_rng(42)
        n = create_neuron(rng, 3)
        assert len(n.weights) == 3
        assert n.bias == 0.0

    def test_forward_step(self):
        # Hand-computed: dot([1, 1], [1, 1]) + 0 = 2, step(2) = 1
        n = Neuron(weights=[1.0, 1.0], bias=0.0)
        assert forward(n, [1.0, 1.0], step) == 1.0

    def test_forward_step_negative(self):
        # dot([1, 1], [-1, -1]) + 0 = -2, step(-2) = 0
        n = Neuron(weights=[1.0, 1.0], bias=0.0)
        assert forward(n, [-1.0, -1.0], step) == 0.0

    def test_forward_sigmoid(self):
        # dot([0, 0], [1, 1]) + 0 = 0, sigmoid(0) = 0.5
        n = Neuron(weights=[0.0, 0.0], bias=0.0)
        assert abs(forward(n, [1.0, 1.0], sigmoid) - 0.5) < 1e-10

    def test_forward_with_bias(self):
        n = Neuron(weights=[1.0, 1.0], bias=-3.0)
        # dot([1,1],[1,1]) + (-3) = -1, step(-1) = 0
        assert forward(n, [1.0, 1.0], step) == 0.0

    def test_reproducibility(self):
        n1 = create_neuron(create_rng(42), 5)
        n2 = create_neuron(create_rng(42), 5)
        assert n1.weights == n2.weights


class TestDenseLayer:
    def test_create_dense_shape(self):
        rng = create_rng(42)
        layer = create_dense(rng, 3, 2)
        assert len(layer.weights) == 2     # 2 outputs
        assert len(layer.weights[0]) == 3  # 3 inputs
        assert len(layer.biases) == 2

    def test_forward_output_shape(self):
        rng = create_rng(42)
        layer = create_dense(rng, 3, 2)
        output, cache = dense_forward(layer, [1.0, 2.0, 3.0], sigmoid)
        assert len(output) == 2

    def test_forward_cache(self):
        rng = create_rng(42)
        layer = create_dense(rng, 2, 1)
        output, cache = dense_forward(layer, [1.0, 0.5], sigmoid)
        assert cache.inputs == [1.0, 0.5]
        assert len(cache.z) == 1
        assert len(cache.a) == 1
        assert cache.a == output

    def test_hand_computed(self):
        # W = [[1, 2]], b = [0.5], input = [1, 1]
        # z = 1*1 + 2*1 + 0.5 = 3.5, sigmoid(3.5) ≈ 0.9706
        layer = DenseLayer(weights=[[1.0, 2.0]], biases=[0.5])
        output, _ = dense_forward(layer, [1.0, 1.0], sigmoid)
        assert abs(output[0] - 0.9706) < 0.001

    def test_zero_biases(self):
        rng = create_rng(42)
        layer = create_dense(rng, 2, 3)
        assert all(b == 0.0 for b in layer.biases)


class TestNetwork:
    def test_create_network_shapes(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 3, 1], [sigmoid, sigmoid])
        assert len(net.layers) == 2
        assert len(net.layers[0].weights) == 3     # 3 outputs
        assert len(net.layers[0].weights[0]) == 2  # 2 inputs
        assert len(net.layers[1].weights) == 1     # 1 output
        assert len(net.layers[1].weights[0]) == 3  # 3 inputs

    def test_forward_output_shape(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 4, 1], [sigmoid, sigmoid])
        output, cache = network_forward(net, [1.0, 0.0])
        assert len(output) == 1
        assert len(cache.layer_caches) == 2

    def test_forward_deterministic(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 3, 1], [sigmoid, sigmoid])
        out1, _ = network_forward(net, [1.0, 0.5])
        out2, _ = network_forward(net, [1.0, 0.5])
        assert out1 == out2

    def test_sigmoid_output_range(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 3, 1], [sigmoid, sigmoid])
        output, _ = network_forward(net, [1.0, 1.0])
        assert 0.0 < output[0] < 1.0


class TestGradients:
    def test_gradient_check_small_network(self):
        """Analytical gradients should match numerical gradients."""
        rng = create_rng(42)
        net = create_network(rng, [2, 3, 1], [sigmoid, sigmoid])
        max_error = numerical_gradient_check(
            net, [1.0, 0.5], [1.0], mse
        )
        assert max_error < 1e-5, f"Gradient check failed: max error {max_error}"

    def test_gradient_check_deeper_network(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 4, 3, 1], [sigmoid, sigmoid, sigmoid])
        max_error = numerical_gradient_check(
            net, [0.5, -0.3], [0.8], mse
        )
        assert max_error < 1e-5, f"Gradient check failed: max error {max_error}"

    def test_backward_returns_correct_count(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 3, 1], [sigmoid, sigmoid])
        output, cache = network_forward(net, [1.0, 0.5])
        from modelwerk.primitives.losses import mse_derivative
        loss_grad = mse_derivative(output, [1.0])
        grads = backward(net, cache, loss_grad)
        assert len(grads) == 2  # one per layer

    def test_backward_gradient_shapes(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 3, 1], [sigmoid, sigmoid])
        output, cache = network_forward(net, [1.0, 0.5])
        from modelwerk.primitives.losses import mse_derivative
        loss_grad = mse_derivative(output, [1.0])
        grads = backward(net, cache, loss_grad)
        # Layer 0: (3, 2) weights, (3,) biases
        assert len(grads[0].weight_grads) == 3
        assert len(grads[0].weight_grads[0]) == 2
        assert len(grads[0].bias_grads) == 3
        # Layer 1: (1, 3) weights, (1,) biases
        assert len(grads[1].weight_grads) == 1
        assert len(grads[1].weight_grads[0]) == 3
        assert len(grads[1].bias_grads) == 1


class TestOptimizers:
    def test_sgd_changes_weights(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 1], [sigmoid])
        w_before = [row[:] for row in net.layers[0].weights]
        output, cache = network_forward(net, [1.0, 1.0])
        from modelwerk.primitives.losses import mse_derivative
        loss_grad = mse_derivative(output, [1.0])
        grads = backward(net, cache, loss_grad)
        sgd_update(net, grads, 0.1)
        w_after = net.layers[0].weights
        assert w_before != w_after

    def test_sgd_reduces_loss(self):
        rng = create_rng(42)
        net = create_network(rng, [2, 3, 1], [sigmoid, sigmoid])
        inputs = [1.0, 0.0]
        targets = [1.0]
        from modelwerk.primitives.losses import mse_derivative
        out1, _ = network_forward(net, inputs)
        loss1 = mse(out1, targets)
        for _ in range(10):
            output, cache = network_forward(net, inputs)
            loss_grad = mse_derivative(output, targets)
            grads = backward(net, cache, loss_grad)
            sgd_update(net, grads, 0.5)
        out2, _ = network_forward(net, inputs)
        loss2 = mse(out2, targets)
        assert loss2 < loss1


class TestConvLayer:
    def test_create_conv_shape(self):
        rng = create_rng(42)
        layer = create_conv(rng, 1, 2, kernel_size=3)
        assert len(layer.filters) == 2
        assert len(layer.filters[0]) == 1
        assert len(layer.filters[0][0]) == 3
        assert len(layer.filters[0][0][0]) == 3
        assert len(layer.biases) == 2

    def test_conv_forward_output_shape(self):
        rng = create_rng(42)
        layer = create_conv(rng, 1, 2, kernel_size=3)
        inp = tensor3d_zeros(1, 5, 5)
        out, cache = conv_forward(layer, inp, tanh_)
        assert len(out) == 2
        assert len(out[0]) == 3
        assert len(out[0][0]) == 3

    def test_conv_forward_5x5_kernel(self):
        """5x5 kernel on 7x7 input gives 3x3 output."""
        rng = create_rng(42)
        layer = create_conv(rng, 1, 1, kernel_size=5)
        inp = tensor3d_zeros(1, 7, 7)
        out, _ = conv_forward(layer, inp, tanh_)
        assert len(out[0]) == 3
        assert len(out[0][0]) == 3

    def test_conv_backward_shapes(self):
        rng = create_rng(42)
        layer = create_conv(rng, 1, 2, kernel_size=3)
        inp = tensor3d_zeros(1, 5, 5)
        inp[0][2][2] = 1.0
        out, cache = conv_forward(layer, inp, tanh_)
        # output_grad same shape as output
        output_grad = tensor3d_zeros(2, 3, 3)
        output_grad[0][1][1] = 1.0
        from modelwerk.primitives.activations import tanh_derivative
        input_grad, filter_grads, bias_grads = conv_backward(
            layer, cache, output_grad, tanh_derivative
        )
        assert len(input_grad) == 1
        assert len(input_grad[0]) == 5
        assert len(input_grad[0][0]) == 5
        assert len(filter_grads) == 2
        assert len(filter_grads[0]) == 1
        assert len(filter_grads[0][0]) == 3
        assert len(bias_grads) == 2


class TestPooling:
    def test_avg_pool_output_shape(self):
        inp = tensor3d_zeros(1, 4, 4)
        out, cache = avg_pool_forward(inp, pool_size=2)
        assert len(out) == 1
        assert len(out[0]) == 2
        assert len(out[0][0]) == 2

    def test_avg_pool_values(self):
        inp = tensor3d_zeros(1, 2, 2)
        inp[0][0][0] = 1.0
        inp[0][0][1] = 2.0
        inp[0][1][0] = 3.0
        inp[0][1][1] = 4.0
        out, _ = avg_pool_forward(inp, pool_size=2)
        assert abs(out[0][0][0] - 2.5) < 1e-10

    def test_avg_pool_multichannel(self):
        inp = tensor3d_zeros(3, 4, 4)
        out, _ = avg_pool_forward(inp, pool_size=2)
        assert len(out) == 3
        assert len(out[0]) == 2

    def test_avg_pool_backward_distributes(self):
        inp = tensor3d_zeros(1, 4, 4)
        for r in range(4):
            for c in range(4):
                inp[0][r][c] = float(r * 4 + c)
        out, cache = avg_pool_forward(inp, pool_size=2)
        output_grad = tensor3d_zeros(1, 2, 2)
        output_grad[0][0][0] = 1.0
        input_grad = avg_pool_backward(output_grad, cache, pool_size=2)
        # Gradient distributed to 4 positions, each gets 0.25
        assert abs(input_grad[0][0][0] - 0.25) < 1e-10
        assert abs(input_grad[0][0][1] - 0.25) < 1e-10
        assert abs(input_grad[0][1][0] - 0.25) < 1e-10
        assert abs(input_grad[0][1][1] - 0.25) < 1e-10
        # Other positions should be 0
        assert abs(input_grad[0][2][0]) < 1e-10

    def test_avg_pool_backward_shape(self):
        inp = tensor3d_zeros(2, 6, 6)
        _, cache = avg_pool_forward(inp, pool_size=2)
        output_grad = tensor3d_zeros(2, 3, 3)
        input_grad = avg_pool_backward(output_grad, cache, pool_size=2)
        assert len(input_grad) == 2
        assert len(input_grad[0]) == 6
        assert len(input_grad[0][0]) == 6
