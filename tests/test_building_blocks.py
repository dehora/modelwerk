"""Tests for building blocks (neuron, layers, network, gradients)."""

from modelwerk.primitives.activations import step, sigmoid, relu
from modelwerk.primitives.random import create_rng
from modelwerk.building_blocks.neuron import Neuron, create_neuron, forward


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
