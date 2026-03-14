"""Tests for complete model architectures."""

from modelwerk.primitives.random import create_rng
from modelwerk.primitives.activations import sigmoid
from modelwerk.models.perceptron import create_perceptron, predict, train
from modelwerk.models.mlp import create_mlp, predict as mlp_predict, train as mlp_train
from modelwerk.data.generators import and_gate, or_gate, nand_gate, xor_gate


class TestPerceptron:
    def test_learns_and(self):
        rng = create_rng(42)
        p = create_perceptron(rng, 2)
        data, labels = and_gate()
        train(p, data, labels, learning_rate=0.1, epochs=20)
        for inputs, expected in zip(data, labels):
            assert predict(p, inputs) == int(expected)

    def test_learns_or(self):
        rng = create_rng(42)
        p = create_perceptron(rng, 2)
        data, labels = or_gate()
        train(p, data, labels, learning_rate=0.1, epochs=20)
        for inputs, expected in zip(data, labels):
            assert predict(p, inputs) == int(expected)

    def test_learns_nand(self):
        rng = create_rng(42)
        p = create_perceptron(rng, 2)
        data, labels = nand_gate()
        train(p, data, labels, learning_rate=0.1, epochs=20)
        for inputs, expected in zip(data, labels):
            assert predict(p, inputs) == int(expected)

    def test_fails_xor(self):
        """XOR is not linearly separable — perceptron cannot learn it perfectly."""
        rng = create_rng(42)
        p = create_perceptron(rng, 2)
        data, labels = xor_gate()
        train(p, data, labels, learning_rate=0.1, epochs=100)
        predictions = [predict(p, x) for x in data]
        # At least one prediction should be wrong
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == int(label))
        assert correct < 4, "Perceptron should not perfectly learn XOR"


class TestMLP:
    def test_learns_xor(self):
        """The whole point: MLP solves what the perceptron cannot."""
        rng = create_rng(42)
        net = create_mlp(rng, [2, 4, 1])
        data, labels = xor_gate()
        targets = [[y] for y in labels]  # wrap scalars as vectors
        mlp_train(net, data, targets, learning_rate=1.0, epochs=2000)
        for inputs, expected in zip(data, labels):
            output = mlp_predict(net, inputs)
            prediction = 1 if output[0] > 0.5 else 0
            assert prediction == int(expected), (
                f"XOR({inputs}) = {output[0]:.3f}, expected {int(expected)}"
            )

    def test_learns_and(self):
        rng = create_rng(42)
        net = create_mlp(rng, [2, 2, 1])
        data, labels = and_gate()
        targets = [[y] for y in labels]
        mlp_train(net, data, targets, learning_rate=1.0, epochs=1000)
        for inputs, expected in zip(data, labels):
            output = mlp_predict(net, inputs)
            prediction = 1 if output[0] > 0.5 else 0
            assert prediction == int(expected)

    def test_output_shape(self):
        rng = create_rng(42)
        net = create_mlp(rng, [3, 4, 2])
        output = mlp_predict(net, [1.0, 2.0, 3.0])
        assert len(output) == 2

    def test_loss_decreases(self):
        rng = create_rng(42)
        net = create_mlp(rng, [2, 4, 1])
        data, labels = xor_gate()
        targets = [[y] for y in labels]
        losses = mlp_train(net, data, targets, learning_rate=1.0, epochs=500)
        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_reproducibility(self):
        data, labels = xor_gate()
        targets = [[y] for y in labels]

        net1 = create_mlp(create_rng(42), [2, 4, 1])
        mlp_train(net1, data, targets, learning_rate=1.0, epochs=100)
        out1 = mlp_predict(net1, [1.0, 0.0])

        net2 = create_mlp(create_rng(42), [2, 4, 1])
        mlp_train(net2, data, targets, learning_rate=1.0, epochs=100)
        out2 = mlp_predict(net2, [1.0, 0.0])

        assert out1 == out2
