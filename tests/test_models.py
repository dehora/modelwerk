"""Tests for complete model architectures."""

from modelwerk.primitives.random import create_rng
from modelwerk.models.perceptron import create_perceptron, predict, train
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
