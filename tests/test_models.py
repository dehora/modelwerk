"""Tests for complete model architectures."""

from modelwerk.primitives.random import create_rng
from modelwerk.primitives.activations import sigmoid
from modelwerk.primitives.matrix import tensor3d_zeros
from modelwerk.primitives.losses import cross_entropy
from modelwerk.data.utils import one_hot
from modelwerk.models.perceptron import create_perceptron, predict, train
from modelwerk.models.mlp import create_mlp, predict as mlp_predict, train as mlp_train
from modelwerk.models.lenet5 import (
    create_lenet5, lenet5_forward, lenet5_backward, lenet5_sgd_update,
    predict as lenet_predict,
)
from modelwerk.models.transformer import (
    create_transformer_lm, transformer_forward, transformer_backward,
    transformer_sgd_update,
)
from modelwerk.data.generators import and_gate, or_gate, nand_gate, xor_gate
from modelwerk.data.text import build_vocab, prepare_sequences


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


class TestLeNet5:
    def test_create_lenet5(self):
        rng = create_rng(42)
        model = create_lenet5(rng)
        assert model.conv1 is not None
        assert model.conv2 is not None
        assert model.dense1 is not None
        assert model.dense2 is not None

    def test_forward_output_shape(self):
        rng = create_rng(42)
        model = create_lenet5(rng)
        image = tensor3d_zeros(1, 28, 28)
        probs, cache = lenet5_forward(model, image)
        assert len(probs) == 10

    def test_softmax_sums_to_one(self):
        rng = create_rng(42)
        model = create_lenet5(rng)
        image = tensor3d_zeros(1, 28, 28)
        image[0][14][14] = 1.0
        probs, _ = lenet5_forward(model, image)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_predict_returns_digit(self):
        rng = create_rng(42)
        model = create_lenet5(rng)
        image = tensor3d_zeros(1, 28, 28)
        digit = lenet_predict(model, image)
        assert 0 <= digit <= 9

    def test_loss_decreases_single_sample(self):
        rng = create_rng(42)
        model = create_lenet5(rng)
        image = tensor3d_zeros(1, 28, 28)
        image[0][14][14] = 1.0
        target = one_hot(3)

        probs1, cache1 = lenet5_forward(model, image)
        loss1 = cross_entropy(probs1, target)
        grads = lenet5_backward(model, cache1, target)
        lenet5_sgd_update(model, grads, 0.01)

        probs2, _ = lenet5_forward(model, image)
        loss2 = cross_entropy(probs2, target)
        assert loss2 < loss1, f"Loss should decrease: {loss1:.4f} -> {loss2:.4f}"

    def test_reproducibility(self):
        image = tensor3d_zeros(1, 28, 28)
        image[0][10][10] = 0.5
        m1 = create_lenet5(create_rng(42))
        m2 = create_lenet5(create_rng(42))
        p1, _ = lenet5_forward(m1, image)
        p2, _ = lenet5_forward(m2, image)
        assert p1 == p2


class TestTransformer:
    def test_create_transformer(self):
        rng = create_rng(42)
        model = create_transformer_lm(rng, vocab_size=10, d_model=8, num_heads=2,
                                       ff_dim=16, seq_len=4)
        assert model.vocab_size == 10
        assert model.d_model == 8
        assert model.seq_len == 4

    def test_forward_output_shape(self):
        rng = create_rng(42)
        model = create_transformer_lm(rng, vocab_size=10, d_model=8, num_heads=2,
                                       ff_dim=16, seq_len=4)
        probs, cache = transformer_forward(model, [0, 1, 2, 3])
        assert len(probs) == 4
        assert len(probs[0]) == 10

    def test_softmax_sums_to_one(self):
        rng = create_rng(42)
        model = create_transformer_lm(rng, vocab_size=10, d_model=8, num_heads=2,
                                       ff_dim=16, seq_len=4)
        probs, _ = transformer_forward(model, [0, 1, 2, 3])
        for p in probs:
            assert abs(sum(p) - 1.0) < 1e-5

    def test_loss_decreases(self):
        rng = create_rng(42)
        vocab_size = 10
        model = create_transformer_lm(rng, vocab_size=vocab_size, d_model=8,
                                       num_heads=2, ff_dim=16, seq_len=4)
        token_ids = [0, 1, 2, 3]
        target_ids = [1, 2, 3, 4]
        target_onehots = [one_hot(t, vocab_size) for t in target_ids]

        probs1, cache1 = transformer_forward(model, token_ids)
        loss1 = sum(cross_entropy(probs1[t], target_onehots[t]) for t in range(4)) / 4

        # Train for a few steps
        for _ in range(5):
            probs, cache = transformer_forward(model, token_ids)
            targets = [one_hot(t, vocab_size) for t in target_ids]
            grads = transformer_backward(model, cache, targets)
            transformer_sgd_update(model, grads, 0.01)

        probs2, _ = transformer_forward(model, token_ids)
        loss2 = sum(cross_entropy(probs2[t], target_onehots[t]) for t in range(4)) / 4
        assert loss2 < loss1, f"Loss should decrease: {loss1:.4f} -> {loss2:.4f}"

    def test_reproducibility(self):
        m1 = create_transformer_lm(create_rng(42), vocab_size=10, d_model=8,
                                    num_heads=2, ff_dim=16, seq_len=4)
        m2 = create_transformer_lm(create_rng(42), vocab_size=10, d_model=8,
                                    num_heads=2, ff_dim=16, seq_len=4)
        p1, _ = transformer_forward(m1, [0, 1, 2, 3])
        p2, _ = transformer_forward(m2, [0, 1, 2, 3])
        assert p1 == p2

    def test_vocab_and_sequences(self):
        text = "hello world"
        char_to_id, id_to_char = build_vocab(text)
        assert len(char_to_id) == len(set(text))
        inputs, targets = prepare_sequences(text, char_to_id, seq_len=4)
        assert len(inputs) == len(text) - 4
        assert len(inputs[0]) == 4
        # Target is shifted by 1
        assert inputs[0][1:] == targets[0][:3]
