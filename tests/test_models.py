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
    transformer_sgd_update, generate,
)
from modelwerk.models.ctm import (
    create_ctm, ctm_forward, ctm_loss, ctm_backward, ctm_sgd_update,
    ctm_adamw_update, create_adamw_state, _adamw_update_scalar, _clip_grad_norm,
    predict as ctm_predict, train as ctm_train,
    _sync_update,
)
from modelwerk.models.mamba import (
    create_mamba_lm, mamba_forward, mamba_backward, _create_adamw_state,
    _mamba_adamw_update, _clip_grad_norm as mamba_clip_grad_norm,
    predict as mamba_predict, train as mamba_train,
)
from modelwerk.primitives.activations import softplus, softplus_derivative
from modelwerk.data.generators import and_gate, or_gate, nand_gate, xor_gate, parity, selective_copying
from modelwerk.data.text import build_vocab, prepare_sequences
from modelwerk.primitives import vector


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

    def test_generate_greedy(self):
        """Greedy generation should produce deterministic output."""
        rng = create_rng(42)
        vocab_size = 5
        model = create_transformer_lm(rng, vocab_size=vocab_size, d_model=8,
                                       num_heads=2, ff_dim=16, seq_len=4)
        id_to_char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        text1, attn1 = generate(model, [0, 1, 2, 3], length=5, id_to_char=id_to_char)
        text2, attn2 = generate(model, [0, 1, 2, 3], length=5, id_to_char=id_to_char)
        # Greedy should be deterministic
        assert text1 == text2
        # Output length = prompt (4) + generated (5) = 9 chars
        assert len(text1) == 9
        # Should return attention weights
        assert len(attn1) > 0

    def test_generate_with_sampling(self):
        """Sampled generation with same seed should be reproducible."""
        rng = create_rng(42)
        vocab_size = 5
        model = create_transformer_lm(rng, vocab_size=vocab_size, d_model=8,
                                       num_heads=2, ff_dim=16, seq_len=4)
        id_to_char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        text1, _ = generate(model, [0, 1, 2, 3], length=5, id_to_char=id_to_char,
                            temperature=0.8, rng=create_rng(123))
        text2, _ = generate(model, [0, 1, 2, 3], length=5, id_to_char=id_to_char,
                            temperature=0.8, rng=create_rng(123))
        assert text1 == text2
        assert len(text1) == 9

    def test_generate_temperature(self):
        """Temperature should affect the output distribution."""
        rng = create_rng(42)
        vocab_size = 5
        model = create_transformer_lm(rng, vocab_size=vocab_size, d_model=8,
                                       num_heads=2, ff_dim=16, seq_len=4)
        id_to_char = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
        # Very low temperature should behave like greedy
        text_cold, _ = generate(model, [0, 1, 2, 3], length=5, id_to_char=id_to_char,
                                temperature=0.01, rng=create_rng(99))
        text_greedy, _ = generate(model, [0, 1, 2, 3], length=5, id_to_char=id_to_char)
        # With very low temperature, sampling should approximate greedy
        assert text_cold == text_greedy


class TestCTM:
    """Small CTM config used across tests."""

    @staticmethod
    def _small_model(seed=42):
        rng = create_rng(seed)
        return create_ctm(
            rng, d_model=16, d_input=8, d_embed=4,
            d_hidden_syn=16, d_hidden_nlm=4,
            M=3, T=3, num_classes=2, seq_len=4,
            J_out=4, J_action=4,
        )

    def test_create_ctm(self):
        model = self._small_model()
        assert model.d_model == 16
        assert model.T == 3
        assert model.num_classes == 2
        assert len(model.synapse.W1) == 16  # d_hidden_syn rows
        assert len(model.nlm.W1) == 16  # one per neuron

    def test_forward_output_shape(self):
        model = self._small_model()
        per_tick_probs, cache = ctm_forward(model, [1.0, -1.0, 1.0, -1.0])
        assert len(per_tick_probs) == 3  # T ticks
        assert len(per_tick_probs[0]) == 2  # num_classes
        # Probabilities sum to 1
        for probs in per_tick_probs:
            assert abs(sum(probs) - 1.0) < 1e-6

    def test_loss_decreases(self):
        model = self._small_model()
        input_seq = [1.0, -1.0, 1.0, -1.0]
        target = one_hot(1, 2)

        # Get initial loss
        model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
        model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
        model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
        model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)
        probs1, cache1 = ctm_forward(model, input_seq)
        loss1, _, _, _, _ = ctm_loss(probs1, target)

        # Train a few steps
        for _ in range(5):
            model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
            model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
            model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
            model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)
            probs, cache = ctm_forward(model, input_seq)
            grads = ctm_backward(model, cache, target)
            ctm_sgd_update(model, grads, 0.01)

        # Check loss decreased
        model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
        model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
        model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
        model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)
        probs2, _ = ctm_forward(model, input_seq)
        loss2, _, _, _, _ = ctm_loss(probs2, target)
        assert loss2 < loss1, f"Loss should decrease: {loss1:.4f} -> {loss2:.4f}"

    def test_reproducibility(self):
        input_seq = [1.0, -1.0, 1.0, -1.0]
        m1 = self._small_model(42)
        m2 = self._small_model(42)
        p1, _ = ctm_forward(m1, input_seq)
        p2, _ = ctm_forward(m2, input_seq)
        assert p1 == p2

    def test_parity_generator(self):
        rng = create_rng(42)
        inputs, targets = parity(rng, seq_len=4, n_samples=5)
        assert len(inputs) == 5
        assert len(targets) == 5
        assert len(inputs[0]) == 4
        assert len(targets[0]) == 4
        # Check parity logic: product of ±1 values
        for seq, tgt in zip(inputs, targets):
            product = 1.0
            for val, expected in zip(seq, tgt):
                product *= val
                assert expected == (1.0 if product > 0 else 0.0)

    def test_sync_recursive_matches_naive(self):
        """Verify recursive sync matches direct computation over multiple steps."""
        from modelwerk.models.ctm import SyncState, SyncPairs
        import math

        pairs = SyncPairs(left_indices=[0, 1], right_indices=[2, 3], n_pairs=4)
        state = SyncState(
            alpha=vector.zeros(4),
            beta=vector.zeros(4),
            decay_rates=[0.1, 0.2, 0.3, 0.0],
        )

        z_sequence = [
            [0.5, -0.3, 0.8, 0.1],
            [0.2, 0.7, -0.4, 0.6],
            [-0.1, 0.4, 0.3, -0.5],
        ]

        # Run recursive updates
        all_z_left = []
        all_z_right = []
        for z in z_sequence:
            S, new_a, new_b, zl, zr = _sync_update(state, z, pairs)
            state.alpha = new_a
            state.beta = new_b
            all_z_left.append(zl)
            all_z_right.append(zr)

        # Verify against naive computation for the last step
        # After 3 steps, alpha should be sum of decayed z_i*z_j products
        for p_idx in range(4):
            r = [0.1, 0.2, 0.3, 0.0][p_idx]
            decay = math.exp(-r)
            # Manual 3-step unroll
            a0 = all_z_left[0][p_idx] * all_z_right[0][p_idx]
            b0 = 1.0
            a1 = decay * a0 + all_z_left[1][p_idx] * all_z_right[1][p_idx]
            b1 = decay * b0 + 1.0
            a2 = decay * a1 + all_z_left[2][p_idx] * all_z_right[2][p_idx]
            b2 = decay * b1 + 1.0
            assert abs(state.alpha[p_idx] - a2) < 1e-10
            assert abs(state.beta[p_idx] - b2) < 1e-10

    # --- P1: AdamW tests ---

    def test_adamw_update_scalar_momentum(self):
        """Verify AdamW scalar update math against hand computation."""
        import math
        param, grad = 0.5, 0.1
        m, v = 0.0, 0.0
        lr, beta1, beta2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.01
        bc1 = 1.0 - beta1 ** 1  # 0.1
        bc2 = 1.0 - beta2 ** 1  # 0.001

        new_param, new_m, new_v = _adamw_update_scalar(
            param, grad, m, v, lr, beta1, beta2, eps, wd, bc1, bc2,
        )

        expected_m = (1.0 - beta1) * grad  # 0.01
        expected_v = (1.0 - beta2) * grad * grad  # 0.0001 * 0.01 = 0.00001
        m_hat = expected_m / bc1  # 0.1
        v_hat = expected_v / bc2  # 0.01
        expected_param = param * (1.0 - lr * wd) - lr * m_hat / (math.sqrt(v_hat) + eps)

        assert abs(new_m - expected_m) < 1e-10
        assert abs(new_v - expected_v) < 1e-10
        assert abs(new_param - expected_param) < 1e-10

    def test_adamw_update_scalar_zero_grad(self):
        """With zero gradient, only weight decay changes the parameter."""
        param, grad = 1.0, 0.0
        lr, wd = 0.01, 0.01
        bc1 = 1.0 - 0.9  # step=1
        bc2 = 1.0 - 0.999

        new_param, new_m, new_v = _adamw_update_scalar(
            param, grad, 0.0, 0.0, lr, 0.9, 0.999, 1e-8, wd, bc1, bc2,
        )

        assert new_m == 0.0
        assert new_v == 0.0
        assert abs(new_param - param * (1.0 - lr * wd)) < 1e-10

    def test_create_adamw_state_shapes(self):
        """AdamW state has matching keys and shapes, all zeros."""
        model = self._small_model()
        input_seq = [1.0, -1.0, 1.0, -1.0]
        target = one_hot(1, 2)

        model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
        model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
        model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
        model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)

        probs, cache = ctm_forward(model, input_seq)
        grads = ctm_backward(model, cache, target)
        m_state, v_state = create_adamw_state(grads)

        assert set(m_state.keys()) == set(grads.keys())
        assert set(v_state.keys()) == set(grads.keys())

        # Vector shape
        assert len(m_state["d_b_out"]) == len(grads["d_b_out"])
        assert all(x == 0.0 for x in m_state["d_b_out"])

        # Matrix shape
        assert len(m_state["d_W_out"]) == len(grads["d_W_out"])
        assert len(m_state["d_W_out"][0]) == len(grads["d_W_out"][0])

        # list[Matrix] shape (NLMs)
        assert len(m_state["d_nlm_W1"]) == len(grads["d_nlm_W1"])
        assert len(m_state["d_nlm_W1"][0]) == len(grads["d_nlm_W1"][0])
        assert len(m_state["d_nlm_W1"][0][0]) == len(grads["d_nlm_W1"][0][0])

    def test_loss_decreases_adamw(self):
        """AdamW training reduces loss over epochs."""
        rng = create_rng(42)
        model = self._small_model()
        inputs, targets = parity(rng, seq_len=4, n_samples=3)

        losses = ctm_train(
            model, inputs, targets,
            learning_rate=0.001, epochs=5, optimizer="adamw",
        )
        assert losses[-1] < losses[0], (
            f"AdamW loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    # --- P2: Gradient clipping tests ---

    def test_clip_grad_norm_clips(self):
        """Gradients are scaled when norm exceeds max_norm."""
        grads = {"vec": [3.0, 4.0]}  # norm = 5.0
        original_norm = _clip_grad_norm(grads, 2.5)

        assert abs(original_norm - 5.0) < 1e-10
        assert abs(grads["vec"][0] - 1.5) < 1e-10  # 3.0 * 2.5/5.0
        assert abs(grads["vec"][1] - 2.0) < 1e-10  # 4.0 * 2.5/5.0

    def test_clip_grad_norm_noop(self):
        """Gradients are unchanged when norm is below max_norm."""
        import math
        grads = {"vec": [0.1, 0.2]}
        original_norm = _clip_grad_norm(grads, 10.0)

        assert abs(original_norm - math.sqrt(0.01 + 0.04)) < 1e-10
        assert grads["vec"][0] == 0.1
        assert grads["vec"][1] == 0.2

    def test_clip_grad_norm_mixed_types(self):
        """Clipping computes norm across vectors, matrices, and list[Matrix]."""
        import math
        grads = {
            "a_vec": [3.0, 0.0],
            "b_mat": [[4.0, 0.0], [0.0, 0.0]],
            "c_nlm": [[[0.0, 0.0], [0.0, 0.0]]],
        }
        # norm = sqrt(9 + 16) = 5.0
        original_norm = _clip_grad_norm(grads, 2.5)
        scale = 2.5 / 5.0

        assert abs(original_norm - 5.0) < 1e-10
        assert abs(grads["a_vec"][0] - 3.0 * scale) < 1e-10
        assert abs(grads["b_mat"][0][0] - 4.0 * scale) < 1e-10
        assert grads["c_nlm"][0][0][0] == 0.0  # zero stays zero

    # --- P3: Existing functions ---

    def test_ctm_loss_correct_prediction(self):
        """Low loss and high certainty when predictions match target."""
        target = [0.0, 1.0]
        per_tick_probs = [
            [0.5, 0.5],
            [0.1, 0.9],
            [0.01, 0.99],
        ]
        loss, t1, t2, losses, certs = ctm_loss(per_tick_probs, target)

        assert t1 == 2  # tick 2 has lowest loss
        assert t2 == 2  # tick 2 has highest certainty
        assert loss < 0.1
        assert certs[2] > certs[0]
        assert losses[2] < losses[0]

    def test_ctm_loss_wrong_prediction(self):
        """High loss when predictions are confidently wrong."""
        target = [1.0, 0.0]
        per_tick_probs = [
            [0.5, 0.5],
            [0.1, 0.9],  # certain but wrong
            [0.4, 0.6],
        ]
        loss, t1, t2, losses, certs = ctm_loss(per_tick_probs, target)

        assert t2 == 1  # tick 1 has highest certainty (despite being wrong)
        assert losses[1] > losses[0]  # certain-wrong > uncertain
        assert loss > 0.5

    def test_predict_returns_class_index(self):
        """predict() returns an integer class index in valid range."""
        model = self._small_model()
        result = ctm_predict(model, [1.0, -1.0, 1.0, -1.0])

        assert isinstance(result, int)
        assert 0 <= result < model.num_classes

    def test_predict_deterministic(self):
        """Same model and input gives same prediction."""
        model = self._small_model()
        input_seq = [1.0, -1.0, 1.0, -1.0]
        r1 = ctm_predict(model, input_seq)
        r2 = ctm_predict(model, input_seq)
        assert r1 == r2

    def test_embed_kv_gradients_nonzero(self):
        """W_embed, W_kv, and b_kv receive non-zero gradients from backward pass.

        Regression test: these were previously dead (initialized but never
        accumulated) because the gradient chain stopped at W_attn_k/W_attn_v.
        """
        model = self._small_model()
        input_seq = [1.0, -1.0, 1.0, -1.0]
        target = one_hot(1, 2)

        model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
        model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
        model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
        model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)

        probs, cache = ctm_forward(model, input_seq)
        grads = ctm_backward(model, cache, target)

        # All three must have at least one non-zero element
        def max_abs_matrix(m):
            return max(abs(x) for row in m for x in row)

        assert max_abs_matrix(grads["d_W_embed"]) > 1e-10, "d_W_embed is all zeros"
        assert max_abs_matrix(grads["d_W_kv"]) > 1e-10, "d_W_kv is all zeros"
        assert max(abs(x) for x in grads["d_b_kv"]) > 1e-10, "d_b_kv is all zeros"

    def test_kv_gradient_finite_difference(self):
        """Finite-difference check on one W_kv element.

        Uses W_kv rather than W_embed because the certainty-based loss
        can select different ticks (t1/t2) under small perturbations,
        creating discontinuities. W_kv is smoother and more reliably
        matches. We check same sign, same order of magnitude.
        """
        model = self._small_model()
        input_seq = [1.0, -1.0, 1.0, -1.0]
        target = one_hot(1, 2)
        eps = 1e-5

        def forward_loss(m):
            m.sync_out.alpha = vector.zeros(m.pairs_out.n_pairs)
            m.sync_out.beta = vector.zeros(m.pairs_out.n_pairs)
            m.sync_action.alpha = vector.zeros(m.pairs_action.n_pairs)
            m.sync_action.beta = vector.zeros(m.pairs_action.n_pairs)
            probs, _ = ctm_forward(m, input_seq)
            loss, _, _, _, _ = ctm_loss(probs, target)
            return loss

        # Analytical gradient
        model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
        model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
        model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
        model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)
        _, cache = ctm_forward(model, input_seq)
        grads = ctm_backward(model, cache, target)
        analytical = grads["d_W_kv"][0][0]

        # Numerical gradient (central difference)
        orig = model.W_kv[0][0]
        model.W_kv[0][0] = orig + eps
        loss_plus = forward_loss(model)
        model.W_kv[0][0] = orig - eps
        loss_minus = forward_loss(model)
        model.W_kv[0][0] = orig  # restore
        numerical = (loss_plus - loss_minus) / (2 * eps)

        # Loose tolerance: the CTM loss selects ticks discretely, so
        # numerical and analytical can diverge at tick-selection boundaries
        assert abs(analytical - numerical) < 0.01, (
            f"W_kv grad mismatch: analytical={analytical:.6f}, numerical={numerical:.6f}"
        )

    def test_predict_resets_sync_state(self):
        """predict() produces the same result regardless of prior forward passes."""
        model = self._small_model()
        input_seq = [1.0, -1.0, 1.0, -1.0]

        # First prediction from clean state
        r1 = ctm_predict(model, input_seq)

        # Pollute sync state with a different forward pass
        model.sync_out.alpha = [99.0] * model.pairs_out.n_pairs
        model.sync_out.beta = [99.0] * model.pairs_out.n_pairs
        model.sync_action.alpha = [99.0] * model.pairs_action.n_pairs
        model.sync_action.beta = [99.0] * model.pairs_action.n_pairs

        # Second prediction should be identical
        r2 = ctm_predict(model, input_seq)
        assert r1 == r2, "predict() should reset sync state — got different results"

    def test_decay_rate_clamping(self):
        """Decay rates stay in [0, 15] even with extreme gradients."""
        model = self._small_model()
        input_seq = [1.0, -1.0, 1.0, -1.0]
        target = one_hot(1, 2)

        model.sync_out.alpha = vector.zeros(model.pairs_out.n_pairs)
        model.sync_out.beta = vector.zeros(model.pairs_out.n_pairs)
        model.sync_action.alpha = vector.zeros(model.pairs_action.n_pairs)
        model.sync_action.beta = vector.zeros(model.pairs_action.n_pairs)

        probs, cache = ctm_forward(model, input_seq)
        grads = ctm_backward(model, cache, target)

        # Push decay rates to extremes
        for i in range(len(grads["d_decay_out"])):
            grads["d_decay_out"][i] = -1000.0  # large negative → param goes high
        for i in range(len(grads["d_decay_action"])):
            grads["d_decay_action"][i] = 1000.0  # large positive → param goes low

        ctm_sgd_update(model, grads, lr=1.0)

        for i in range(model.pairs_out.n_pairs):
            assert model.sync_out.decay_rates[i] <= 15.0
            assert model.sync_out.decay_rates[i] >= 0.0
        for i in range(model.pairs_action.n_pairs):
            assert model.sync_action.decay_rates[i] <= 15.0
            assert model.sync_action.decay_rates[i] >= 0.0


class TestMamba:
    """Tests for Mamba selective state space model."""

    @staticmethod
    def _small_model(seed=42):
        rng = create_rng(seed)
        return create_mamba_lm(
            rng, vocab_size=8, d_model=8, d_inner=16,
            d_state=4, d_conv=2, dt_rank=4, seq_len=8,
        )

    def test_create_mamba(self):
        model = self._small_model()
        assert model.vocab_size == 8
        assert model.d_model == 8
        assert model.d_inner == 16
        assert model.d_state == 4
        assert len(model.embedding) == 8
        assert len(model.embedding[0]) == 8
        assert len(model.block.in_proj) == 32  # 2*d_inner
        assert len(model.block.ssm.A_log) == 16  # d_inner
        assert len(model.block.ssm.A_log[0]) == 4  # d_state

    def test_forward_output_shape(self):
        model = self._small_model()
        probs, cache = mamba_forward(model, [0, 1, 2, 3])
        assert len(probs) == 4
        assert len(probs[0]) == 8  # vocab_size

    def test_softmax_sums_to_one(self):
        model = self._small_model()
        probs, _ = mamba_forward(model, [0, 1, 2, 3, 5, 6, 7, 2])
        for p in probs:
            assert abs(sum(p) - 1.0) < 1e-5

    def test_loss_decreases(self):
        model = self._small_model()
        inp = [0, 3, 0, 0, 2, 1, 3, 2]
        tgt = [0, 0, 0, 0, 0, 0, 3, 2]
        target_onehots = [one_hot(t, 8) for t in tgt]

        probs1, cache1 = mamba_forward(model, inp)
        loss1 = sum(cross_entropy(probs1[t], target_onehots[t]) for t in range(8)) / 8

        for _ in range(5):
            probs, cache = mamba_forward(model, inp)
            grads = mamba_backward(model, cache, target_onehots)
            mamba_clip_grad_norm(grads, 1.0)
            step = 1
            m_state, v_state = _create_adamw_state(grads)
            _mamba_adamw_update(model, grads, m_state, v_state, 0.001, step)

        probs2, _ = mamba_forward(model, inp)
        loss2 = sum(cross_entropy(probs2[t], target_onehots[t]) for t in range(8)) / 8
        assert loss2 < loss1, f"Loss should decrease: {loss1:.4f} -> {loss2:.4f}"

    def test_reproducibility(self):
        m1 = self._small_model(42)
        m2 = self._small_model(42)
        p1, _ = mamba_forward(m1, [0, 1, 2, 3])
        p2, _ = mamba_forward(m2, [0, 1, 2, 3])
        assert p1 == p2

    def test_selective_copying_generator(self):
        rng = create_rng(42)
        inputs, targets = selective_copying(rng, seq_len=16, n_copy=3, vocab_size=8, n_samples=5)
        assert len(inputs) == 5
        assert len(targets) == 5
        assert len(inputs[0]) == 16
        assert len(targets[0]) == 16
        marker_pos = 16 - 3 - 1  # 12
        for inp, tgt in zip(inputs, targets):
            # Marker at correct position
            assert inp[marker_pos] == 1
            # Target blanks before marker
            for i in range(marker_pos + 1):
                assert tgt[i] == 0
            # Data tokens in target match those in input
            data_in = [inp[i] for i in range(marker_pos) if inp[i] >= 2]
            data_out = [tgt[i] for i in range(marker_pos + 1, 16) if tgt[i] >= 2]
            assert data_in == data_out
            assert len(data_in) == 3

    def test_ssm_recurrence(self):
        """Manual 2-step check of the SSM recurrence."""
        import math
        model = self._small_model()
        probs, cache = mamba_forward(model, [2, 3])

        d_inner = model.d_inner
        d_state = model.d_state

        # Step 0: h[1] = A_bar[0] * h[0] + B_bar[0] * x[0]
        # h[0] is all zeros, so h[1] = B_bar[0] * x[0]
        for d in range(d_inner):
            for n in range(d_state):
                expected = cache.B_bar[0][d][n] * cache.conv_silu_out[0][d]
                assert abs(cache.h_states[1][d][n] - expected) < 1e-10

        # Step 1: h[2] = A_bar[1] * h[1] + B_bar[1] * x[1]
        for d in range(d_inner):
            for n in range(d_state):
                expected = (cache.A_bar[1][d][n] * cache.h_states[1][d][n]
                            + cache.B_bar[1][d][n] * cache.conv_silu_out[1][d])
                assert abs(cache.h_states[2][d][n] - expected) < 1e-10

    def test_softplus(self):
        import math
        # softplus(0) = log(2)
        assert abs(softplus(0.0) - math.log(2.0)) < 1e-10
        # For large x, softplus(x) ≈ x
        assert abs(softplus(25.0) - 25.0) < 1e-10
        # softplus is always positive
        assert softplus(-5.0) > 0.0

    def test_softplus_derivative(self):
        # softplus'(0) = sigmoid(0) = 0.5
        assert abs(softplus_derivative(0.0) - 0.5) < 1e-10
        # softplus'(large) → 1
        assert abs(softplus_derivative(20.0) - 1.0) < 1e-6
