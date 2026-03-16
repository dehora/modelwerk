"""Tests for primitive operations (scalar, vector, matrix, activations, losses)."""

import math

from modelwerk.primitives import scalar, vector, matrix, activations, losses
from modelwerk.primitives.random import create_rng, random_vector, random_matrix, xavier_init


# --- Scalar ---

class TestScalar:
    def test_multiply(self):
        assert scalar.multiply(3.0, 4.0) == 12.0
        assert scalar.multiply(0.0, 5.0) == 0.0
        assert scalar.multiply(-2.0, 3.0) == -6.0

    def test_add(self):
        assert scalar.add(1.0, 2.0) == 3.0

    def test_subtract(self):
        assert scalar.subtract(5.0, 3.0) == 2.0

    def test_negate(self):
        assert scalar.negate(3.0) == -3.0
        assert scalar.negate(-1.0) == 1.0

    def test_inverse(self):
        assert scalar.inverse(2.0) == 0.5
        assert scalar.inverse(4.0) == 0.25

    def test_exp(self):
        assert abs(scalar.exp(0.0) - 1.0) < 1e-10
        assert abs(scalar.exp(1.0) - math.e) < 1e-10
        # Should not overflow on large values
        result = scalar.exp(1000.0)
        assert math.isfinite(result)

    def test_log(self):
        assert abs(scalar.log(1.0) - 0.0) < 1e-10
        assert abs(scalar.log(math.e) - 1.0) < 1e-10
        # Should not error on zero or negative
        result = scalar.log(0.0)
        assert math.isfinite(result)

    def test_power(self):
        assert scalar.power(2.0, 3.0) == 8.0
        assert scalar.power(4.0, 0.5) == 2.0

    def test_clamp(self):
        assert scalar.clamp(5.0, 0.0, 10.0) == 5.0
        assert scalar.clamp(-1.0, 0.0, 10.0) == 0.0
        assert scalar.clamp(15.0, 0.0, 10.0) == 10.0


# --- Vector ---

class TestVector:
    def test_dot(self):
        assert vector.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0

    def test_add(self):
        assert vector.add([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]

    def test_subtract(self):
        assert vector.subtract([5.0, 3.0], [1.0, 2.0]) == [4.0, 1.0]

    def test_scale(self):
        assert vector.scale(2.0, [1.0, 3.0]) == [2.0, 6.0]

    def test_elementwise(self):
        result = vector.elementwise(scalar.multiply, [2.0, 3.0], [4.0, 5.0])
        assert result == [8.0, 15.0]

    def test_apply(self):
        result = vector.apply(scalar.negate, [1.0, -2.0])
        assert result == [-1.0, 2.0]

    def test_magnitude(self):
        assert abs(vector.magnitude([3.0, 4.0]) - 5.0) < 1e-10

    def test_zeros_ones(self):
        assert vector.zeros(3) == [0.0, 0.0, 0.0]
        assert vector.ones(2) == [1.0, 1.0]

    def test_sum_all(self):
        assert vector.sum_all([1.0, 2.0, 3.0]) == 6.0

    def test_max_val(self):
        assert vector.max_val([1.0, 5.0, 3.0]) == 5.0
        assert vector.max_val([-1.0, -5.0, -3.0]) == -1.0


# --- Matrix ---

class TestMatrix:
    def test_mat_vec(self):
        M = [[1.0, 2.0], [3.0, 4.0]]
        v = [5.0, 6.0]
        result = matrix.mat_vec(M, v)
        assert result == [17.0, 39.0]

    def test_mat_mat(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        result = matrix.mat_mat(A, B)
        assert result == [[19.0, 22.0], [43.0, 50.0]]

    def test_transpose(self):
        M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        T = matrix.transpose(M)
        assert T == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def test_outer(self):
        result = matrix.outer([1.0, 2.0], [3.0, 4.0])
        assert result == [[3.0, 4.0], [6.0, 8.0]]

    def test_zeros(self):
        Z = matrix.zeros(2, 3)
        assert Z == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    def test_add(self):
        A = [[1.0, 2.0], [3.0, 4.0]]
        B = [[5.0, 6.0], [7.0, 8.0]]
        assert matrix.add(A, B) == [[6.0, 8.0], [10.0, 12.0]]

    def test_scale(self):
        M = [[1.0, 2.0], [3.0, 4.0]]
        assert matrix.scale(2.0, M) == [[2.0, 4.0], [6.0, 8.0]]

    def test_flatten(self):
        assert matrix.flatten([[1.0, 2.0], [3.0, 4.0]]) == [1.0, 2.0, 3.0, 4.0]

    def test_reshape(self):
        v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert matrix.reshape(v, 2, 3) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    def test_tensor3d_flatten(self):
        t = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        assert matrix.tensor3d_flatten(t) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    def test_tensor3d_flatten_roundtrip(self):
        t = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        flat = matrix.tensor3d_flatten(t)
        restored = matrix.tensor3d_reshape(flat, 2, 2, 3)
        assert restored == t


# --- Random ---

class TestRandom:
    def test_reproducibility(self):
        rng1 = create_rng(42)
        rng2 = create_rng(42)
        v1 = random_vector(rng1, 5)
        v2 = random_vector(rng2, 5)
        assert v1 == v2

    def test_random_matrix_shape(self):
        rng = create_rng(0)
        M = random_matrix(rng, 3, 4)
        assert len(M) == 3
        assert all(len(row) == 4 for row in M)

    def test_xavier_init_shape(self):
        rng = create_rng(0)
        M = xavier_init(rng, 10, 5)
        assert len(M) == 5
        assert all(len(row) == 10 for row in M)


# --- Activations ---

class TestActivations:
    def test_step(self):
        assert activations.step(0.0) == 1.0
        assert activations.step(1.0) == 1.0
        assert activations.step(-0.5) == 0.0

    def test_sigmoid(self):
        assert abs(activations.sigmoid(0.0) - 0.5) < 1e-10
        assert activations.sigmoid(100.0) > 0.99
        assert activations.sigmoid(-100.0) < 0.01

    def test_sigmoid_derivative(self):
        # At x=0, derivative is 0.25
        assert abs(activations.sigmoid_derivative(0.0) - 0.25) < 1e-10

    def test_tanh(self):
        assert abs(activations.tanh_(0.0)) < 1e-10
        assert activations.tanh_(10.0) > 0.99
        assert activations.tanh_(-10.0) < -0.99

    def test_tanh_derivative(self):
        # At x=0, derivative is 1.0
        assert abs(activations.tanh_derivative(0.0) - 1.0) < 1e-10

    def test_relu(self):
        assert activations.relu(5.0) == 5.0
        assert activations.relu(-3.0) == 0.0
        assert activations.relu(0.0) == 0.0

    def test_relu_derivative(self):
        assert activations.relu_derivative(5.0) == 1.0
        assert activations.relu_derivative(-3.0) == 0.0

    def test_identity(self):
        assert activations.identity(5.0) == 5.0
        assert activations.identity(-3.0) == -3.0
        assert activations.identity(0.0) == 0.0

    def test_identity_derivative(self):
        assert activations.identity_derivative(5.0) == 1.0
        assert activations.identity_derivative(-3.0) == 1.0
        assert activations.identity_derivative(0.0) == 1.0

    def test_silu(self):
        # silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert abs(activations.silu(0.0)) < 1e-10
        # silu(x) > 0 for x > 0
        assert activations.silu(2.0) > 0
        # silu(x) < 0 for small negative x (non-monotonic)
        assert activations.silu(-1.0) < 0

    def test_silu_derivative(self):
        # At x=0: sigmoid(0) + 0 * sigmoid(0) * (1 - sigmoid(0)) = 0.5
        assert abs(activations.silu_derivative(0.0) - 0.5) < 1e-10
        # Numerical check at x=1
        eps = 1e-5
        numerical = (activations.silu(1.0 + eps) - activations.silu(1.0 - eps)) / (2 * eps)
        analytical = activations.silu_derivative(1.0)
        assert abs(numerical - analytical) < 1e-4

    def test_softmax(self):
        result = activations.softmax([1.0, 2.0, 3.0])
        # Should sum to 1
        assert abs(sum(result) - 1.0) < 1e-10
        # Should be monotonically increasing
        assert result[0] < result[1] < result[2]

    def test_softmax_numerical_stability(self):
        # Large values should not overflow
        result = activations.softmax([1000.0, 1001.0, 1002.0])
        assert abs(sum(result) - 1.0) < 1e-10


# --- Losses ---

class TestLosses:
    def test_mse_zero(self):
        assert losses.mse([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_mse_known(self):
        # MSE of [1, 2] vs [3, 4] = ((2^2 + 2^2) / 2) = 4.0
        assert abs(losses.mse([1.0, 2.0], [3.0, 4.0]) - 4.0) < 1e-10

    def test_mse_derivative_numerical(self):
        """Analytical derivative should match numerical (finite difference)."""
        predicted = [0.5, 0.8, 0.2]
        actual = [1.0, 0.0, 0.5]
        analytical = losses.mse_derivative(predicted, actual)

        eps = 1e-5
        for i in range(len(predicted)):
            p_plus = predicted[:]
            p_minus = predicted[:]
            p_plus[i] += eps
            p_minus[i] -= eps
            numerical = (losses.mse(p_plus, actual) - losses.mse(p_minus, actual)) / (2 * eps)
            assert abs(analytical[i] - numerical) < 1e-5, (
                f"MSE derivative mismatch at index {i}: "
                f"analytical={analytical[i]}, numerical={numerical}"
            )

    def test_cross_entropy(self):
        # -log(0.7) * 1.0 + -log(0.3) * 0.0 = -log(0.7)
        result = losses.cross_entropy([0.7, 0.3], [1.0, 0.0])
        assert abs(result - (-math.log(0.7))) < 1e-10

    def test_cross_entropy_derivative_numerical(self):
        """Analytical derivative should match numerical (finite difference)."""
        predicted = [0.7, 0.2, 0.1]
        actual = [1.0, 0.0, 0.0]
        analytical = losses.cross_entropy_derivative(predicted, actual)

        eps = 1e-5
        for i in range(len(predicted)):
            p_plus = predicted[:]
            p_minus = predicted[:]
            p_plus[i] += eps
            p_minus[i] -= eps
            numerical = (
                losses.cross_entropy(p_plus, actual)
                - losses.cross_entropy(p_minus, actual)
            ) / (2 * eps)
            assert abs(analytical[i] - numerical) < 1e-4, (
                f"CE derivative mismatch at index {i}: "
                f"analytical={analytical[i]}, numerical={numerical}"
            )
