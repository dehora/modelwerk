"""Tests for data generators and utilities."""

from modelwerk.primitives.random import create_rng
from modelwerk.data.generators import circles
from modelwerk.data.utils import one_hot, subsample, shuffle_together


class TestGenerators:
    def test_circles_shape(self):
        rng = create_rng(42)
        data, labels = circles(rng, n_samples=50)
        assert len(data) == 50
        assert len(labels) == 50
        assert all(len(p) == 2 for p in data)

    def test_circles_labels(self):
        rng = create_rng(42)
        data, labels = circles(rng, n_samples=20)
        # Labels should be 0.0 or 1.0
        assert all(l in (0.0, 1.0) for l in labels)
        # Should have both classes
        assert 0.0 in labels
        assert 1.0 in labels

    def test_circles_reproducibility(self):
        d1, l1 = circles(create_rng(42), n_samples=10)
        d2, l2 = circles(create_rng(42), n_samples=10)
        assert d1 == d2
        assert l1 == l2


class TestUtils:
    def test_one_hot(self):
        v = one_hot(3, 5)
        assert v == [0.0, 0.0, 0.0, 1.0, 0.0]

    def test_one_hot_default(self):
        v = one_hot(0)
        assert len(v) == 10
        assert v[0] == 1.0
        assert sum(v) == 1.0

    def test_subsample_size(self):
        rng = create_rng(42)
        data = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        labels = [0.0, 1.0, 0.0, 1.0, 0.0]
        sub_data, sub_labels = subsample(rng, data, labels, 3)
        assert len(sub_data) == 3
        assert len(sub_labels) == 3

    def test_subsample_preserves_pairs(self):
        """Each subsampled data point should have its original label."""
        rng = create_rng(42)
        data = [[float(i)] for i in range(10)]
        labels = [float(i * 10) for i in range(10)]
        sub_data, sub_labels = subsample(rng, data, labels, 5)
        for d, l in zip(sub_data, sub_labels):
            # label should be data[0] * 10
            assert l == d[0] * 10

    def test_subsample_reproducibility(self):
        data = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        labels = [0.0, 1.0, 0.0, 1.0, 0.0]
        d1, l1 = subsample(create_rng(42), data, labels, 3)
        d2, l2 = subsample(create_rng(42), data, labels, 3)
        assert d1 == d2
        assert l1 == l2

    def test_shuffle_together_preserves_all(self):
        rng = create_rng(42)
        data = [[1.0], [2.0], [3.0]]
        labels = [10.0, 20.0, 30.0]
        s_data, s_labels = shuffle_together(rng, data, labels)
        assert len(s_data) == 3
        assert len(s_labels) == 3
        # All original items should be present
        assert sorted([d[0] for d in s_data]) == [1.0, 2.0, 3.0]
        assert sorted(s_labels) == [10.0, 20.0, 30.0]

    def test_shuffle_together_preserves_pairs(self):
        rng = create_rng(42)
        data = [[float(i)] for i in range(5)]
        labels = [float(i * 10) for i in range(5)]
        s_data, s_labels = shuffle_together(rng, data, labels)
        for d, l in zip(s_data, s_labels):
            assert l == d[0] * 10
