import unittest

import numpy as np
import pytest

from leakproofcmap.metrics import (
    PhenotypicMetric,
    calc_spearmansrank_scores,
    calc_zhang_scores,
    get_leak_proof_cmap_standard_L1000_metrics,
)


@pytest.mark.parametrize(
    "anchor, queries, expected_result",
    [
        (np.array([1, 2, 3]), np.array([5, 4, 6]), 0.9285714),
        (np.array([1, 2, 3]), np.array([6, 5, 4]), 0.714285),
    ],
)
def test_calc_zhang_scores(anchor, queries, expected_result):
    result = calc_zhang_scores(anchor, queries)
    assert np.allclose(result, expected_result)


@pytest.mark.parametrize(
    "anchor, queries, expected_result",
    [
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 1),
        (np.array([1, 2, 3]), np.array([6, 5, 4]), -1),
    ],
)
def test_calc_spearmansrank_scores(anchor, queries, expected_result):
    result = calc_spearmansrank_scores(anchor, queries)
    assert np.allclose(result, expected_result)


def test_get_leak_proof_cmap_standard_L1000_metrics():
    metrics = get_leak_proof_cmap_standard_L1000_metrics()
    assert isinstance(metrics, dict)
    assert "Rank" in metrics
    assert "EuclideanPCA" in metrics
    assert "Cosine" in metrics
    assert "Euclidean" in metrics


class TestPhenotypicMetric(unittest.TestCase):
    def setUp(self):
        self.metric = PhenotypicMetric(
            name="test", method=np.dot, range=(0, 1), higher_is_better=True
        )
        self.anchor = np.array([1, 2, 3])
        self.query = np.array([4, 5, 6])

    def test_init(self):
        self.assertEqual(self.metric.name, "test")
        self.assertEqual(self.metric.func, np.dot)
        self.assertEqual(self.metric.range, (0, 1))
        self.assertEqual(self.metric.higher_is_better, True)

    def test_call(self):
        result = self.metric(self.anchor, self.query)
        self.assertEqual(result, np.dot(self.anchor, self.query))

    def test_distance(self):
        result = self.metric.distance(self.anchor, self.query)
        expected = 1 - ((np.dot(self.anchor, self.query) - 0) / (1 - 0))
        self.assertEqual(result, expected)

    def test_similarity(self):
        result = self.metric.similarity(self.anchor, self.query)
        expected = (np.dot(self.anchor, self.query) - 0) / (1 - 0)
        self.assertEqual(result, expected)
