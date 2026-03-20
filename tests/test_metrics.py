"""
Tests for evaluation metrics.
"""

import pytest

from rag_framework.modules.evaluation.metrics import RetrievalMetrics


def test_recall_at_k_perfect():
    rm = RetrievalMetrics()
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b", "c"}
    assert rm.recall_at_k(retrieved, relevant, k=3) == 1.0


def test_recall_at_k_partial():
    rm = RetrievalMetrics()
    assert rm.recall_at_k(["a", "x", "b"], {"a", "b", "c"}, k=3) == pytest.approx(2 / 3)


def test_recall_at_k_empty_relevant():
    rm = RetrievalMetrics()
    assert rm.recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_precision_at_k():
    rm = RetrievalMetrics()
    assert rm.precision_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(2 / 3)


def test_mrr_first_rank():
    rm = RetrievalMetrics()
    mrr = rm.mrr([["a", "b", "c"]], [{"a"}])
    assert mrr == 1.0


def test_mrr_second_rank():
    rm = RetrievalMetrics()
    mrr = rm.mrr([["x", "a", "c"]], [{"a"}])
    assert mrr == pytest.approx(0.5)


def test_mrr_no_hit():
    rm = RetrievalMetrics()
    mrr = rm.mrr([["x", "y"]], [{"a"}])
    assert mrr == 0.0


def test_ndcg_at_k_perfect():
    rm = RetrievalMetrics()
    assert rm.ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_average_precision():
    rm = RetrievalMetrics()
    # a at 1, b at 3 → AP = (1/1 + 2/3) / 2 = 0.833...
    ap = rm.average_precision(["a", "x", "b"], {"a", "b"})
    assert ap == pytest.approx((1.0 + 2 / 3) / 2)
