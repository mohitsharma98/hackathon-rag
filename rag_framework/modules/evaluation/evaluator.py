"""
Main evaluator: aggregates all metric groups into a single EvaluationReport.
"""

from __future__ import annotations

from rag_framework.config.models import EvaluatorConfig
from rag_framework.core.interfaces import BaseEvaluator, EvaluationReport, RetrievalResult
from rag_framework.modules.evaluation.metrics import RetrievalMetrics


class RAGEvaluator(BaseEvaluator):
    """
    Aggregates retrieval, parsing, chunking, and answer metrics.

    Usage:
        evaluator = RAGEvaluator(config)
        report = evaluator.evaluate(
            queries=["what is X?"],
            expected=[["relevant text 1", "relevant text 2"]],
            retrieved=[[RetrievalResult(...), ...]],
        )
        print(report.metrics)
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config

    def health_check(self) -> None:
        """No required external deps for basic metrics."""
        if self.config.enable_answer_metrics:
            # TODO: validate LLM judge API key
            pass

    def evaluate(
        self,
        queries: list[str],
        expected: list[list[str]],
        retrieved: list[list[RetrievalResult]],
        answers: list[str] | None = None,
    ) -> EvaluationReport:
        metrics: dict[str, float] = {}

        if self.config.enable_retrieval_metrics:
            metrics.update(
                self._retrieval_metrics(queries, expected, retrieved)
            )

        if self.config.enable_answer_metrics and answers:
            metrics.update(
                self._answer_metrics(queries, expected, answers)
            )

        return EvaluationReport(
            metrics=metrics,
            metadata={
                "num_queries": len(queries),
                "k_values": self.config.k_values,
            },
        )

    def _retrieval_metrics(
        self,
        queries: list[str],
        expected: list[list[str]],
        retrieved: list[list[RetrievalResult]],
    ) -> dict[str, float]:
        """Compute Recall@K, Precision@K, MRR, NDCG@K."""
        rm = RetrievalMetrics()
        metrics: dict[str, float] = {}

        retrieved_texts = [[r.chunk.text for r in results] for results in retrieved]
        relevant_sets = [set(exp) for exp in expected]

        for k in self.config.k_values:
            recalls = [
                rm.recall_at_k(ret, rel, k)
                for ret, rel in zip(retrieved_texts, relevant_sets)
            ]
            precisions = [
                rm.precision_at_k(ret, rel, k)
                for ret, rel in zip(retrieved_texts, relevant_sets)
            ]
            ndcgs = [
                rm.ndcg_at_k(ret, rel, k)
                for ret, rel in zip(retrieved_texts, relevant_sets)
            ]
            metrics[f"recall_at_{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
            metrics[f"precision_at_{k}"] = sum(precisions) / len(precisions) if precisions else 0.0
            metrics[f"ndcg_at_{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

        metrics["mrr"] = rm.mrr(retrieved_texts, relevant_sets)

        # Average retrieval latency
        latencies = [
            r.metadata.get("retrieval_latency_ms", 0.0)
            for results in retrieved
            for r in results[:1]  # take latency from first result per query
        ]
        if latencies:
            metrics["avg_retrieval_latency_ms"] = sum(latencies) / len(latencies)

        return metrics

    def _answer_metrics(
        self,
        queries: list[str],
        expected: list[list[str]],
        answers: list[str],
    ) -> dict[str, float]:
        """
        Compute answer correctness and faithfulness using an LLM judge.

        TODO: Implement LLM-as-judge scoring.
              Recommended: use GPT-4o-mini with a structured prompt that
              returns a JSON score for correctness (0-1) and faithfulness (0-1).
        """
        # Placeholder — always returns 0.0 until implemented
        return {
            "answer_correctness": 0.0,   # TODO
            "answer_faithfulness": 0.0,  # TODO
        }

    # ------------------------------------------------------------------
    # Comparison helper
    # ------------------------------------------------------------------

    @staticmethod
    def compare(report_a: EvaluationReport, report_b: EvaluationReport) -> dict[str, dict]:
        """
        Return a side-by-side diff of two evaluation reports.

        Example output:
            {
                "recall_at_5": {"a": 0.72, "b": 0.81, "delta": +0.09},
                ...
            }
        """
        all_keys = set(report_a.metrics) | set(report_b.metrics)
        comparison: dict[str, dict] = {}
        for key in sorted(all_keys):
            a_val = report_a.metrics.get(key, float("nan"))
            b_val = report_b.metrics.get(key, float("nan"))
            comparison[key] = {
                "a": round(a_val, 4),
                "b": round(b_val, 4),
                "delta": round(b_val - a_val, 4),
            }
        return comparison
