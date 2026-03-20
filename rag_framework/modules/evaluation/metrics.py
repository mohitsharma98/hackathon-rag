"""
Pure-Python metric computation functions.

Each function is stateless and takes plain Python types so they can be
used independently of the rest of the framework.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    """
    Standard IR metrics for retrieval evaluation.

    All methods assume:
        retrieved: ordered list of retrieved texts (best first).
        relevant:  set of ground-truth relevant texts.
    """

    @staticmethod
    def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """Fraction of relevant docs found in the top-k results."""
        if not relevant:
            return 0.0
        top_k = set(retrieved[:k])
        return len(top_k & relevant) / len(relevant)

    @staticmethod
    def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """Fraction of top-k results that are relevant."""
        if k == 0:
            return 0.0
        top_k = retrieved[:k]
        return sum(1 for t in top_k if t in relevant) / k

    @staticmethod
    def average_precision(retrieved: list[str], relevant: set[str]) -> float:
        """Area under the Precision-Recall curve (AP)."""
        if not relevant:
            return 0.0
        hits = 0
        sum_precision = 0.0
        for i, text in enumerate(retrieved, start=1):
            if text in relevant:
                hits += 1
                sum_precision += hits / i
        return sum_precision / len(relevant)

    @staticmethod
    def mrr(retrieved_lists: list[list[str]], relevant_sets: list[set[str]]) -> float:
        """
        Mean Reciprocal Rank across multiple queries.
        MRR = mean of 1/rank_of_first_relevant_result per query.
        """
        reciprocal_ranks: list[float] = []
        for retrieved, relevant in zip(retrieved_lists, relevant_sets):
            rr = 0.0
            for rank, text in enumerate(retrieved, start=1):
                if text in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    @staticmethod
    def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
        """
        Normalised Discounted Cumulative Gain at k.
        Assumes binary relevance (relevant or not).
        """
        def dcg(items: list[str]) -> float:
            return sum(
                (1 / math.log2(i + 2)) for i, t in enumerate(items) if t in relevant
            )

        ideal_hits = min(len(relevant), k)
        ideal = [1.0] * ideal_hits + [0.0] * (k - ideal_hits)
        idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
        return dcg(retrieved[:k]) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Parsing metrics
# ---------------------------------------------------------------------------

@dataclass
class ParsingMetrics:
    """
    Metrics for evaluating parser quality.

    TODO: Implement OCR accuracy, layout preservation scoring.
    """

    @staticmethod
    def char_count(text: str) -> int:
        """Raw character count — proxy for text completeness."""
        return len(text)

    @staticmethod
    def word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def completeness_ratio(parsed_text: str, reference_text: str) -> float:
        """
        Fraction of reference words present in parsed text.
        Simple word-overlap proxy for parsing completeness.

        TODO: Replace with a proper text similarity metric (ROUGE, BERTScore).
        """
        if not reference_text:
            return 1.0
        ref_words = set(reference_text.lower().split())
        parsed_words = set(parsed_text.lower().split())
        return len(parsed_words & ref_words) / len(ref_words)

    @staticmethod
    def latency_score(parse_time_s: float, budget_s: float = 5.0) -> float:
        """
        Normalised latency score in [0, 1].
        1.0 = instantaneous, 0.0 = at or beyond budget.
        """
        return max(0.0, 1.0 - parse_time_s / budget_s)


# ---------------------------------------------------------------------------
# Chunking metrics
# ---------------------------------------------------------------------------

@dataclass
class ChunkingMetrics:
    """
    Metrics for evaluating chunking quality.

    TODO: Add embedding-based coherence score using intra-chunk vs inter-chunk
          cosine similarity.
    """

    @staticmethod
    def chunk_count(chunks: list) -> int:
        return len(chunks)

    @staticmethod
    def avg_chunk_length(chunks: list) -> float:
        if not chunks:
            return 0.0
        return sum(len(c.text) for c in chunks) / len(chunks)

    @staticmethod
    def size_variance(chunks: list) -> float:
        """Variance in chunk character length — lower is more uniform."""
        if len(chunks) < 2:
            return 0.0
        lengths = [len(c.text) for c in chunks]
        mean = sum(lengths) / len(lengths)
        return sum((l - mean) ** 2 for l in lengths) / len(lengths)

    @staticmethod
    def boundary_precision(
        chunks: list,
        reference_boundaries: list[int],
        tolerance: int = 20,
    ) -> float:
        """
        Fraction of reference boundaries (character offsets) that have a
        corresponding chunk boundary within *tolerance* characters.

        TODO: Implement full boundary offset tracking in Chunk metadata.
        """
        # TODO: requires chunk.metadata["start_char"] to be populated
        return 0.0  # placeholder

    @staticmethod
    def coherence_score(chunks: list, embedder=None) -> float:
        """
        Mean intra-chunk cosine similarity (using sentence embeddings).
        Higher = more semantically coherent chunks.

        TODO: Implement. Requires splitting each chunk into sentences and
              computing pairwise similarity.
        """
        return 0.0  # placeholder
