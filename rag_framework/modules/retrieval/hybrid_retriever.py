"""
Hybrid retriever: BM25 sparse + dense vector, score fusion.

Status: OPTIONAL — functional skeleton. Requires rank_bm25.
"""

import time

from rag_framework.config.models import RetrieverConfig
from rag_framework.core.exceptions import RetrievalError
from rag_framework.core.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore, Chunk, RetrievalResult


class HybridRetriever(BaseRetriever):
    """
    Combines BM25 keyword retrieval with dense vector retrieval via score fusion.

    Fusion formula:
        final_score = bm25_weight * bm25_score + vector_weight * vector_score

    The corpus (all indexed chunk texts) must be provided at construction time
    so BM25 can be initialised.

    TODO: Persist BM25 index alongside the vector store so corpus is not
          needed at query time. Or rebuild from vector store metadata.
    TODO: Consider Reciprocal Rank Fusion (RRF) as an alternative to
          weighted score fusion.
    """

    def __init__(
        self,
        config: RetrieverConfig,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        corpus: list[Chunk] | None = None,
    ):
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.corpus = corpus or []
        self._bm25 = None

    def health_check(self) -> None:
        self.embedder.health_check()
        self.vector_store.health_check()
        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "rank_bm25 not installed. Run: pip install rank-bm25"
            ) from e

    def set_corpus(self, chunks: list[Chunk]) -> None:
        """
        Provide the full set of indexed chunks so BM25 can be built.
        Call this after ingestion.
        """
        self.corpus = chunks
        self._bm25 = None  # Reset so it's rebuilt on next query

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k or self.config.top_k
        fetch_k = k * 3  # Over-fetch to allow re-ranking

        t0 = time.perf_counter()

        # --- Dense retrieval ---
        try:
            [query_embedding] = self.embedder.embed([query])
            vector_results = self.vector_store.query(query_embedding, top_k=fetch_k)
        except Exception as e:
            raise RetrievalError(f"Vector retrieval failed: {e}") from e

        # --- Sparse BM25 retrieval ---
        bm25_scores = self._bm25_scores(query, fetch_k)

        # --- Fuse scores ---
        fused = self._fuse(vector_results, bm25_scores)
        fused.sort(key=lambda r: r.score, reverse=True)
        results = fused[:k]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        for r in results:
            r.metadata["retrieval_latency_ms"] = round(elapsed_ms, 2)
            r.metadata["retriever"] = "hybrid"

        return results

    # ------------------------------------------------------------------

    def _bm25_scores(self, query: str, top_k: int) -> dict[str, float]:
        """Return {chunk_text: normalised_bm25_score} for top_k results."""
        if not self.corpus:
            return {}

        bm25 = self._get_bm25()
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Normalise to [0, 1]
        max_score = max(scores) if max(scores) > 0 else 1.0
        text_to_score: dict[str, float] = {}
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        for idx, score in indexed:
            text = self.corpus[idx].text
            text_to_score[text] = score / max_score

        return text_to_score

    def _get_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [chunk.text.lower().split() for chunk in self.corpus]
            self._bm25 = BM25Okapi(tokenized_corpus)
        return self._bm25

    def _fuse(
        self,
        vector_results: list[RetrievalResult],
        bm25_scores: dict[str, float],
    ) -> list[RetrievalResult]:
        vw = self.config.vector_weight
        bw = self.config.bm25_weight

        # Normalise vector scores to [0, 1]
        max_vs = max((r.score for r in vector_results), default=1.0) or 1.0
        fused: list[RetrievalResult] = []
        for r in vector_results:
            normalised_vs = r.score / max_vs
            bm25_s = bm25_scores.get(r.chunk.text, 0.0)
            fused_score = vw * normalised_vs + bw * bm25_s
            fused.append(RetrievalResult(chunk=r.chunk, score=fused_score, metadata=r.metadata))
        return fused
