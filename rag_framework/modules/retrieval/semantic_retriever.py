"""
Semantic retriever: pure vector similarity search.
"""

import time

from rag_framework.config.models import RetrieverConfig
from rag_framework.core.exceptions import RetrievalError
from rag_framework.core.interfaces import BaseEmbedder, BaseRetriever, BaseVectorStore, RetrievalResult


class SemanticRetriever(BaseRetriever):
    """
    Retrieves documents using dense vector similarity.

    Pipeline: query → embed → vector store query → ranked results.
    """

    def __init__(
        self,
        config: RetrieverConfig,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
    ):
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store

    def health_check(self) -> None:
        self.embedder.health_check()
        self.vector_store.health_check()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        k = top_k or self.config.top_k
        t0 = time.perf_counter()

        try:
            [query_embedding] = self.embedder.embed([query])
        except Exception as e:
            raise RetrievalError(f"Failed to embed query: {e}") from e

        try:
            results = self.vector_store.query(query_embedding, top_k=k)
        except Exception as e:
            raise RetrievalError(f"Vector store query failed: {e}") from e

        elapsed_ms = (time.perf_counter() - t0) * 1000
        for r in results:
            r.metadata["retrieval_latency_ms"] = round(elapsed_ms, 2)
            r.metadata["retriever"] = "semantic"

        return results
