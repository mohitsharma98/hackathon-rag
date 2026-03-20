"""
Cloud vector store using Pinecone.
"""

import os

from rag_framework.config.models import VectorStoreConfig
from rag_framework.core.exceptions import BackendConnectionError, MissingCredentialError, VectorStoreError
from rag_framework.core.interfaces import BaseVectorStore, Chunk, EmbeddedChunk, RetrievalResult


class PineconeStore(BaseVectorStore):
    """
    Stores and queries vectors using Pinecone.

    Requires: PINECONE_API_KEY env var (or config.pinecone_api_key).
    The index must already exist in Pinecone.

    TODO: Add auto-create index logic with correct dimension.
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.api_key = config.pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self._index = None

    def health_check(self) -> None:
        if not self.api_key:
            raise MissingCredentialError("PINECONE_API_KEY")
        try:
            from pinecone import Pinecone  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pinecone not installed. Run: pip install pinecone-client"
            ) from e
        try:
            self._get_index()
        except Exception as e:
            raise BackendConnectionError("Pinecone", str(e)) from e

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        index = self._get_index()
        vectors = [
            {
                "id": str(ec.chunk.index),
                "values": ec.embedding,
                "metadata": {"text": ec.chunk.text, **ec.chunk.metadata},
            }
            for ec in embedded_chunks
        ]
        try:
            # Pinecone recommends batches of ≤100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                index.upsert(vectors=vectors[i : i + batch_size])
        except Exception as e:
            raise VectorStoreError(f"Pinecone upsert failed: {e}") from e

    def query(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        index = self._get_index()
        try:
            response = index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
            )
        except Exception as e:
            raise VectorStoreError(f"Pinecone query failed: {e}") from e

        results = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append(
                RetrievalResult(
                    chunk=Chunk(
                        text=meta.pop("text", ""),
                        index=-1,
                        metadata=meta,
                    ),
                    score=match.score,
                )
            )
        return results

    def delete_collection(self) -> None:
        """Delete all vectors in the index namespace (Pinecone doesn't have collections)."""
        index = self._get_index()
        try:
            index.delete(delete_all=True)
        except Exception as e:
            raise VectorStoreError(f"Pinecone delete_all failed: {e}") from e

    def _get_index(self):
        if self._index is None:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.api_key)
            self._index = pc.Index(self.config.pinecone_index_name)
        return self._index
