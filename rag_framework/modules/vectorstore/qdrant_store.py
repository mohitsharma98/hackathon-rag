"""
Local vector store using Qdrant (local file mode).
"""

from rag_framework.config.models import VectorStoreConfig
from rag_framework.core.exceptions import BackendConnectionError, VectorStoreError
from rag_framework.core.interfaces import BaseVectorStore, Chunk, EmbeddedChunk, RetrievalResult


class QdrantLocalStore(BaseVectorStore):
    """
    Stores and queries vectors using Qdrant in local (on-disk) mode.

    Data is persisted to *config.qdrant_path*.
    No server or credentials required.
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._client = None

    def health_check(self) -> None:
        try:
            from qdrant_client import QdrantClient  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            ) from e
        try:
            self._get_client()
        except Exception as e:
            raise BackendConnectionError("Qdrant local", str(e)) from e

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        from qdrant_client.models import PointStruct

        client = self._get_client()
        self._ensure_collection(client)

        points = [
            PointStruct(
                id=ec.chunk.index,
                vector=ec.embedding,
                payload={"text": ec.chunk.text, **ec.chunk.metadata},
            )
            for ec in embedded_chunks
        ]
        try:
            client.upsert(collection_name=self.config.collection_name, points=points)
        except Exception as e:
            raise VectorStoreError(f"Qdrant upsert failed: {e}") from e

    def query(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        client = self._get_client()
        try:
            hits = client.search(
                collection_name=self.config.collection_name,
                query_vector=embedding,
                limit=top_k,
            )
        except Exception as e:
            raise VectorStoreError(f"Qdrant query failed: {e}") from e

        return [
            RetrievalResult(
                chunk=Chunk(
                    text=hit.payload.get("text", ""),
                    index=hit.id if isinstance(hit.id, int) else -1,
                    metadata={k: v for k, v in hit.payload.items() if k != "text"},
                ),
                score=hit.score,
            )
            for hit in hits
        ]

    def delete_collection(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(self.config.collection_name)
        except Exception as e:
            raise VectorStoreError(f"Qdrant delete_collection failed: {e}") from e

    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(path=self.config.qdrant_path)
        return self._client

    def _ensure_collection(self, client) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in client.get_collections().collections]
        if self.config.collection_name not in existing:
            client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
