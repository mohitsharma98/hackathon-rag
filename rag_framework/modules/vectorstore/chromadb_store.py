"""
Local vector store using ChromaDB.
"""

from rag_framework.config.models import VectorStoreConfig
from rag_framework.core.exceptions import BackendConnectionError, VectorStoreError
from rag_framework.core.interfaces import BaseVectorStore, Chunk, EmbeddedChunk, RetrievalResult


class ChromaDBStore(BaseVectorStore):
    """
    Stores and queries vectors using ChromaDB (persistent local mode).

    Data is persisted to *config.chromadb_path*.
    No credentials required.
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._client = None
        self._collection = None

    def health_check(self) -> None:
        try:
            import chromadb  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "chromadb not installed. Run: pip install chromadb"
            ) from e
        try:
            self._get_collection()
        except Exception as e:
            raise BackendConnectionError("ChromaDB", str(e)) from e

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        collection = self._get_collection()
        try:
            collection.upsert(
                ids=[str(ec.chunk.index) for ec in embedded_chunks],
                embeddings=[ec.embedding for ec in embedded_chunks],
                documents=[ec.chunk.text for ec in embedded_chunks],
                metadatas=[ec.chunk.metadata for ec in embedded_chunks],
            )
        except Exception as e:
            raise VectorStoreError(f"ChromaDB upsert failed: {e}") from e

    def query(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        collection = self._get_collection()
        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"ChromaDB query failed: {e}") from e

        retrieval_results: list[RetrievalResult] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB returns L2 distance; convert to a 0-1 similarity score
            score = 1.0 / (1.0 + dist)
            retrieval_results.append(
                RetrievalResult(
                    chunk=Chunk(text=doc, index=-1, metadata=meta or {}),
                    score=score,
                )
            )
        return retrieval_results

    def delete_collection(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(self.config.collection_name)
            self._collection = None
        except Exception as e:
            raise VectorStoreError(f"ChromaDB delete_collection failed: {e}") from e

    def _get_client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.config.chromadb_path)
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection
