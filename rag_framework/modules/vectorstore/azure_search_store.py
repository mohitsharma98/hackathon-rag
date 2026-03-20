"""
Cloud vector store using Azure AI Search.
"""

import os
import uuid

from rag_framework.config.models import VectorStoreConfig
from rag_framework.core.exceptions import BackendConnectionError, MissingCredentialError, VectorStoreError
from rag_framework.core.interfaces import BaseVectorStore, Chunk, EmbeddedChunk, RetrievalResult


class AzureSearchStore(BaseVectorStore):
    """
    Stores and queries vectors using Azure AI Search (vector search).

    Requires:
        AZURE_SEARCH_ENDPOINT — or config.azure_search_endpoint
        AZURE_SEARCH_API_KEY  — or config.azure_search_api_key

    The index is created automatically if it does not exist.

    TODO: Add hybrid search support (keyword + vector fields).
    TODO: Add semantic ranker configuration.
    """

    VECTOR_FIELD = "embedding"
    TEXT_FIELD = "content"
    ID_FIELD = "id"

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.endpoint = config.azure_search_endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = config.azure_search_api_key or os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = config.azure_search_index_name
        self._search_client = None

    def health_check(self) -> None:
        if not self.endpoint:
            raise MissingCredentialError("AZURE_SEARCH_ENDPOINT")
        if not self.api_key:
            raise MissingCredentialError("AZURE_SEARCH_API_KEY")
        try:
            from azure.search.documents import SearchClient  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "azure-search-documents not installed. "
                "Run: pip install azure-search-documents"
            ) from e
        try:
            self._ensure_index()
        except Exception as e:
            raise BackendConnectionError("Azure AI Search", str(e)) from e

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return
        self._ensure_index()
        client = self._get_search_client()
        docs = [
            {
                self.ID_FIELD: str(uuid.uuid4()),
                self.TEXT_FIELD: ec.chunk.text,
                self.VECTOR_FIELD: ec.embedding,
                **{f"meta_{k}": str(v) for k, v in ec.chunk.metadata.items()},
            }
            for ec in embedded_chunks
        ]
        try:
            client.upload_documents(documents=docs)
        except Exception as e:
            raise VectorStoreError(f"Azure Search upsert failed: {e}") from e

    def query(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        from azure.search.documents.models import VectorizedQuery

        client = self._get_search_client()
        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=top_k,
            fields=self.VECTOR_FIELD,
        )
        try:
            results = client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=[self.ID_FIELD, self.TEXT_FIELD],
                top=top_k,
            )
            return [
                RetrievalResult(
                    chunk=Chunk(
                        text=r[self.TEXT_FIELD],
                        index=-1,
                        metadata={"id": r[self.ID_FIELD]},
                    ),
                    score=r.get("@search.score", 0.0),
                )
                for r in results
            ]
        except Exception as e:
            raise VectorStoreError(f"Azure Search query failed: {e}") from e

    def delete_collection(self) -> None:
        """Delete and recreate the Azure Search index."""
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import SearchIndexClient

        client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
        )
        try:
            client.delete_index(self.index_name)
            self._search_client = None
        except Exception as e:
            raise VectorStoreError(f"Azure Search delete index failed: {e}") from e

    def _get_search_client(self):
        if self._search_client is None:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
            self._search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key),
            )
        return self._search_client

    def _ensure_index(self) -> None:
        """Create the index if it doesn't exist yet."""
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes import SearchIndexClient
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            VectorSearch,
            VectorSearchProfile,
        )

        idx_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
        )
        existing = [idx.name for idx in idx_client.list_indexes()]
        if self.index_name in existing:
            return

        fields = [
            SearchField(name=self.ID_FIELD, type=SearchFieldDataType.String, key=True),
            SearchField(name=self.TEXT_FIELD, type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name=self.VECTOR_FIELD,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.config.embedding_dim,
                vector_search_profile_name="default-profile",
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="default-profile", algorithm_configuration_name="hnsw")],
        )
        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        idx_client.create_index(index)
