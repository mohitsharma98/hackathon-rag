"""
Cloud vector store using Databricks Vector Search.

Two modes controlled by VectorStoreConfig.databricks_index_type:

  "direct_access" (default)
      The framework manages embeddings. Call upsert() with pre-computed
      vectors, and query() with a float vector.
      Requires: DATABRICKS_HOST + DATABRICKS_TOKEN (or SP credentials).

  "delta_sync"
      Databricks manages embedding via a Foundation Model endpoint.
      upsert() writes raw text rows to the backing Delta table; the index
      syncs automatically (CONTINUOUS) or on demand (TRIGGERED).
      query() uses similarity_search() with query_text instead of a vector.
      Requires: databricks_source_table and databricks_embedding_model_endpoint
      to be set in config.
"""

from __future__ import annotations

import os
import uuid

from rag_framework.config.models import VectorStoreConfig
from rag_framework.core.exceptions import (
    BackendConnectionError,
    MissingCredentialError,
    VectorStoreError,
)
from rag_framework.core.interfaces import BaseVectorStore, Chunk, EmbeddedChunk, RetrievalResult


class DatabricksVectorSearchStore(BaseVectorStore):
    """
    Stores and queries vectors using Databricks Vector Search.

    Direct-access mode:
        Pre-computed embeddings are upserted via the SDK.
        query() accepts a float vector.

    Delta-sync mode:
        Text rows are written to a Delta table; Databricks auto-embeds them.
        query() accepts query_text via similarity_search().

    Required env vars (or set via config):
        DATABRICKS_HOST   — e.g. https://<workspace>.azuredatabricks.net
        DATABRICKS_TOKEN  — personal access token or service-principal secret
    """

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.host = config.databricks_host or os.getenv("DATABRICKS_HOST", "")
        self.token = config.databricks_token or os.getenv("DATABRICKS_TOKEN", "")
        self._client = None
        self._index = None

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def health_check(self) -> None:
        if not self.host:
            raise MissingCredentialError("DATABRICKS_HOST")
        if not self.token:
            raise MissingCredentialError("DATABRICKS_TOKEN")
        try:
            from databricks.vector_search.client import VectorSearchClient  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "databricks-vectorsearch not installed. "
                "Run: pip install databricks-vectorsearch"
            ) from e
        if not self.config.databricks_endpoint_name:
            raise VectorStoreError(
                "databricks_endpoint_name must be set in VectorStoreConfig."
            )
        if not self.config.databricks_index_name:
            raise VectorStoreError(
                "databricks_index_name must be set in VectorStoreConfig "
                "(format: catalog.schema.index_name)."
            )
        try:
            self._get_index()
        except Exception as e:
            raise BackendConnectionError("DatabricksVectorSearch", str(e)) from e

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        if self.config.databricks_index_type == "delta_sync":
            self._upsert_delta_sync(embedded_chunks)
        else:
            self._upsert_direct_access(embedded_chunks)

    def query(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        if self.config.databricks_index_type == "delta_sync":
            raise VectorStoreError(
                "delta_sync indexes do not support vector queries. "
                "Use DatabricksDeltaSyncRetriever or query with query_text instead."
            )
        return self._query_direct_access(embedding, top_k)

    def query_text(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        """
        Text-based similarity search — only valid for delta_sync indexes
        where Databricks manages embeddings via a Foundation Model endpoint.
        """
        if self.config.databricks_index_type != "delta_sync":
            raise VectorStoreError(
                "query_text() is only supported for delta_sync indexes. "
                "Use query() with a float vector for direct_access indexes."
            )
        return self._query_delta_sync(query_text, top_k)

    def delete_collection(self) -> None:
        """
        Direct-access: deletes all vectors by chunk_id.
        Delta-sync: not supported (table is managed externally).
        """
        if self.config.databricks_index_type == "delta_sync":
            raise VectorStoreError(
                "delete_collection() is not supported for delta_sync indexes. "
                "Truncate the backing Delta table directly."
            )
        index = self._get_index()
        try:
            # Databricks direct-access delete requires explicit IDs.
            # We delete by querying all stored IDs first.
            # For a full reset, the index itself should be deleted via the UI/API.
            # This is a best-effort clear for demo/test purposes.
            index.delete({"chunk_id": {"$exists": True}})
        except Exception as e:
            raise VectorStoreError(
                f"DatabricksVectorSearch delete failed: {e}. "
                "For a full reset, delete and recreate the index via the UI."
            ) from e

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from databricks.vector_search.client import VectorSearchClient
            self._client = VectorSearchClient(
                workspace_url=self.host,
                personal_access_token=self.token,
            )
        return self._client

    def _get_index(self):
        if self._index is None:
            client = self._get_client()
            self._index = client.get_index(
                endpoint_name=self.config.databricks_endpoint_name,
                index_name=self.config.databricks_index_name,
            )
        return self._index

    # ---- direct_access mode ------------------------------------------

    def _upsert_direct_access(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        index = self._get_index()
        rows = [
            {
                "chunk_id": str(ec.chunk.index) if ec.chunk.index >= 0 else str(uuid.uuid4()),
                "text": ec.chunk.text,
                "embedding": ec.embedding,
                **{k: str(v) for k, v in ec.chunk.metadata.items()},
            }
            for ec in embedded_chunks
        ]
        try:
            # SDK upsert accepts a list of dicts
            index.upsert(rows)
        except Exception as e:
            raise VectorStoreError(f"DatabricksVectorSearch upsert failed: {e}") from e

    def _query_direct_access(
        self, embedding: list[float], top_k: int
    ) -> list[RetrievalResult]:
        index = self._get_index()
        try:
            response = index.similarity_search(
                query_vector=embedding,
                columns=["chunk_id", "text"],
                num_results=top_k,
            )
        except Exception as e:
            raise VectorStoreError(f"DatabricksVectorSearch query failed: {e}") from e

        return self._parse_response(response)

    # ---- delta_sync mode ---------------------------------------------

    def _upsert_delta_sync(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Write rows to the backing Delta table.
        Databricks will auto-embed via the configured Foundation Model endpoint.
        Requires a SparkSession available in the runtime (Databricks notebook / job).
        """
        try:
            from pyspark.sql import SparkSession
        except ImportError as e:
            raise ImportError(
                "pyspark is required for delta_sync upsert. "
                "This mode is intended to run inside a Databricks cluster."
            ) from e

        if not self.config.databricks_source_table:
            raise VectorStoreError(
                "databricks_source_table must be set for delta_sync index upsert."
            )

        spark = SparkSession.getActiveSession()
        if spark is None:
            raise VectorStoreError(
                "No active SparkSession found. "
                "delta_sync upsert must run inside a Databricks cluster."
            )

        from pyspark.sql import Row

        rows = [
            Row(
                chunk_id=str(ec.chunk.index) if ec.chunk.index >= 0 else str(uuid.uuid4()),
                text=ec.chunk.text,
                **{k: str(v) for k, v in ec.chunk.metadata.items()},
            )
            for ec in embedded_chunks
        ]
        try:
            df = spark.createDataFrame(rows)
            df.write.mode("append").saveAsTable(self.config.databricks_source_table)
        except Exception as e:
            raise VectorStoreError(
                f"DatabricksVectorSearch delta_sync write failed: {e}"
            ) from e

        # Optionally trigger a sync for TRIGGERED pipelines
        if self.config.databricks_trigger_sync:
            try:
                self._get_index().sync()
            except Exception as e:
                raise VectorStoreError(
                    f"DatabricksVectorSearch index sync failed: {e}"
                ) from e

    def _query_delta_sync(self, query_text: str, top_k: int) -> list[RetrievalResult]:
        index = self._get_index()
        try:
            response = index.similarity_search(
                query_text=query_text,
                columns=["chunk_id", "text"],
                num_results=top_k,
            )
        except Exception as e:
            raise VectorStoreError(
                f"DatabricksVectorSearch delta_sync query failed: {e}"
            ) from e

        return self._parse_response(response)

    # ---- shared ------------------------------------------------------

    @staticmethod
    def _parse_response(response: dict) -> list[RetrievalResult]:
        """
        Parse the Databricks similarity_search response into RetrievalResult objects.

        Response shape:
          {
            "result": {
              "data_array": [[col1_val, col2_val, score], ...],
              "manifest": {"columns": [{"name": "chunk_id"}, {"name": "text"}, {"name": "score"}]}
            }
          }
        """
        try:
            manifest = response["result"]["manifest"]["columns"]
            col_names = [c["name"] for c in manifest]
            data = response["result"]["data_array"]
        except (KeyError, TypeError) as e:
            raise VectorStoreError(
                f"Unexpected DatabricksVectorSearch response format: {e}"
            ) from e

        results: list[RetrievalResult] = []
        for row in data:
            row_dict = dict(zip(col_names, row))
            score = float(row_dict.get("score", 0.0))
            text = row_dict.get("text", "")
            chunk_id = row_dict.get("chunk_id", "")
            # Remaining columns become metadata
            meta = {
                k: v for k, v in row_dict.items()
                if k not in ("score", "text", "chunk_id") and v is not None
            }
            results.append(
                RetrievalResult(
                    chunk=Chunk(text=text, index=-1, metadata={"chunk_id": chunk_id, **meta}),
                    score=score,
                )
            )
        return results
