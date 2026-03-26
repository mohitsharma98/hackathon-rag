"""
Abstract base classes for every RAG module.

Design rules:
- Each interface is minimal — only the essential contract.
- All implementations must subclass the relevant ABC.
- health_check() is required on every module for fail-fast validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared data models (lightweight, no Pydantic overhead at runtime)
# ---------------------------------------------------------------------------

@dataclass
class ParsedDocument:
    """Output of a Parser."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # e.g. {"source": "file.pdf", "page_count": 10, "parse_time_s": 0.4}
    pages: list[str] = field(default_factory=list)
    # Per-page text, populated by parsers that support page-level extraction.
    tables: list[dict[str, Any]] = field(default_factory=list)
    # Extracted tables. Each entry: {"page_number": int, "row_count": int,
    # "column_count": int, "data": list[list[str]], "markdown": str}
    # Populated by parsers that support table extraction (e.g. azure_di).


@dataclass
class Chunk:
    """A single text chunk produced by a Chunker."""
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    # e.g. {"source": "file.pdf", "chunk_strategy": "recursive"}


@dataclass
class EmbeddedChunk:
    """A Chunk paired with its embedding vector."""
    chunk: Chunk
    embedding: list[float]


@dataclass
class RetrievalResult:
    """A single result returned by a Retriever."""
    chunk: Chunk
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Aggregated evaluation metrics for a pipeline run."""
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    # metrics example:
    # {
    #   "recall_at_5": 0.82,
    #   "precision_at_5": 0.74,
    #   "mrr": 0.68,
    #   "answer_correctness": 0.71,
    #   "answer_faithfulness": 0.85,
    #   "retrieval_latency_ms": 120.4,
    # }


# ---------------------------------------------------------------------------
# Module interfaces
# ---------------------------------------------------------------------------

class BaseParser(ABC):
    """Converts a raw file into a ParsedDocument."""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """Parse the file at *file_path* and return structured text."""

    @abstractmethod
    def health_check(self) -> None:
        """Validate connectivity / credentials. Raise on failure (fail-fast)."""


class BaseChunker(ABC):
    """Splits a ParsedDocument into a list of Chunks."""

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Split *document* into chunks."""

    @abstractmethod
    def health_check(self) -> None:
        """Validate any required resources. Raise on failure."""


class BaseEmbedder(ABC):
    """Converts text strings into embedding vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per input text."""

    @abstractmethod
    def health_check(self) -> None:
        """Validate API keys / model availability. Raise on failure."""


class BaseVectorStore(ABC):
    """Stores and queries embedding vectors."""

    @abstractmethod
    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Insert or update chunks with their embeddings."""

    @abstractmethod
    def query(self, embedding: list[float], top_k: int) -> list[RetrievalResult]:
        """Return the *top_k* most similar chunks to *embedding*."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Drop / reset the collection (useful between demo runs)."""

    @abstractmethod
    def health_check(self) -> None:
        """Validate store connectivity. Raise on failure."""


class BaseRetriever(ABC):
    """
    Retrieves relevant chunks for a query string.

    A Retriever composes an Embedder + VectorStore and may add
    re-ranking or hybrid logic on top.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Return *top_k* chunks most relevant to *query*."""

    @abstractmethod
    def health_check(self) -> None:
        """Validate all sub-components. Raise on failure."""


class BaseEvaluator(ABC):
    """Computes evaluation metrics for a retrieval or pipeline run."""

    @abstractmethod
    def evaluate(
        self,
        queries: list[str],
        expected: list[list[str]],
        retrieved: list[list[RetrievalResult]],
        answers: list[str] | None = None,
    ) -> EvaluationReport:
        """
        Compute metrics.

        Args:
            queries:   The query strings.
            expected:  Ground-truth relevant texts per query.
            retrieved: Retrieved results per query.
            answers:   Generated answers per query (optional, for end-to-end eval).
        """

    @abstractmethod
    def health_check(self) -> None:
        """Validate any external deps. Raise on failure."""
