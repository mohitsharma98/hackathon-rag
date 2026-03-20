"""
Pydantic config models for the RAG framework.

Each module has a typed config block. The top-level PipelineConfig
combines them and is the single source of truth for a pipeline run.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums for valid implementation keys
# ---------------------------------------------------------------------------

class ParserImpl(str, Enum):
    azure_di = "azure_di"       # Azure Document Intelligence (cloud)
    pdfplumber = "pdfplumber"   # pdfplumber (local)
    pymupdf = "pymupdf"         # PyMuPDF (local)


class ChunkerImpl(str, Enum):
    recursive = "recursive"     # Recursive character splitting
    semantic = "semantic"       # Semantic / embedding-guided splitting


class EmbedderImpl(str, Enum):
    openai = "openai"                       # OpenAI text-embedding-3-* (cloud)
    azure_openai = "azure_openai"           # Azure OpenAI embeddings (cloud)
    sentence_transformers = "sentence_transformers"  # Local HuggingFace model


class VectorStoreImpl(str, Enum):
    pinecone = "pinecone"       # Pinecone (cloud)
    azure_search = "azure_search"  # Azure AI Search (cloud)
    chromadb = "chromadb"       # ChromaDB (local)
    qdrant_local = "qdrant_local"  # Qdrant local (local)


class RetrieverImpl(str, Enum):
    semantic = "semantic"       # Pure vector similarity
    hybrid = "hybrid"           # BM25 + vector (optional, marked in impl)


# ---------------------------------------------------------------------------
# Per-module config blocks
# ---------------------------------------------------------------------------

class ParserConfig(BaseModel):
    implementation: ParserImpl = ParserImpl.pdfplumber
    # Azure Document Intelligence options
    azure_endpoint: str | None = None
    azure_api_key: str | None = None
    azure_model_id: str = "prebuilt-document"
    # Local options
    extract_images: bool = False
    # Shared
    extra: dict[str, Any] = Field(default_factory=dict)


class ChunkerConfig(BaseModel):
    implementation: ChunkerImpl = ChunkerImpl.recursive
    # Recursive chunker
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    # Semantic chunker
    similarity_threshold: float = 0.75
    min_chunk_size: int = 100
    max_chunk_size: int = 1024
    # Shared
    extra: dict[str, Any] = Field(default_factory=dict)


class EmbedderConfig(BaseModel):
    implementation: EmbedderImpl = EmbedderImpl.sentence_transformers
    # OpenAI / Azure OpenAI
    api_key: str | None = None
    model: str = "text-embedding-3-small"
    azure_endpoint: str | None = None
    azure_api_version: str = "2024-02-01"
    azure_deployment: str | None = None
    # Sentence Transformers
    local_model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    # Shared
    extra: dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    implementation: VectorStoreImpl = VectorStoreImpl.chromadb
    collection_name: str = "rag_demo"
    # Pinecone
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "rag-demo"
    pinecone_environment: str | None = None
    # Azure AI Search
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None
    azure_search_index_name: str = "rag-demo"
    # ChromaDB
    chromadb_path: str = "./.chromadb"
    # Qdrant local
    qdrant_path: str = "./.qdrant"
    qdrant_port: int = 6333
    # Shared
    embedding_dim: int = 384   # Must match the embedder output size
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrieverConfig(BaseModel):
    implementation: RetrieverImpl = RetrieverImpl.semantic
    top_k: int = 5
    # Hybrid retrieval
    bm25_weight: float = 0.3    # weight for BM25 score in hybrid
    vector_weight: float = 0.7  # weight for vector score in hybrid
    # Shared
    extra: dict[str, Any] = Field(default_factory=dict)


class EvaluatorConfig(BaseModel):
    # Which metric groups to compute
    enable_retrieval_metrics: bool = True   # Recall@K, Precision@K, MRR
    enable_parsing_metrics: bool = True     # Parsing accuracy, latency
    enable_chunking_metrics: bool = True    # Chunk coherence, boundary precision
    enable_answer_metrics: bool = False     # Requires an LLM judge
    # Retrieval thresholds
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    # LLM judge (for answer metrics, optional)
    judge_model: str = "gpt-4o-mini"
    judge_api_key: str | None = None
    # Shared
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """
    Single source of truth for a full RAG pipeline configuration.
    Validated by Pydantic at load time — fail-fast.
    """
    name: str = "rag-demo"
    description: str = ""

    parser: ParserConfig = Field(default_factory=ParserConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)

    @model_validator(mode="after")
    def _cross_validate(self) -> "PipelineConfig":
        """Cross-field validation: catch obvious mismatches early."""
        # Hybrid retrieval requires a vector store (always true here, just an example)
        if self.retriever.implementation == RetrieverImpl.hybrid:
            if self.retriever.bm25_weight + self.retriever.vector_weight != 1.0:
                # Normalize silently rather than reject
                total = self.retriever.bm25_weight + self.retriever.vector_weight
                self.retriever.bm25_weight /= total
                self.retriever.vector_weight /= total
        return self

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Export config to a YAML string."""
        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "PipelineConfig":
        """Load and validate config from a YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_file(cls, path: str) -> "PipelineConfig":
        """Load and validate config from a YAML file."""
        with open(path) as f:
            return cls.from_yaml(f.read())

    def to_yaml_file(self, path: str) -> None:
        """Write config to a YAML file."""
        with open(path, "w") as f:
            f.write(self.to_yaml())
