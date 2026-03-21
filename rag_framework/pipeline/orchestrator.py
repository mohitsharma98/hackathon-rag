"""
Pipeline orchestrator — assembles and runs the full RAG pipeline.

Responsibilities:
1. Read a PipelineConfig.
2. Instantiate the correct implementation for each module.
3. Validate all modules (health_check) before doing any real work — fail-fast.
4. Expose ingest() and query() as the two primary operations.
"""

from __future__ import annotations

import logging

from rag_framework.config.models import (
    ChunkerImpl,
    EmbedderImpl,
    ParserImpl,
    PipelineConfig,
    RetrieverImpl,
    VectorStoreImpl,
)
from rag_framework.core.exceptions import PipelineSetupError, UnsupportedImplementationError
from rag_framework.core.interfaces import (
    BaseChunker,
    BaseEmbedder,
    BaseEvaluator,
    BaseParser,
    BaseRetriever,
    BaseVectorStore,
    Chunk,
    EmbeddedChunk,
    ParsedDocument,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Assembles and orchestrates the full RAG pipeline.

    Example:
        config = PipelineConfig.from_yaml_file("config/default.yaml")
        pipeline = RAGPipeline(config)
        pipeline.validate()          # fail-fast health checks
        pipeline.ingest("docs/")     # parse → chunk → embed → store
        results = pipeline.query("What is X?")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.parser: BaseParser = _build_parser(config)
        self.chunker: BaseChunker = _build_chunker(config, self)
        self.embedder: BaseEmbedder = _build_embedder(config)
        self.vector_store: BaseVectorStore = _build_vector_store(config)
        self.retriever: BaseRetriever = _build_retriever(config, self)
        self.evaluator: BaseEvaluator = _build_evaluator(config)

        # Track all ingested chunks (needed for hybrid BM25 corpus)
        self._ingested_chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Fail-fast validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Run health checks on every module.
        Raises immediately if any module cannot connect or is misconfigured.
        """
        logger.info("Validating pipeline: %s", self.config.name)
        modules = {
            "parser": self.parser,
            "chunker": self.chunker,
            "embedder": self.embedder,
            "vector_store": self.vector_store,
            "retriever": self.retriever,
            "evaluator": self.evaluator,
        }
        for name, module in modules.items():
            try:
                module.health_check()
                logger.info("  ✓ %s", name)
            except Exception as e:
                raise PipelineSetupError(name, str(e)) from e
        logger.info("Pipeline validation passed.")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> dict:
        """
        Full ingestion flow: parse → chunk → embed → upsert.

        Args:
            file_path: Path to a single document file.

        Returns:
            A summary dict with counts and timing metadata.
        """
        logger.info("Ingesting: %s", file_path)

        # 1. Parse
        document: ParsedDocument = self.parser.parse(file_path)
        logger.info("  Parsed: %d chars", len(document.text))

        # 2. Chunk
        chunks: list[Chunk] = self.chunker.chunk(document)
        logger.info("  Chunked: %d chunks", len(chunks))

        # 3. Embed
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        embedded = [EmbeddedChunk(chunk=c, embedding=e) for c, e in zip(chunks, embeddings)]
        logger.info("  Embedded: %d vectors", len(embedded))

        # 4. Upsert
        self.vector_store.upsert(embedded)
        logger.info("  Stored in vector store.")

        # Track for hybrid retrieval
        self._ingested_chunks.extend(chunks)
        if hasattr(self.retriever, "set_corpus"):
            self.retriever.set_corpus(self._ingested_chunks)

        return {
            "file": file_path,
            "page_count": document.metadata.get("page_count"),
            "char_count": len(document.text),
            "chunk_count": len(chunks),
            "parse_time_s": document.metadata.get("parse_time_s"),
        }

    def ingest_directory(self, directory: str, glob_pattern: str = "**/*.pdf") -> list[dict]:
        """
        Ingest all files matching *glob_pattern* under *directory*.

        TODO: Add parallel ingestion with concurrent.futures.
        """
        import glob as glob_module

        files = glob_module.glob(f"{directory}/{glob_pattern}", recursive=True)
        summaries = []
        for f in files:
            summaries.append(self.ingest(f))
        return summaries

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_text: str, top_k: int | None = None) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for *query_text*.
        """
        return self.retriever.retrieve(query_text, top_k=top_k)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset_store(self) -> None:
        """Drop all vectors from the store (useful between demo runs)."""
        self.vector_store.delete_collection()
        self._ingested_chunks.clear()
        logger.info("Vector store reset.")


# ---------------------------------------------------------------------------
# Factory functions — one per module
# ---------------------------------------------------------------------------

def _build_parser(config: PipelineConfig) -> BaseParser:
    impl = config.parser.implementation
    if impl == ParserImpl.pdfplumber:
        from rag_framework.modules.parsing.pdfplumber_parser import PDFPlumberParser
        return PDFPlumberParser(config.parser)
    elif impl == ParserImpl.pymupdf:
        from rag_framework.modules.parsing.pymupdf_parser import PyMuPDFParser
        return PyMuPDFParser(config.parser)
    elif impl == ParserImpl.azure_di:
        from rag_framework.modules.parsing.azure_di_parser import AzureDIParser
        return AzureDIParser(config.parser)
    raise UnsupportedImplementationError("parser", impl, [e.value for e in ParserImpl])


def _build_chunker(config: PipelineConfig, pipeline: "RAGPipeline") -> BaseChunker:
    impl = config.chunker.implementation
    if impl == ChunkerImpl.recursive:
        from rag_framework.modules.chunking.recursive_chunker import RecursiveChunker
        return RecursiveChunker(config.chunker)
    elif impl == ChunkerImpl.semantic:
        from rag_framework.modules.chunking.semantic_chunker import SemanticChunker
        # Semantic chunker needs an embedder — reuse the pipeline's embedder
        embedder = _build_embedder(config)
        return SemanticChunker(config.chunker, embedder)
    raise UnsupportedImplementationError("chunker", impl, [e.value for e in ChunkerImpl])


def _build_embedder(config: PipelineConfig) -> BaseEmbedder:
    impl = config.embedder.implementation
    if impl == EmbedderImpl.sentence_transformers:
        from rag_framework.modules.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(config.embedder)
    elif impl == EmbedderImpl.openai:
        from rag_framework.modules.embedding.openai_embedder import OpenAIEmbedder
        return OpenAIEmbedder(config.embedder)
    elif impl == EmbedderImpl.azure_openai:
        from rag_framework.modules.embedding.azure_openai_embedder import AzureOpenAIEmbedder
        return AzureOpenAIEmbedder(config.embedder)
    raise UnsupportedImplementationError("embedder", impl, [e.value for e in EmbedderImpl])


def _build_vector_store(config: PipelineConfig) -> BaseVectorStore:
    impl = config.vector_store.implementation
    if impl == VectorStoreImpl.chromadb:
        from rag_framework.modules.vectorstore.chromadb_store import ChromaDBStore
        return ChromaDBStore(config.vector_store)
    elif impl == VectorStoreImpl.qdrant_local:
        from rag_framework.modules.vectorstore.qdrant_store import QdrantLocalStore
        return QdrantLocalStore(config.vector_store)
    elif impl == VectorStoreImpl.pinecone:
        from rag_framework.modules.vectorstore.pinecone_store import PineconeStore
        return PineconeStore(config.vector_store)
    elif impl == VectorStoreImpl.azure_search:
        from rag_framework.modules.vectorstore.azure_search_store import AzureSearchStore
        return AzureSearchStore(config.vector_store)
    elif impl == VectorStoreImpl.databricks:
        from rag_framework.modules.vectorstore.databricks_vector_search_store import DatabricksVectorSearchStore
        return DatabricksVectorSearchStore(config.vector_store)
    raise UnsupportedImplementationError("vector_store", impl, [e.value for e in VectorStoreImpl])


def _build_retriever(config: PipelineConfig, pipeline: "RAGPipeline") -> BaseRetriever:
    impl = config.retriever.implementation
    if impl == RetrieverImpl.semantic:
        from rag_framework.modules.retrieval.semantic_retriever import SemanticRetriever
        return SemanticRetriever(config.retriever, pipeline.embedder, pipeline.vector_store)
    elif impl == RetrieverImpl.hybrid:
        from rag_framework.modules.retrieval.hybrid_retriever import HybridRetriever
        return HybridRetriever(config.retriever, pipeline.embedder, pipeline.vector_store)
    raise UnsupportedImplementationError("retriever", impl, [e.value for e in RetrieverImpl])


def _build_evaluator(config: PipelineConfig) -> BaseEvaluator:
    from rag_framework.modules.evaluation.evaluator import RAGEvaluator
    return RAGEvaluator(config.evaluator)
