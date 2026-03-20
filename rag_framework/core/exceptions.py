"""
Custom exceptions for the RAG framework.
Fail-fast design: raise early, be explicit.
"""


class RAGFrameworkError(Exception):
    """Base exception for all RAG framework errors."""


# --- Configuration Errors ---

class ConfigValidationError(RAGFrameworkError):
    """Raised when a pipeline config fails Pydantic validation."""


class MissingCredentialError(RAGFrameworkError):
    """Raised when a required API key or credential is absent."""

    def __init__(self, credential_name: str):
        super().__init__(
            f"Missing required credential: '{credential_name}'. "
            "Set it as an environment variable or in the config."
        )


# --- Backend Connectivity Errors ---

class BackendConnectionError(RAGFrameworkError):
    """Raised when a module cannot connect to its backend (cloud or local)."""

    def __init__(self, backend: str, reason: str):
        super().__init__(f"Cannot connect to backend '{backend}': {reason}")


# --- Module Errors ---

class ParsingError(RAGFrameworkError):
    """Raised when a parser fails to process a document."""


class ChunkingError(RAGFrameworkError):
    """Raised when a chunker fails to split a document."""


class EmbeddingError(RAGFrameworkError):
    """Raised when an embedder fails to produce embeddings."""


class VectorStoreError(RAGFrameworkError):
    """Raised when a vector store operation fails."""


class RetrievalError(RAGFrameworkError):
    """Raised when retrieval fails."""


class EvaluationError(RAGFrameworkError):
    """Raised when evaluation cannot be computed."""


# --- Pipeline Errors ---

class PipelineSetupError(RAGFrameworkError):
    """Raised when the pipeline cannot be assembled from the given config."""

    def __init__(self, module: str, reason: str):
        super().__init__(f"Pipeline setup failed for module '{module}': {reason}")


class UnsupportedImplementationError(RAGFrameworkError):
    """Raised when an unknown implementation key is requested."""

    def __init__(self, module: str, impl: str, available: list[str]):
        super().__init__(
            f"Unknown implementation '{impl}' for module '{module}'. "
            f"Available: {available}"
        )
