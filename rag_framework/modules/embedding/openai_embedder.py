"""
Cloud embedder using the OpenAI Embeddings API.
"""

import os

from rag_framework.config.models import EmbedderConfig
from rag_framework.core.exceptions import BackendConnectionError, EmbeddingError, MissingCredentialError
from rag_framework.core.interfaces import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    Embeds text using OpenAI's text-embedding-3-* models.

    Requires: OPENAI_API_KEY env var (or config.api_key).
    """

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self.model = config.model

    def health_check(self) -> None:
        if not self.api_key:
            raise MissingCredentialError("OPENAI_API_KEY")
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e
        # TODO: make a lightweight test call (e.g. embed a single token) to validate auth

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_client()
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
            # Sort by index in case API reorders
            items = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in items]
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                raise BackendConnectionError("OpenAI", f"Auth failed: {e}") from e
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e

    def _ensure_client(self) -> None:
        if not hasattr(self, "_client"):
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
