"""
Cloud embedder using Azure OpenAI Embeddings.
"""

import os

from rag_framework.config.models import EmbedderConfig
from rag_framework.core.exceptions import BackendConnectionError, EmbeddingError, MissingCredentialError
from rag_framework.core.interfaces import BaseEmbedder


class AzureOpenAIEmbedder(BaseEmbedder):
    """
    Embeds text using an Azure OpenAI deployment.

    Requires:
        AZURE_OPENAI_API_KEY  — or config.api_key
        AZURE_OPENAI_ENDPOINT — or config.azure_endpoint
        config.azure_deployment must be set to your deployment name.
    """

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = config.azure_deployment
        self.api_version = config.azure_api_version

    def health_check(self) -> None:
        if not self.api_key:
            raise MissingCredentialError("AZURE_OPENAI_API_KEY")
        if not self.endpoint:
            raise MissingCredentialError("AZURE_OPENAI_ENDPOINT")
        if not self.deployment:
            raise MissingCredentialError("azure_deployment (in config)")
        try:
            from openai import AzureOpenAI  # noqa: F401
        except ImportError as e:
            raise ImportError("openai package not installed. Run: pip install openai") from e
        # TODO: make a lightweight test call to validate auth

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_client()
        try:
            response = self._client.embeddings.create(model=self.deployment, input=texts)
            items = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in items]
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                raise BackendConnectionError("Azure OpenAI", f"Auth failed: {e}") from e
            raise EmbeddingError(f"Azure OpenAI embedding failed: {e}") from e

    def _ensure_client(self) -> None:
        if not hasattr(self, "_client"):
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
            )
