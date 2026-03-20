"""
Local embedder using sentence-transformers (HuggingFace).
"""

from rag_framework.config.models import EmbedderConfig
from rag_framework.core.exceptions import EmbeddingError
from rag_framework.core.interfaces import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embeds text locally using a HuggingFace sentence-transformers model.

    Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality).

    No API key required. Model is downloaded on first use (~80MB for default).
    """

    def __init__(self, config: EmbedderConfig):
        self.config = config
        self.model_name = config.local_model_name
        self.device = config.device
        self.batch_size = config.batch_size
        self._model = None

    def health_check(self) -> None:
        """Verify sentence-transformers is installed and model is loadable."""
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from e
        # Load model eagerly so any download errors surface at health_check time
        self._load_model()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                device=self.device,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return [v.tolist() for v in vectors]
        except Exception as e:
            raise EmbeddingError(f"SentenceTransformer embedding failed: {e}") from e

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model
