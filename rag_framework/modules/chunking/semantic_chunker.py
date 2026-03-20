"""
Semantic chunker.

Splits text at sentence boundaries where the embedding similarity drops
below a threshold, keeping semantically coherent passages together.

Requires an Embedder to compute sentence-level similarity.
"""

import re

from rag_framework.config.models import ChunkerConfig
from rag_framework.core.exceptions import ChunkingError
from rag_framework.core.interfaces import BaseChunker, BaseEmbedder, Chunk, ParsedDocument


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(BaseChunker):
    """
    Embedding-guided semantic chunker.

    Algorithm:
    1. Split document into sentences.
    2. Embed each sentence.
    3. Compute cosine similarity between adjacent sentences.
    4. Start a new chunk when similarity drops below *similarity_threshold*.
    5. Enforce min/max chunk size by merging or splitting as needed.

    TODO: Replace sentence splitter with a proper NLP sentence tokenizer
          (e.g. spaCy or nltk) for better accuracy on edge cases.
    """

    def __init__(self, config: ChunkerConfig, embedder: BaseEmbedder):
        self.config = config
        self.embedder = embedder
        self.similarity_threshold = config.similarity_threshold
        self.min_chunk_size = config.min_chunk_size
        self.max_chunk_size = config.max_chunk_size

    def health_check(self) -> None:
        """Delegate to the embedder."""
        self.embedder.health_check()

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        sentences = self._split_sentences(document.text)
        if not sentences:
            return []

        try:
            embeddings = self.embedder.embed(sentences)
        except Exception as e:
            raise ChunkingError(f"Embedding failed during semantic chunking: {e}") from e

        # Group sentences into chunks by similarity drop
        groups = self._group_by_similarity(sentences, embeddings)

        # Enforce size constraints
        groups = self._enforce_size(groups)

        return [
            Chunk(
                text=text,
                index=i,
                metadata={
                    **document.metadata,
                    "chunk_strategy": "semantic",
                    "similarity_threshold": self.similarity_threshold,
                },
            )
            for i, text in enumerate(groups)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        """
        Naive sentence splitter on '.', '!', '?'.
        TODO: Replace with spaCy / nltk for production quality.
        """
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if s.strip()]

    def _group_by_similarity(
        self,
        sentences: list[str],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Merge consecutive sentences; break when similarity drops."""
        groups: list[str] = []
        current_sentences = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim >= self.similarity_threshold:
                current_sentences.append(sentences[i])
            else:
                groups.append(" ".join(current_sentences))
                current_sentences = [sentences[i]]

        if current_sentences:
            groups.append(" ".join(current_sentences))

        return groups

    def _enforce_size(self, groups: list[str]) -> list[str]:
        """
        Merge too-small groups with their neighbour; split too-large ones.
        TODO: Improve splitting strategy for over-limit chunks.
        """
        result: list[str] = []
        for group in groups:
            if len(group) < self.min_chunk_size and result:
                result[-1] = result[-1] + " " + group
            elif len(group) > self.max_chunk_size:
                # Hard split — TODO: use recursive strategy instead
                for start in range(0, len(group), self.max_chunk_size):
                    result.append(group[start : start + self.max_chunk_size])
            else:
                result.append(group)
        return [r.strip() for r in result if r.strip()]
