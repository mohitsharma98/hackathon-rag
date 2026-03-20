"""
Recursive character-based chunker.

Splits text by progressively smaller separators until chunks are within size limits.
No external dependencies — pure Python.
"""

from rag_framework.config.models import ChunkerConfig
from rag_framework.core.interfaces import BaseChunker, Chunk, ParsedDocument


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text on a hierarchy of separators.

    Strategy:
    1. Try splitting on paragraphs ("\n\n").
    2. If a piece is still too large, split on lines ("\n").
    3. Continue with sentences, words, characters.
    4. Merge small adjacent pieces with overlap.
    """

    def __init__(self, config: ChunkerConfig):
        self.config = config
        self.separators = config.separators
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap

    def health_check(self) -> None:
        """No external deps — always healthy."""
        pass

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        raw_chunks = self._split(document.text, self.separators)
        merged = self._merge(raw_chunks)

        return [
            Chunk(
                text=text,
                index=i,
                metadata={
                    **document.metadata,
                    "chunk_strategy": "recursive",
                    "chunk_size_cfg": self.chunk_size,
                    "chunk_overlap_cfg": self.chunk_overlap,
                },
            )
            for i, text in enumerate(merged)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the next separator that fits."""
        if not separators or len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0]
        remaining_seps = separators[1:]

        pieces = text.split(sep)
        result: list[str] = []
        for piece in pieces:
            if len(piece) <= self.chunk_size:
                if piece.strip():
                    result.append(piece)
            else:
                result.extend(self._split(piece, remaining_seps))
        return result

    def _merge(self, pieces: list[str]) -> list[str]:
        """
        Merge short pieces into chunks of at most *chunk_size*,
        with *chunk_overlap* characters of context carried forward.
        """
        if not pieces:
            return []

        chunks: list[str] = []
        current = pieces[0]

        for piece in pieces[1:]:
            candidate = current + " " + piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current.strip())
                # Carry overlap from end of current chunk
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current = (overlap_text + " " + piece).strip()

        if current.strip():
            chunks.append(current.strip())

        return chunks
