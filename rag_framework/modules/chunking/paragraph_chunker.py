"""
Paragraph-based chunker.

Splits text on blank-line boundaries (``\\n\\n``), then merges adjacent
short paragraphs up to ``chunk_size`` and splits oversized ones down to
``max_chunk_size``.
"""

from rag_framework.config.models import ChunkerConfig
from rag_framework.core.interfaces import BaseChunker, Chunk, ParsedDocument


class ParagraphChunker(BaseChunker):
    """
    Splits documents at paragraph boundaries (double newlines).

    Strategy:
    1. Split text on ``\\n\\n``.
    2. Merge consecutive short paragraphs until the merged size exceeds
       ``chunk_size``.
    3. If a single paragraph is larger than ``max_chunk_size``, split it
       on single newlines and re-merge.
    """

    def __init__(self, config: ChunkerConfig):
        self.config = config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.max_chunk_size = config.max_chunk_size

    def health_check(self) -> None:
        """No external deps — always healthy."""
        pass

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        paragraphs = [p.strip() for p in document.text.split("\n\n") if p.strip()]
        merged = self._merge_paragraphs(paragraphs)

        return [
            Chunk(
                text=text,
                index=i,
                metadata={
                    **document.metadata,
                    "chunk_strategy": "paragraph",
                    "chunk_size_cfg": self.chunk_size,
                    "max_chunk_size_cfg": self.max_chunk_size,
                },
            )
            for i, text in enumerate(merged)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Merge short paragraphs; split oversized ones."""
        # First expand any paragraph that is too large
        expanded: list[str] = []
        for para in paragraphs:
            if len(para) > self.max_chunk_size:
                expanded.extend(self._split_large(para))
            else:
                expanded.append(para)

        if not expanded:
            return []

        chunks: list[str] = []
        current = expanded[0]

        for para in expanded[1:]:
            candidate = current + "\n\n" + para
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current)
                # Carry overlap from end of current chunk into next
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current = (overlap_text + "\n\n" + para).strip() if overlap_text else para

        if current:
            chunks.append(current)

        return chunks

    def _split_large(self, text: str) -> list[str]:
        """Split a single oversized paragraph on newlines, then re-merge."""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            return []

        pieces: list[str] = []
        current = lines[0]
        for line in lines[1:]:
            candidate = current + "\n" + line
            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                pieces.append(current)
                current = line
        if current:
            pieces.append(current)
        return pieces
