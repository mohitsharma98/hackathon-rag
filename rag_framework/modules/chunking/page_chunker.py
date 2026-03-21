"""
Page-wise chunker.

Each page of the source document becomes one chunk.
Falls back to splitting on double-newlines when page-level text is unavailable.
"""

from rag_framework.config.models import ChunkerConfig
from rag_framework.core.interfaces import BaseChunker, Chunk, ParsedDocument


class PageChunker(BaseChunker):
    """
    Produces one chunk per page.

    If the parser populated ``document.pages``, each element becomes a chunk.
    Otherwise, the full text is split on ``"\\n\\n"`` as a page-boundary proxy.
    """

    def __init__(self, config: ChunkerConfig):
        self.config = config

    def health_check(self) -> None:
        """No external deps — always healthy."""
        pass

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        pages = document.pages if document.pages else document.text.split("\n\n")
        chunks: list[Chunk] = []
        chunk_index = 0
        for page_num, page_text in enumerate(pages):
            text = page_text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    index=chunk_index,
                    metadata={
                        **document.metadata,
                        "chunk_strategy": "pagewise",
                        "page_number": page_num + 1,
                    },
                )
            )
            chunk_index += 1
        return chunks
