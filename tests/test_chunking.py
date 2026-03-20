"""
Tests for chunking implementations.
"""

import pytest

from rag_framework.config.models import ChunkerConfig, ChunkerImpl
from rag_framework.core.interfaces import ParsedDocument
from rag_framework.modules.chunking.recursive_chunker import RecursiveChunker


def _make_document(text: str) -> ParsedDocument:
    return ParsedDocument(text=text, metadata={"source": "test.pdf"})


def test_recursive_chunker_basic():
    cfg = ChunkerConfig(implementation=ChunkerImpl.recursive, chunk_size=100, chunk_overlap=10)
    chunker = RecursiveChunker(cfg)
    doc = _make_document("Hello world. " * 50)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= cfg.chunk_size + cfg.chunk_overlap + 50  # some flex for merging


def test_recursive_chunker_empty_doc():
    cfg = ChunkerConfig(implementation=ChunkerImpl.recursive)
    chunker = RecursiveChunker(cfg)
    chunks = chunker.chunk(_make_document(""))
    assert chunks == []


def test_recursive_chunker_short_doc():
    cfg = ChunkerConfig(implementation=ChunkerImpl.recursive, chunk_size=1000)
    chunker = RecursiveChunker(cfg)
    text = "A short document."
    chunks = chunker.chunk(_make_document(text))
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_recursive_chunker_metadata_propagated():
    cfg = ChunkerConfig(implementation=ChunkerImpl.recursive, chunk_size=50)
    chunker = RecursiveChunker(cfg)
    doc = _make_document("Paragraph one.\n\nParagraph two.\n\nParagraph three.")
    chunks = chunker.chunk(doc)
    for c in chunks:
        assert c.metadata["source"] == "test.pdf"
        assert c.metadata["chunk_strategy"] == "recursive"
