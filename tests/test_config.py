"""
Tests for config models — validation, serialization, round-trip.
"""

import pytest

from rag_framework.config.models import (
    ChunkerImpl,
    EmbedderImpl,
    ParserImpl,
    PipelineConfig,
    RetrieverImpl,
    VectorStoreImpl,
)


def test_default_config_is_valid():
    config = PipelineConfig()
    assert config.parser.implementation == ParserImpl.pdfplumber
    assert config.chunker.implementation == ChunkerImpl.recursive
    assert config.embedder.implementation == EmbedderImpl.sentence_transformers
    assert config.vector_store.implementation == VectorStoreImpl.chromadb
    assert config.retriever.implementation == RetrieverImpl.semantic


def test_yaml_round_trip():
    config = PipelineConfig(name="test-round-trip")
    yaml_str = config.to_yaml()
    loaded = PipelineConfig.from_yaml(yaml_str)
    assert loaded.name == "test-round-trip"
    assert loaded.parser.implementation == config.parser.implementation


def test_yaml_round_trip_with_custom_values():
    config = PipelineConfig(name="custom")
    config.chunker.chunk_size = 256
    config.retriever.top_k = 10

    loaded = PipelineConfig.from_yaml(config.to_yaml())
    assert loaded.chunker.chunk_size == 256
    assert loaded.retriever.top_k == 10


def test_invalid_impl_raises():
    with pytest.raises(Exception):
        PipelineConfig.model_validate({
            "parser": {"implementation": "not_a_real_impl"}
        })


def test_hybrid_weights_are_normalised():
    """Cross-validator should normalise hybrid weights that don't sum to 1."""
    config = PipelineConfig()
    config.retriever.implementation = RetrieverImpl.hybrid
    config.retriever.bm25_weight = 1.0
    config.retriever.vector_weight = 1.0
    # Trigger the validator by re-validating
    revalidated = PipelineConfig.model_validate(config.model_dump())
    total = revalidated.retriever.bm25_weight + revalidated.retriever.vector_weight
    assert abs(total - 1.0) < 1e-9
