"""
In-memory session state for the demo server.
Single-process only — fine for a local hackathon demo.
"""

from __future__ import annotations
from typing import Any

from rag_framework.config.models import PipelineConfig
from rag_framework.core.interfaces import RetrievalResult

_state: dict[str, Any] = {
    "config": PipelineConfig(),
    "pipeline": None,
    "ingestion_summaries": [],
    "last_query_results": [],
}


def get_config() -> PipelineConfig:
    return _state["config"]


def set_config(config: PipelineConfig) -> None:
    _state["config"] = config
    # Reset pipeline when config changes so next op rebuilds it
    _state["pipeline"] = None


def get_pipeline():
    return _state["pipeline"]


def set_pipeline(pipeline) -> None:
    _state["pipeline"] = pipeline


def get_ingestion_summaries() -> list[dict]:
    return _state["ingestion_summaries"]


def add_ingestion_summary(summary: dict) -> None:
    _state["ingestion_summaries"].append(summary)


def clear_ingestion_summaries() -> None:
    _state["ingestion_summaries"].clear()


def get_query_results() -> list:
    return _state["last_query_results"]


def set_query_results(results: list) -> None:
    _state["last_query_results"] = results
