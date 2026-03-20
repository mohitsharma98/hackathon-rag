"""
Evaluate page — run ingestion, query, and compare two configurations.

Sections:
1. Ingest — upload documents and run the full ingestion pipeline.
2. Query  — run a query and inspect retrieved chunks.
3. Compare — run two configs on the same eval set and display metric diffs.
"""

from __future__ import annotations

import json
import time

import streamlit as st

from rag_framework.config.models import PipelineConfig
from rag_framework.modules.evaluation.evaluator import RAGEvaluator

_CONFIG_KEY = "pipeline_config"
_CONFIG_B_KEY = "pipeline_config_b"
_PIPELINE_KEY = "active_pipeline"
_RESULTS_KEY = "last_results"
_EVAL_REPORT_A = "eval_report_a"
_EVAL_REPORT_B = "eval_report_b"


def _get_config() -> PipelineConfig:
    return st.session_state.get(_CONFIG_KEY, PipelineConfig())


def render() -> None:
    st.title("📊 Evaluate")
    st.caption("Ingest documents, run queries, and compare pipeline configurations.")

    ingest_tab, query_tab, compare_tab = st.tabs(["📥 Ingest", "🔎 Query", "⚖️ Compare"])

    with ingest_tab:
        _ingest_section()

    with query_tab:
        _query_section()

    with compare_tab:
        _compare_section()


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------

def _ingest_section() -> None:
    st.subheader("Document Ingestion")

    config = _get_config()
    st.info(
        f"**Active config:** `{config.name}` — "
        f"Parser: `{config.parser.implementation.value}` · "
        f"Chunker: `{config.chunker.implementation.value}` · "
        f"Embedder: `{config.embedder.implementation.value}` · "
        f"Store: `{config.vector_store.implementation.value}`"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Run Ingestion", use_container_width=True, disabled=not uploaded_files):
            _run_ingestion(config, uploaded_files)
    with col2:
        if st.button("🗑️ Reset Vector Store", use_container_width=True):
            _reset_store(config)


def _run_ingestion(config: PipelineConfig, uploaded_files) -> None:
    import tempfile, os

    from rag_framework.pipeline.orchestrator import RAGPipeline

    with st.spinner("Validating pipeline..."):
        try:
            pipeline = RAGPipeline(config)
            pipeline.validate()
        except Exception as e:
            st.error(f"Pipeline validation failed: {e}")
            return

    st.session_state[_PIPELINE_KEY] = pipeline
    summaries = []

    progress = st.progress(0, text="Starting ingestion...")
    for i, f in enumerate(uploaded_files):
        progress.progress((i) / len(uploaded_files), text=f"Processing {f.name}...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            summary = pipeline.ingest(tmp_path)
            summary["filename"] = f.name
            summaries.append(summary)
        except Exception as e:
            st.error(f"Failed to ingest {f.name}: {e}")
        finally:
            os.unlink(tmp_path)

    progress.progress(1.0, text="Done.")

    if summaries:
        st.success(f"Ingested {len(summaries)} document(s).")
        st.dataframe(summaries, use_container_width=True)


def _reset_store(config: PipelineConfig) -> None:
    from rag_framework.pipeline.orchestrator import RAGPipeline

    try:
        pipeline = RAGPipeline(config)
        pipeline.reset_store()
        st.session_state.pop(_PIPELINE_KEY, None)
        st.success("Vector store reset.")
    except Exception as e:
        st.error(f"Reset failed: {e}")


# --------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------

def _query_section() -> None:
    st.subheader("Query")

    pipeline = st.session_state.get(_PIPELINE_KEY)
    if pipeline is None:
        st.warning("No active pipeline. Run ingestion first.")
        return

    query = st.text_input("Enter a query", placeholder="What is the main topic of the document?")
    top_k = st.slider("Top-K results", 1, 20, pipeline.config.retriever.top_k)

    if st.button("🔍 Retrieve", disabled=not query):
        with st.spinner("Retrieving..."):
            try:
                t0 = time.perf_counter()
                results = pipeline.query(query, top_k=top_k)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                st.session_state[_RESULTS_KEY] = results
                st.success(f"Retrieved {len(results)} chunks in {elapsed_ms:.1f} ms")
            except Exception as e:
                st.error(f"Query failed: {e}")

    results = st.session_state.get(_RESULTS_KEY, [])
    if results:
        st.divider()
        for i, r in enumerate(results):
            with st.expander(f"Result {i + 1} — score: {r.score:.4f}", expanded=i == 0):
                st.markdown(r.chunk.text)
                if r.chunk.metadata:
                    st.json(r.chunk.metadata, expanded=False)


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

def _compare_section() -> None:
    st.subheader("Configuration Comparison")
    st.caption(
        "Run the same evaluation set against two different configurations "
        "and compare metrics side-by-side."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Config A** (active session config)")
        config_a = _get_config()
        st.code(f"Parser: {config_a.parser.implementation.value}\n"
                f"Chunker: {config_a.chunker.implementation.value}\n"
                f"Embedder: {config_a.embedder.implementation.value}\n"
                f"Store: {config_a.vector_store.implementation.value}\n"
                f"Retriever: {config_a.retriever.implementation.value}")

    with col_b:
        st.markdown("**Config B** (import from YAML)")
        uploaded_b = st.file_uploader("Upload Config B YAML", type=["yaml", "yml"], key="config_b_upload")
        if uploaded_b:
            try:
                config_b = PipelineConfig.from_yaml(uploaded_b.read().decode("utf-8"))
                st.session_state[_CONFIG_B_KEY] = config_b
                st.success(f"Loaded: {config_b.name}")
            except Exception as e:
                st.error(f"Failed to load Config B: {e}")

        config_b = st.session_state.get(_CONFIG_B_KEY)
        if config_b:
            st.code(f"Parser: {config_b.parser.implementation.value}\n"
                    f"Chunker: {config_b.chunker.implementation.value}\n"
                    f"Embedder: {config_b.embedder.implementation.value}\n"
                    f"Store: {config_b.vector_store.implementation.value}\n"
                    f"Retriever: {config_b.retriever.implementation.value}")

    st.divider()

    # Evaluation dataset input
    st.markdown("**Evaluation dataset**")
    st.caption("Provide queries and expected relevant texts (JSON format).")

    default_eval_set = json.dumps(
        [
            {
                "query": "What is retrieval-augmented generation?",
                "expected": ["RAG combines retrieval with language generation."],
            }
        ],
        indent=2,
    )
    eval_json = st.text_area("Eval set (JSON)", value=default_eval_set, height=200)

    if st.button("⚖️ Run Comparison", disabled=not st.session_state.get(_CONFIG_B_KEY)):
        _run_comparison(config_a, st.session_state[_CONFIG_B_KEY], eval_json)

    # Show results
    report_a = st.session_state.get(_EVAL_REPORT_A)
    report_b = st.session_state.get(_EVAL_REPORT_B)
    if report_a and report_b:
        _render_comparison(report_a, report_b)


def _run_comparison(config_a: PipelineConfig, config_b: PipelineConfig, eval_json: str) -> None:
    from rag_framework.pipeline.orchestrator import RAGPipeline

    try:
        eval_set = json.loads(eval_json)
        queries = [item["query"] for item in eval_set]
        expected = [item["expected"] for item in eval_set]
    except Exception as e:
        st.error(f"Invalid eval set JSON: {e}")
        return

    with st.spinner("Running evaluation on both configs..."):
        for label, config, report_key in [
            ("A", config_a, _EVAL_REPORT_A),
            ("B", config_b, _EVAL_REPORT_B),
        ]:
            try:
                pipeline = RAGPipeline(config)
                pipeline.validate()
                retrieved = [pipeline.query(q) for q in queries]
                evaluator = RAGEvaluator(config.evaluator)
                report = evaluator.evaluate(queries=queries, expected=expected, retrieved=retrieved)
                st.session_state[report_key] = report
            except Exception as e:
                st.error(f"Config {label} evaluation failed: {e}")
                return

    st.success("Comparison complete.")


def _render_comparison(report_a, report_b) -> None:
    comparison = RAGEvaluator.compare(report_a, report_b)

    st.divider()
    st.subheader("Metric Comparison")

    rows = []
    for metric, vals in comparison.items():
        delta = vals["delta"]
        winner = "B ✅" if delta > 0 else ("A ✅" if delta < 0 else "tie")
        rows.append({
            "Metric": metric,
            "Config A": vals["a"],
            "Config B": vals["b"],
            "Delta (B-A)": f"{delta:+.4f}",
            "Better": winner,
        })

    st.dataframe(rows, use_container_width=True)

    # Bar chart for key metrics
    import pandas as pd

    chart_metrics = [r for r in rows if "latency" not in r["Metric"]]
    if chart_metrics:
        df = pd.DataFrame(chart_metrics).set_index("Metric")[["Config A", "Config B"]]
        st.bar_chart(df)
