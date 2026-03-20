"""
Configure page — select and tune each pipeline module.

Users can:
- Choose implementations (cloud vs local) for each module.
- Tune key parameters via sliders / dropdowns.
- Export the resulting config as YAML.
- Import a previously saved YAML config.
- Run a fail-fast validation against all selected backends.
"""

from __future__ import annotations

import io

import streamlit as st

from rag_framework.config.models import (
    ChunkerConfig,
    ChunkerImpl,
    EmbedderConfig,
    EmbedderImpl,
    EvaluatorConfig,
    ParserConfig,
    ParserImpl,
    PipelineConfig,
    RetrieverConfig,
    RetrieverImpl,
    VectorStoreConfig,
    VectorStoreImpl,
)

# Session state key for the active config
_CONFIG_KEY = "pipeline_config"


def _get_config() -> PipelineConfig:
    if _CONFIG_KEY not in st.session_state:
        st.session_state[_CONFIG_KEY] = PipelineConfig()
    return st.session_state[_CONFIG_KEY]


def _set_config(config: PipelineConfig) -> None:
    st.session_state[_CONFIG_KEY] = config


def render() -> None:
    st.title("⚙️ Pipeline Configuration")
    st.caption("Select implementations and tune parameters. Export as YAML when ready.")

    config = _get_config()

    # ------------------------------------------------------------------ #
    # Import YAML                                                         #
    # ------------------------------------------------------------------ #
    with st.expander("📥 Import YAML config", expanded=False):
        uploaded = st.file_uploader("Upload a YAML config file", type=["yaml", "yml"])
        if uploaded:
            try:
                yaml_str = uploaded.read().decode("utf-8")
                config = PipelineConfig.from_yaml(yaml_str)
                _set_config(config)
                st.success("Config loaded successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to parse config: {e}")

    st.divider()

    # ------------------------------------------------------------------ #
    # Pipeline name                                                       #
    # ------------------------------------------------------------------ #
    col1, col2 = st.columns([2, 3])
    with col1:
        name = st.text_input("Pipeline name", value=config.name)
    with col2:
        description = st.text_input("Description (optional)", value=config.description)

    # ------------------------------------------------------------------ #
    # Module tabs                                                         #
    # ------------------------------------------------------------------ #
    tabs = st.tabs(["🗂 Parser", "✂️ Chunker", "🔢 Embedder", "🗄 Vector Store", "🔎 Retriever", "📏 Evaluator"])

    parser_cfg = _parser_tab(tabs[0], config.parser)
    chunker_cfg = _chunker_tab(tabs[1], config.chunker)
    embedder_cfg = _embedder_tab(tabs[2], config.embedder)
    vs_cfg = _vector_store_tab(tabs[3], config.vector_store)
    retriever_cfg = _retriever_tab(tabs[4], config.retriever)
    evaluator_cfg = _evaluator_tab(tabs[5], config.evaluator)

    # ------------------------------------------------------------------ #
    # Save config                                                         #
    # ------------------------------------------------------------------ #
    st.divider()
    col_save, col_validate, col_export = st.columns(3)

    with col_save:
        if st.button("💾 Save configuration", use_container_width=True):
            try:
                new_config = PipelineConfig(
                    name=name,
                    description=description,
                    parser=parser_cfg,
                    chunker=chunker_cfg,
                    embedder=embedder_cfg,
                    vector_store=vs_cfg,
                    retriever=retriever_cfg,
                    evaluator=evaluator_cfg,
                )
                _set_config(new_config)
                st.success("Configuration saved to session.")
            except Exception as e:
                st.error(f"Validation error: {e}")

    with col_validate:
        if st.button("✅ Validate backends", use_container_width=True):
            _run_validation(_get_config())

    with col_export:
        config_for_export = _get_config()
        yaml_bytes = config_for_export.to_yaml().encode("utf-8")
        st.download_button(
            label="📤 Export YAML",
            data=io.BytesIO(yaml_bytes),
            file_name=f"{config_for_export.name}.yaml",
            mime="text/yaml",
            use_container_width=True,
        )

    # ------------------------------------------------------------------ #
    # YAML preview                                                        #
    # ------------------------------------------------------------------ #
    with st.expander("🔍 Current config (YAML preview)", expanded=False):
        st.code(_get_config().to_yaml(), language="yaml")


# --------------------------------------------------------------------------
# Module panels
# --------------------------------------------------------------------------

def _parser_tab(tab, cfg: ParserConfig) -> ParserConfig:
    with tab:
        st.subheader("Parser")
        st.caption("Extracts text from raw documents.")

        impl = st.selectbox(
            "Implementation",
            options=[e.value for e in ParserImpl],
            index=[e.value for e in ParserImpl].index(cfg.implementation.value),
            help="pdfplumber / pymupdf = local. azure_di = cloud (requires credentials).",
        )

        new_cfg = cfg.model_copy(update={"implementation": impl})

        if impl == ParserImpl.azure_di.value:
            st.info("Azure Document Intelligence requires credentials.")
            new_cfg = new_cfg.model_copy(update={
                "azure_endpoint": st.text_input("Azure DI Endpoint", value=cfg.azure_endpoint or ""),
                "azure_api_key": st.text_input("Azure DI API Key", value=cfg.azure_api_key or "", type="password"),
                "azure_model_id": st.text_input("Model ID", value=cfg.azure_model_id),
            })

        return new_cfg


def _chunker_tab(tab, cfg: ChunkerConfig) -> ChunkerConfig:
    with tab:
        st.subheader("Chunker")
        st.caption("Splits parsed documents into chunks for embedding.")

        impl = st.selectbox(
            "Implementation",
            options=[e.value for e in ChunkerImpl],
            index=[e.value for e in ChunkerImpl].index(cfg.implementation.value),
        )

        new_cfg = cfg.model_copy(update={"implementation": impl})

        if impl == ChunkerImpl.recursive.value:
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider("Chunk size (chars)", 128, 2048, cfg.chunk_size, step=64)
            with col2:
                chunk_overlap = st.slider("Chunk overlap (chars)", 0, 512, cfg.chunk_overlap, step=16)
            new_cfg = new_cfg.model_copy(update={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})

        elif impl == ChunkerImpl.semantic.value:
            st.info("Semantic chunker uses the configured Embedder to detect topic boundaries.")
            col1, col2, col3 = st.columns(3)
            with col1:
                threshold = st.slider("Similarity threshold", 0.5, 0.99, cfg.similarity_threshold, step=0.01)
            with col2:
                min_size = st.number_input("Min chunk size (chars)", 50, 512, cfg.min_chunk_size)
            with col3:
                max_size = st.number_input("Max chunk size (chars)", 256, 4096, cfg.max_chunk_size)
            new_cfg = new_cfg.model_copy(update={
                "similarity_threshold": threshold,
                "min_chunk_size": min_size,
                "max_chunk_size": max_size,
            })

        return new_cfg


def _embedder_tab(tab, cfg: EmbedderConfig) -> EmbedderConfig:
    with tab:
        st.subheader("Embedder")
        st.caption("Converts text chunks into vector embeddings.")

        impl = st.selectbox(
            "Implementation",
            options=[e.value for e in EmbedderImpl],
            index=[e.value for e in EmbedderImpl].index(cfg.implementation.value),
        )

        new_cfg = cfg.model_copy(update={"implementation": impl})

        if impl == EmbedderImpl.sentence_transformers.value:
            new_cfg = new_cfg.model_copy(update={
                "local_model_name": st.text_input(
                    "HuggingFace model name",
                    value=cfg.local_model_name,
                    help="e.g. all-MiniLM-L6-v2, all-mpnet-base-v2",
                ),
                "device": st.selectbox("Device", ["cpu", "cuda", "mps"],
                                       index=["cpu", "cuda", "mps"].index(cfg.device)),
                "batch_size": st.number_input("Batch size", 8, 256, cfg.batch_size),
            })

        elif impl in (EmbedderImpl.openai.value, EmbedderImpl.azure_openai.value):
            new_cfg = new_cfg.model_copy(update={
                "api_key": st.text_input("API Key", value=cfg.api_key or "", type="password"),
                "model": st.text_input("Model / deployment", value=cfg.model),
            })
            if impl == EmbedderImpl.azure_openai.value:
                new_cfg = new_cfg.model_copy(update={
                    "azure_endpoint": st.text_input("Azure endpoint", value=cfg.azure_endpoint or ""),
                    "azure_deployment": st.text_input("Deployment name", value=cfg.azure_deployment or ""),
                })

        return new_cfg


def _vector_store_tab(tab, cfg: VectorStoreConfig) -> VectorStoreConfig:
    with tab:
        st.subheader("Vector Store")
        st.caption("Stores and indexes embedding vectors.")

        impl = st.selectbox(
            "Implementation",
            options=[e.value for e in VectorStoreImpl],
            index=[e.value for e in VectorStoreImpl].index(cfg.implementation.value),
        )

        new_cfg = cfg.model_copy(update={"implementation": impl})

        col1, col2 = st.columns(2)
        with col1:
            collection = st.text_input("Collection / index name", value=cfg.collection_name)
        with col2:
            emb_dim = st.number_input("Embedding dimension", 64, 4096, cfg.embedding_dim,
                                      help="Must match the embedder's output size.")
        new_cfg = new_cfg.model_copy(update={"collection_name": collection, "embedding_dim": emb_dim})

        if impl == VectorStoreImpl.chromadb.value:
            new_cfg = new_cfg.model_copy(update={
                "chromadb_path": st.text_input("ChromaDB persist path", value=cfg.chromadb_path),
            })
        elif impl == VectorStoreImpl.qdrant_local.value:
            new_cfg = new_cfg.model_copy(update={
                "qdrant_path": st.text_input("Qdrant persist path", value=cfg.qdrant_path),
            })
        elif impl == VectorStoreImpl.pinecone.value:
            new_cfg = new_cfg.model_copy(update={
                "pinecone_api_key": st.text_input("Pinecone API Key", value=cfg.pinecone_api_key or "", type="password"),
                "pinecone_index_name": st.text_input("Pinecone index name", value=cfg.pinecone_index_name),
            })
        elif impl == VectorStoreImpl.azure_search.value:
            new_cfg = new_cfg.model_copy(update={
                "azure_search_endpoint": st.text_input("Azure Search endpoint", value=cfg.azure_search_endpoint or ""),
                "azure_search_api_key": st.text_input("Azure Search API Key", value=cfg.azure_search_api_key or "", type="password"),
                "azure_search_index_name": st.text_input("Index name", value=cfg.azure_search_index_name),
            })

        return new_cfg


def _retriever_tab(tab, cfg: RetrieverConfig) -> RetrieverConfig:
    with tab:
        st.subheader("Retriever")
        st.caption("Fetches the most relevant chunks for a query.")

        impl = st.selectbox(
            "Implementation",
            options=[e.value for e in RetrieverImpl],
            index=[e.value for e in RetrieverImpl].index(cfg.implementation.value),
        )

        top_k = st.slider("Top-K results", 1, 20, cfg.top_k)
        new_cfg = cfg.model_copy(update={"implementation": impl, "top_k": top_k})

        if impl == RetrieverImpl.hybrid.value:
            st.info("Hybrid retrieval combines BM25 keyword search with vector similarity.")
            col1, col2 = st.columns(2)
            with col1:
                vw = st.slider("Vector weight", 0.0, 1.0, cfg.vector_weight, step=0.05)
            with col2:
                bw = st.slider("BM25 weight", 0.0, 1.0, cfg.bm25_weight, step=0.05)
            new_cfg = new_cfg.model_copy(update={"vector_weight": vw, "bm25_weight": bw})

        return new_cfg


def _evaluator_tab(tab, cfg: EvaluatorConfig) -> EvaluatorConfig:
    with tab:
        st.subheader("Evaluator")
        st.caption("Computes metrics for pipeline quality assessment.")

        col1, col2 = st.columns(2)
        with col1:
            retrieval = st.checkbox("Retrieval metrics (Recall, Precision, MRR, NDCG)", value=cfg.enable_retrieval_metrics)
            parsing = st.checkbox("Parsing metrics (completeness, latency)", value=cfg.enable_parsing_metrics)
            chunking = st.checkbox("Chunking metrics (coherence, variance)", value=cfg.enable_chunking_metrics)
        with col2:
            answers = st.checkbox("Answer metrics (requires LLM judge)", value=cfg.enable_answer_metrics)
            k_input = st.text_input("K values for Recall/Precision/NDCG", value=", ".join(map(str, cfg.k_values)))

        try:
            k_values = [int(k.strip()) for k in k_input.split(",") if k.strip()]
        except ValueError:
            st.warning("K values must be comma-separated integers.")
            k_values = cfg.k_values

        return cfg.model_copy(update={
            "enable_retrieval_metrics": retrieval,
            "enable_parsing_metrics": parsing,
            "enable_chunking_metrics": chunking,
            "enable_answer_metrics": answers,
            "k_values": k_values,
        })


# --------------------------------------------------------------------------
# Validation helper
# --------------------------------------------------------------------------

def _run_validation(config: PipelineConfig) -> None:
    from rag_framework.pipeline.orchestrator import RAGPipeline

    with st.spinner("Running fail-fast validation..."):
        try:
            pipeline = RAGPipeline(config)
            pipeline.validate()
            st.success("All modules passed health checks.")
        except Exception as e:
            st.error(f"Validation failed: {e}")
