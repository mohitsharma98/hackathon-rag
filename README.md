# RAG Framework

A **modular, evaluation-first** Retrieval-Augmented Generation (RAG) framework.

> **Goal:** RAG design choices can be compared quantitatively, and every major component is swappable — from a UI — without code edits.

---

## Project structure

```
hackathon-rag/
├── config/
│   ├── default.yaml          # Local stack preset
│   └── cloud.yaml            # Cloud stack preset
├── rag_framework/
│   ├── core/
│   │   ├── interfaces.py     # Abstract base classes (Parser, Chunker, …)
│   │   └── exceptions.py     # Custom exceptions (fail-fast)
│   ├── config/
│   │   └── models.py         # Pydantic config models + YAML I/O
│   ├── modules/
│   │   ├── parsing/          # pdfplumber · PyMuPDF · Azure DI
│   │   ├── chunking/         # Recursive · Semantic
│   │   ├── embedding/        # OpenAI · Azure OpenAI · sentence-transformers
│   │   ├── vectorstore/      # ChromaDB · Qdrant · Pinecone · Azure AI Search
│   │   ├── retrieval/        # Semantic · Hybrid (BM25 + vector)
│   │   └── evaluation/       # Metrics + RAGEvaluator + comparison
│   ├── pipeline/
│   │   └── orchestrator.py   # RAGPipeline — assembles and runs everything
│   └── ui/
│       ├── app.py            # Streamlit entry point
│       └── pages/
│           ├── configure.py  # Module configuration UI
│           └── evaluate.py   # Ingest · Query · Compare
├── tests/
│   ├── test_config.py
│   ├── test_chunking.py
│   └── test_metrics.py
├── pyproject.toml
└── requirements.txt
```

---

## Module implementations

| Module | Local / On-prem | Cloud |
|--------|-----------------|-------|
| **Parser** | `pdfplumber`, `PyMuPDF` | Azure Document Intelligence |
| **Chunker** | Recursive (pure Python) | Semantic (uses Embedder) |
| **Embedder** | `sentence-transformers` | OpenAI, Azure OpenAI |
| **Vector Store** | ChromaDB, Qdrant local | Pinecone, Azure AI Search, Databricks |
| **Retriever** | Semantic (vector), Hybrid (BM25+vector) | — |
| **Evaluator** | Pure Python metrics | LLM judge (optional) |

---

## Interface → Implementation map

Every module is defined as an abstract base class in `rag_framework/core/interfaces.py`. Concrete implementations live under `rag_framework/modules/` and are selected at runtime via the config enum.

```
core/interfaces.py                    modules/
─────────────────────────────────────────────────────────────────────────────

BaseParser                            parsing/
  .parse(file_path) → ParsedDocument  ├── AzureDIParser          (azure_di)
  .health_check()                     │     creds: AZURE_DI_ENDPOINT + AZURE_DI_KEY
                                      │     strength: OCR, tables, scanned docs
                                      ├── pdfplumber parser       (pdfplumber)
                                      └── PyMuPDF parser          (pymupdf)

─────────────────────────────────────────────────────────────────────────────

BaseChunker                           chunking/
  .chunk(doc) → list[Chunk]           ├── RecursiveChunker        (recursive)
  .health_check()                     │     params: chunk_size, chunk_overlap, separators
                                      │     algo: split [\n\n → \n → ". " → " "] recursively
                                      │             then merge + overlap pass
                                      ├── SemanticChunker         (semantic)
                                      │     params: similarity_threshold, min/max_chunk_size
                                      │     algo: embedding-guided boundary detection
                                      ├── pagewise chunker        (pagewise)
                                      └── paragraph chunker       (paragraph)

─────────────────────────────────────────────────────────────────────────────

BaseEmbedder                          embedding/
  .embed(texts) → list[list[float]]   ├── SentenceTransformerEmbedder  (sentence_transformers)
  .health_check()                     │     model: all-MiniLM-L6-v2  dim=384  runs locally
                                      ├── OpenAIEmbedder               (openai)
                                      │     model: text-embedding-3-small/large  dim=1536
                                      │     creds: OPENAI_API_KEY
                                      └── AzureOpenAIEmbedder          (azure_openai)
                                            creds: AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT

─────────────────────────────────────────────────────────────────────────────

BaseVectorStore                       vectorstore/
  .upsert(embedded_chunks)            ├── ChromaDBStore           (chromadb)
  .query(embedding, top_k)            │     local persistent, no creds, cosine distance
  .delete_collection()                ├── QdrantLocalStore        (qdrant_local)
  .health_check()                     │     local file-based
                                      ├── PineconeStore           (pinecone)
                                      │     creds: PINECONE_API_KEY  (index must pre-exist)
                                      ├── AzureSearchStore        (azure_search)
                                      │     creds: AZURE_SEARCH_ENDPOINT + AZURE_SEARCH_API_KEY
                                      └── DatabricksVectorSearchStore  (databricks)
                                            creds: DATABRICKS_HOST + DATABRICKS_TOKEN
                                            ├── direct_access
                                            │     upsert → sends pre-computed vectors
                                            │     query  → similarity_search(vector)
                                            └── delta_sync
                                                  upsert → writes rows to Delta table
                                                           Databricks auto-embeds via FM endpoint
                                                  query  → similarity_search(query_text)

─────────────────────────────────────────────────────────────────────────────

BaseRetriever                         retrieval/
  .retrieve(query) → list[Result]     ├── SemanticRetriever       (semantic)
  .health_check()                     │     composes: Embedder + VectorStore
                                      │     flow: query → embed → .query(vector)
                                      └── HybridRetriever         (hybrid)
                                            composes: Embedder + VectorStore + BM25
                                            flow: embed path  →  vector score  × 0.7
                                                  BM25 path   →  keyword score × 0.3
                                            requires: rank_bm25, corpus at init

─────────────────────────────────────────────────────────────────────────────

BaseEvaluator                         evaluation/
  .evaluate(queries, expected,        └── RAGEvaluator
            retrieved, answers)             metric groups (each toggleable in config):
    → EvaluationReport                      ├── retrieval  → Recall@K, Precision@K, MRR, NDCG
  .health_check()                           ├── parsing    → char count, latency, completeness
                                            ├── chunking   → chunk count, avg length, variance
                                            └── answer     → LLM judge (gpt-4o-mini, off by default)
```

### Key design rule

Every implementation follows the same four-step contract:

```
__init__(config)     store config, set all clients to None (lazy init)
health_check()       validate credentials and imports — fail-fast before any work
core method(...)     parse / chunk / embed / upsert / query / evaluate
_get_client()        SDK client created on first use, then cached
```

The `Retriever` is the only module that **composes** others — it holds a `BaseEmbedder` and a `BaseVectorStore` instance rather than owning its own SDK client.

---

## Quick start

### 1. Install (local stack)

```bash
pip install -e ".[local,ui]"
```

### 2. Launch the UI

```bash
uvicorn rag_framework.server.app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

### 3. Run via Python API

```python
from rag_framework.config.models import PipelineConfig
from rag_framework.pipeline.orchestrator import RAGPipeline

config = PipelineConfig.from_yaml_file("config/default.yaml")
pipeline = RAGPipeline(config)
pipeline.validate()               # fail-fast — raises if any module is broken

summary = pipeline.ingest("docs/my_document.pdf")
print(summary)

results = pipeline.query("What is the main finding?", top_k=5)
for r in results:
    print(f"[{r.score:.3f}] {r.chunk.text[:120]}")
```

### 4. Compare two configs

```python
from rag_framework.modules.evaluation.evaluator import RAGEvaluator
from rag_framework.config.models import PipelineConfig
from rag_framework.pipeline.orchestrator import RAGPipeline

queries = ["What is X?", "How does Y work?"]
expected = [["relevant passage A"], ["relevant passage B"]]

def eval_config(config_path):
    config = PipelineConfig.from_yaml_file(config_path)
    pipeline = RAGPipeline(config)
    pipeline.validate()
    retrieved = [pipeline.query(q) for q in queries]
    evaluator = RAGEvaluator(config.evaluator)
    return evaluator.evaluate(queries=queries, expected=expected, retrieved=retrieved)

report_local = eval_config("config/default.yaml")
report_cloud = eval_config("config/cloud.yaml")

comparison = RAGEvaluator.compare(report_local, report_cloud)
for metric, vals in comparison.items():
    print(f"{metric:30s}  local={vals['a']:.3f}  cloud={vals['b']:.3f}  Δ={vals['delta']:+.3f}")
```

---

## Evaluation metrics

### Retrieval
| Metric | Description |
|--------|-------------|
| `recall_at_k` | Fraction of relevant docs in top-K |
| `precision_at_k` | Fraction of top-K that are relevant |
| `mrr` | Mean Reciprocal Rank |
| `ndcg_at_k` | Normalised Discounted Cumulative Gain |
| `avg_retrieval_latency_ms` | End-to-end query latency |

### Parsing *(via `ParsingMetrics`)*
| Metric | Description |
|--------|-------------|
| `char_count` | Raw character count |
| `word_count` | Word count |
| `completeness_ratio` | Word-overlap vs reference |
| `latency_score` | Normalised parse speed |

### Chunking *(via `ChunkingMetrics`)*
| Metric | Description |
|--------|-------------|
| `chunk_count` | Number of chunks produced |
| `avg_chunk_length` | Mean characters per chunk |
| `size_variance` | Variance in chunk sizes |
| `boundary_precision` | *(TODO)* Alignment to semantic boundaries |
| `coherence_score` | *(TODO)* Intra-chunk embedding coherence |

### Answer quality *(optional, LLM judge)*
| Metric | Description |
|--------|-------------|
| `answer_correctness` | *(TODO)* Factual accuracy vs ground truth |
| `answer_faithfulness` | *(TODO)* Grounded in retrieved context |

---

## Fail-fast design

Every module implements `health_check()`. Call `pipeline.validate()` before ingestion to catch:
- Missing API keys / credentials
- Unreachable backends (Pinecone, Azure, etc.)
- Missing Python packages
- Dimension mismatches between embedder and vector store

Errors surface as typed exceptions from `rag_framework.core.exceptions`.

---

## Environment variables

| Variable | Module | Required for |
|----------|--------|--------------|
| `OPENAI_API_KEY` | Embedder | OpenAI embeddings |
| `AZURE_OPENAI_API_KEY` | Embedder | Azure OpenAI embeddings |
| `AZURE_OPENAI_ENDPOINT` | Embedder | Azure OpenAI embeddings |
| `AZURE_DI_ENDPOINT` | Parser | Azure Document Intelligence |
| `AZURE_DI_KEY` | Parser | Azure Document Intelligence |
| `PINECONE_API_KEY` | Vector Store | Pinecone |
| `AZURE_SEARCH_ENDPOINT` | Vector Store | Azure AI Search |
| `AZURE_SEARCH_API_KEY` | Vector Store | Azure AI Search |
| `DATABRICKS_HOST` | Vector Store | Databricks Vector Search |
| `DATABRICKS_TOKEN` | Vector Store | Databricks Vector Search |

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Extending the framework

To add a new implementation (e.g. a new parser):

1. Create `rag_framework/modules/parsing/my_parser.py` subclassing `BaseParser`.
2. Add `my_impl` to the `ParserImpl` enum in `config/models.py`.
3. Add a branch in `_build_parser()` in `pipeline/orchestrator.py`.
4. It will automatically appear in the UI dropdown.

---

