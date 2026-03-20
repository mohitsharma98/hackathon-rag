# RAG Framework Demo

A **modular, evaluation-first** Retrieval-Augmented Generation (RAG) framework built for a 10-day hackathon demo.

> **Goal:** Prove that RAG design choices can be compared quantitatively, and that every major component is swappable — from a UI — without code edits.

---

## What this demo proves

| # | Claim | How |
|---|-------|-----|
| 1 | Each module supports swappable implementations | Abstract base classes + factory pattern |
| 2 | Cloud and local options exist per module | See module table below |
| 3 | Pipeline is UI-configurable, no code edits needed | Streamlit UI + YAML export/import |
| 4 | Design decisions are backed by metrics | Built-in evaluator with Recall@K, MRR, NDCG, latency |

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
| **Vector Store** | ChromaDB, Qdrant local | Pinecone, Azure AI Search |
| **Retriever** | Semantic (vector), Hybrid (BM25+vector) | — |
| **Evaluator** | Pure Python metrics | LLM judge (optional) |

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

## Demo script (10-minute walkthrough)

1. Open the UI: `uvicorn rag_framework.server.app:app --reload --port 8000`
2. **Configure** tab → select local stack → click **Validate backends** → all green.
3. **Evaluate** tab → upload a PDF → click **Run Ingestion** → see chunk count + parse time.
4. **Query** tab → ask a question → inspect retrieved chunks + scores.
5. **Compare** tab → upload `config/cloud.yaml` → run comparison → show metric table.
6. Discuss delta in Recall@5 and latency — this is your quantitative design argument.
