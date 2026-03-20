/* ═══════════════════════════════════════════════════════
   RAG Framework Dashboard — Alpine.js data + utilities
   ═══════════════════════════════════════════════════════ */

'use strict';

// ── Theme toggle ────────────────────────────────────────────

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'light' ? 'dark' : 'light';
  if (next === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
    localStorage.setItem('theme', 'light');
  } else {
    document.documentElement.removeAttribute('data-theme');
    localStorage.removeItem('theme');
  }
  _updateThemeIcon(next);
}

function _updateThemeIcon(theme) {
  const icon = document.getElementById('theme-icon');
  if (icon) icon.textContent = theme === 'light' ? '☾' : '☀';
}

// ── Shared API helpers ──────────────────────────────────────

async function apiFetch(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// ── Status bar updater (shared across pages) ────────────────

async function refreshStatusBar() {
  try {
    const s = await apiFetch('/api/pipeline/status');
    document.getElementById('sb-config-name').textContent = s.config_name || '—';
    document.getElementById('sb-parser').textContent     = s.parser || '—';
    document.getElementById('sb-embedder').textContent   = s.embedder || '—';
    document.getElementById('sb-store').textContent      = s.vector_store || '—';
    document.getElementById('sb-docs').textContent       = s.documents_ingested ?? 0;
    document.getElementById('sb-chunks').textContent     = s.total_chunks ?? 0;

    const dot   = document.getElementById('status-dot');
    const label = document.getElementById('status-label');
    if (s.pipeline_ready) {
      dot.className   = 'status-dot ok';
      label.textContent = 'ready';
    } else {
      dot.className   = 'status-dot';
      label.textContent = 'idle';
    }
  } catch (_) { /* silently ignore if server not up yet */ }
}

// Refresh status bar on load and every 15s; sync theme icon
document.addEventListener('DOMContentLoaded', () => {
  refreshStatusBar();
  setInterval(refreshStatusBar, 15000);
  _updateThemeIcon(document.documentElement.getAttribute('data-theme') || 'dark');
});

// ── YAML serialisation helper (client-side approximation) ───
// Used for the live preview without a round-trip to the server.

function objectToYaml(obj, indent = 0) {
  const pad = '  '.repeat(indent);
  const lines = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined || v === '') continue;
    if (Array.isArray(v)) {
      lines.push(`${pad}${k}: [${v.join(', ')}]`);
    } else if (typeof v === 'object') {
      lines.push(`${pad}${k}:`);
      lines.push(objectToYaml(v, indent + 1));
    } else {
      const val = typeof v === 'string' && v.includes(':') ? `"${v}"` : v;
      lines.push(`${pad}${k}: ${val}`);
    }
  }
  return lines.join('\n');
}

// ═══════════════════════════════════════════════════════════
//  Configure page Alpine.js data
// ═══════════════════════════════════════════════════════════

function configureApp() {
  return {
    activeModule: 'parser',
    validating: false,
    validationResults: [],

    // Module metadata for the sidebar
    modules: [
      { key: 'parser',       icon: '▸', label: 'Parser',       desc: 'Extract text from documents' },
      { key: 'chunker',      icon: '▸', label: 'Chunker',      desc: 'Split into chunks' },
      { key: 'embedder',     icon: '▸', label: 'Embedder',     desc: 'Generate embeddings' },
      { key: 'vector_store', icon: '▸', label: 'Vector Store', desc: 'Index embeddings' },
      { key: 'retriever',    icon: '▸', label: 'Retriever',    desc: 'Retrieve relevant chunks' },
      { key: 'evaluator',    icon: '▸', label: 'Evaluator',    desc: 'Compute metrics' },
    ],

    // Implementation options for each module
    parserOptions: [
      { value: 'pdfplumber', label: 'pdfplumber', badge: 'local', desc: 'Complex layouts, tables' },
      { value: 'pymupdf',    label: 'PyMuPDF',    badge: 'local', desc: 'Fast, large files' },
      { value: 'azure_di',   label: 'Azure DI',   badge: 'cloud', desc: 'OCR, scanned docs' },
    ],
    chunkerOptions: [
      { value: 'recursive', label: 'Recursive',  badge: 'default', desc: 'Character-based splitting' },
      { value: 'semantic',  label: 'Semantic',   badge: 'advanced', desc: 'Embedding-guided boundaries' },
    ],
    embedderOptions: [
      { value: 'sentence_transformers', label: 'Sentence Transformers', badge: 'local',  desc: 'No API key needed' },
      { value: 'openai',               label: 'OpenAI',                badge: 'cloud',  desc: 'text-embedding-3-*' },
      { value: 'azure_openai',         label: 'Azure OpenAI',          badge: 'cloud',  desc: 'Azure deployment' },
    ],
    vectorStoreOptions: [
      { value: 'chromadb',     label: 'ChromaDB',       badge: 'local',  desc: 'Persistent, no server' },
      { value: 'qdrant_local', label: 'Qdrant local',   badge: 'local',  desc: 'On-disk HNSW index' },
      { value: 'pinecone',     label: 'Pinecone',       badge: 'cloud',  desc: 'Managed vector DB' },
      { value: 'azure_search', label: 'Azure AI Search',badge: 'cloud',  desc: 'Enterprise vector search' },
    ],
    retrieverOptions: [
      { value: 'semantic', label: 'Semantic',  badge: 'default',  desc: 'Pure vector similarity' },
      { value: 'hybrid',   label: 'Hybrid',   badge: 'advanced', desc: 'BM25 + vector fusion' },
    ],

    // Config state — mirrors PipelineConfig shape
    config: {
      name: 'rag-demo',
      description: '',
      parser: {
        implementation: 'pdfplumber',
        azure_endpoint: '', azure_api_key: '', azure_model_id: 'prebuilt-document',
      },
      chunker: {
        implementation: 'recursive',
        chunk_size: 512, chunk_overlap: 64,
        separators: ['\n\n', '\n', '. ', ' '],
        similarity_threshold: 0.75, min_chunk_size: 100, max_chunk_size: 1024,
      },
      embedder: {
        implementation: 'sentence_transformers',
        local_model_name: 'all-MiniLM-L6-v2', device: 'cpu', batch_size: 32,
        api_key: '', model: 'text-embedding-3-small',
        azure_endpoint: '', azure_deployment: '',
      },
      vector_store: {
        implementation: 'chromadb',
        collection_name: 'rag_demo',
        chromadb_path: './.chromadb',
        qdrant_path: './.qdrant',
        pinecone_api_key: '', pinecone_index_name: 'rag-demo',
        azure_search_endpoint: '', azure_search_api_key: '', azure_search_index_name: 'rag-demo',
        embedding_dim: 384,
      },
      retriever: {
        implementation: 'semantic',
        top_k: 5, vector_weight: 0.7, bm25_weight: 0.3,
      },
      evaluator: {
        enable_retrieval_metrics: true,
        enable_parsing_metrics: true,
        enable_chunking_metrics: true,
        enable_answer_metrics: false,
        k_values: [1, 3, 5],
      },
    },

    yamlPreview: '',

    // ── Lifecycle ────────────────────────────────────────────
    async init() {
      // Load current server config on mount
      try {
        const data = await apiFetch('/api/config/current');
        this.config = this._mergeConfig(this.config, data);
      } catch (_) { /* use defaults */ }
      this.updateYamlPreview();
    },

    // ── Sidebar helpers ──────────────────────────────────────
    activeModuleLabel() {
      return this.modules.find(m => m.key === this.activeModule)?.label ?? '';
    },
    activeModuleDesc() {
      return this.modules.find(m => m.key === this.activeModule)?.desc ?? '';
    },
    implLabel(moduleKey) {
      const map = {
        parser:       this.config.parser.implementation,
        chunker:      this.config.chunker.implementation,
        embedder:     this.config.embedder.implementation,
        vector_store: this.config.vector_store.implementation,
        retriever:    this.config.retriever.implementation,
        evaluator:    'configured',
      };
      return map[moduleKey] ?? '';
    },

    // ── YAML preview ─────────────────────────────────────────
    updateYamlPreview() {
      // Filter out empty credential fields for cleaner preview
      const clean = JSON.parse(JSON.stringify(this.config));
      const creds = ['azure_api_key', 'api_key', 'pinecone_api_key',
                     'azure_search_api_key', 'azure_openai_api_key'];
      const scrub = (obj) => {
        for (const k of Object.keys(obj)) {
          if (creds.includes(k) && obj[k]) obj[k] = '***';
          else if (typeof obj[k] === 'object' && obj[k]) scrub(obj[k]);
        }
      };
      scrub(clean);
      this.yamlPreview = objectToYaml(clean);
    },
    debouncedPreview: debounce(function() { this.updateYamlPreview(); }, 250),

    // ── Actions ──────────────────────────────────────────────
    async saveConfig() {
      try {
        await apiFetch('/api/config/save', {
          method: 'POST',
          body: JSON.stringify({ config: this.config }),
        });
        refreshStatusBar();
        this._flash('Saved.', 'ok');
      } catch (e) {
        this._flash(`Save failed: ${e.message}`, 'error');
      }
    },

    async saveAndValidate() {
      this.validating = true;
      this.validationResults = [];
      const dot = document.getElementById('status-dot');
      const label = document.getElementById('status-label');
      if (dot) { dot.className = 'status-dot active'; label.textContent = 'validating…'; }

      try {
        const res = await apiFetch('/api/config/validate', {
          method: 'POST',
          body: JSON.stringify({ config: this.config }),
        });
        this.validationResults = res.modules || [];
        refreshStatusBar();
      } catch (e) {
        this.validationResults = [{ module: 'pipeline', ok: false, message: e.message }];
      } finally {
        this.validating = false;
      }
    },

    exportYAML() {
      window.location.href = '/api/config/export';
    },

    async importYAML(event) {
      const file = event.target.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append('file', file);
      try {
        const res = await fetch('/api/config/import', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        this.config = this._mergeConfig(this.config, data.config);
        this.updateYamlPreview();
        refreshStatusBar();
      } catch (e) {
        alert(`Import failed: ${e.message}`);
      }
    },

    // ── Helpers ──────────────────────────────────────────────
    _mergeConfig(base, incoming) {
      // Deep merge incoming over base defaults
      const out = JSON.parse(JSON.stringify(base));
      const merge = (a, b) => {
        for (const k of Object.keys(b ?? {})) {
          if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k]) {
            merge(a[k], b[k]);
          } else if (b[k] !== null && b[k] !== undefined) {
            a[k] = b[k];
          }
        }
      };
      merge(out, incoming);
      return out;
    },
    _flash(msg, type) {
      // Small status flash in the navbar
      const label = document.getElementById('status-label');
      const dot   = document.getElementById('status-dot');
      if (label) label.textContent = msg;
      if (dot)   dot.className = `status-dot ${type === 'ok' ? 'ok' : 'error'}`;
      setTimeout(() => refreshStatusBar(), 2000);
    },
  };
}

// ═══════════════════════════════════════════════════════════
//  Evaluate page Alpine.js data
// ═══════════════════════════════════════════════════════════

function evaluateApp() {
  return {
    tab: 'ingest',

    // ── Ingest ───────────────────────────────────────────────
    files: [],
    dragging: false,
    ingesting: false,
    ingestionResults: null,

    // ── Query ────────────────────────────────────────────────
    queryText: '',
    topK: 5,
    querying: false,
    queryResults: [],
    queryError: null,

    // ── Compare ──────────────────────────────────────────────
    configB: null,
    comparing: false,
    comparisonResults: null,
    evalSetJson: JSON.stringify([
      { query: 'What is the main topic?', expected: [''] },
      { query: 'What are the key findings?', expected: [''] },
    ], null, 2),

    // Shared
    activeConfigName: 'rag-demo',

    // ── Lifecycle ────────────────────────────────────────────
    async init() {
      try {
        const s = await apiFetch('/api/pipeline/status');
        this.activeConfigName = s.config_name || 'rag-demo';
        this.topK = 5;
      } catch (_) {}
    },

    // ── File handling ────────────────────────────────────────
    handleDrop(event) {
      this.dragging = false;
      const dropped = Array.from(event.dataTransfer.files).filter(f => f.type === 'application/pdf');
      this.files.push(...dropped);
    },
    handleFileSelect(event) {
      this.files.push(...Array.from(event.target.files));
      event.target.value = '';
    },
    formatBytes(bytes) {
      if (bytes < 1024) return `${bytes}B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
      return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
    },

    // ── Ingestion ────────────────────────────────────────────
    async runIngestion() {
      if (!this.files.length) return;
      this.ingesting = true;
      this.ingestionResults = null;

      const fd = new FormData();
      this.files.forEach(f => fd.append('files', f));

      try {
        const res = await fetch('/api/pipeline/ingest', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Ingestion failed');
        this.ingestionResults = data;
        this.files = [];
        refreshStatusBar();
      } catch (e) {
        this.ingestionResults = { summaries: [], errors: [{ filename: '—', error: e.message }] };
      } finally {
        this.ingesting = false;
      }
    },

    async resetStore() {
      if (!confirm('Reset the vector store? All ingested data will be lost.')) return;
      try {
        await apiFetch('/api/pipeline/reset', { method: 'DELETE' });
        this.ingestionResults = null;
        this.queryResults = [];
        refreshStatusBar();
      } catch (e) {
        alert(`Reset failed: ${e.message}`);
      }
    },

    totalChunks() {
      return (this.ingestionResults?.summaries ?? []).reduce((s, r) => s + (r.chunk_count || 0), 0);
    },
    avgParseTime() {
      const times = (this.ingestionResults?.summaries ?? [])
        .map(r => r.parse_time_s).filter(Boolean);
      if (!times.length) return '—';
      return (times.reduce((a, b) => a + b, 0) / times.length).toFixed(2);
    },

    // ── Query ────────────────────────────────────────────────
    async runQuery() {
      if (!this.queryText.trim()) return;
      this.querying = true;
      this.queryError = null;
      this.queryResults = [];

      try {
        const data = await apiFetch('/api/pipeline/query', {
          method: 'POST',
          body: JSON.stringify({ query: this.queryText, top_k: this.topK }),
        });
        this.queryResults = data.results || [];
      } catch (e) {
        this.queryError = e.message;
      } finally {
        this.querying = false;
      }
    },

    // ── Compare ──────────────────────────────────────────────
    async loadConfigB(event) {
      const file = event.target.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append('file', file);
      try {
        const res = await fetch('/api/config/import', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        // Restore active config (import endpoint replaces it — we just want config_b data)
        this.configB = data.config;
        // Restore active config
        const activeRes = await apiFetch('/api/config/current');
        this.activeConfigName = activeRes.name || 'rag-demo';
      } catch (e) {
        alert(`Failed to load Config B: ${e.message}`);
      }
    },

    async runComparison() {
      if (!this.configB) return;
      let evalSet;
      try {
        evalSet = JSON.parse(this.evalSetJson);
      } catch (e) {
        alert('Invalid JSON in evaluation dataset.');
        return;
      }

      this.comparing = true;
      this.comparisonResults = null;
      try {
        const data = await apiFetch('/api/evaluate/compare', {
          method: 'POST',
          body: JSON.stringify({ eval_set: evalSet, config_b: this.configB }),
        });
        this.comparisonResults = data;
      } catch (e) {
        alert(`Comparison failed: ${e.message}`);
      } finally {
        this.comparing = false;
      }
    },

    barPct(value, row) {
      const max = Math.max(row.a, row.b, 0.001);
      return Math.round((value / max) * 100);
    },
  };
}
