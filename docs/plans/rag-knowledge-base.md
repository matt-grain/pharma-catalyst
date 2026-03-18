# RAG Knowledge Base Implementation Plan

## Context

Matt is preparing for a Sanofi AI/ML Lead interview where RAG is a key competency. The pharma-agents project already has a lightweight literature RAG (cosine similarity over paper embeddings). We're adding a **Knowledge Base RAG** that demonstrates hybrid retrieval (BM25 + dense + RRF) over diverse internal documents — exactly what a pharma company like Sanofi would need for internal data (assay results, SOPs, safety docs, model benchmarks).

This turns textbook RAG knowledge into lived, demonstrable experience.

## What We're Building

A `KnowledgeBase` system that:
1. **Ingests** markdown docs and CSV files, chunked with overlap
2. **Indexes** with both sparse (BM25) and dense (fastembed) representations
3. **Retrieves** via hybrid search with Reciprocal Rank Fusion (RRF)
4. **Attributes** every result to source file + section
5. Integrates as CrewAI tools for the hypothesis agent

## Architecture

```
experiments/<exp>/knowledge_base/
├── index.json              # Chunk index (embeddings + BM25 vocab)
├── internal_reports/*.md   # Source documents
├── assay_data/*.csv
├── safety_docs/*.md
└── sops/*.md
```

New file: `src/pharma_agents/tools/knowledge_base.py`

```
┌─────────────────────────────────────────────────────────┐
│                  KNOWLEDGE BASE RAG                      │
│                                                          │
│  Ingest:  md/csv → chunk (500 tok, 50 overlap)          │
│           → embed (fastembed BGE-small) + BM25 vocab     │
│           → store in index.json                          │
│                                                          │
│  Query:   query → embed + tokenize                       │
│           → dense top-20 (cosine) + sparse top-20 (BM25) │
│           → RRF merge (k=60)                             │
│           → return top-5 with source attribution         │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Chunking & Indexing Engine
**Assignee:** Sonnet agent
**File:** `src/pharma_agents/tools/knowledge_base.py` (create)

### Functions to implement:

```python
# --- Constants ---
CHUNK_SIZE = 500        # tokens (approx words)
CHUNK_OVERLAP = 50      # token overlap between chunks
RRF_K = 60              # RRF constant
BM25_K1 = 1.5           # BM25 term frequency saturation
BM25_B = 0.75           # BM25 length normalization

# --- Data structures ---
@dataclass
class Chunk:
    chunk_id: str           # f"{doc_id}::chunk_{i}"
    doc_id: str             # source filename (stem)
    source_file: str        # relative path from knowledge_base/
    section: str            # heading or "row_N" for CSV
    content: str            # chunk text
    line_start: int         # line range in source
    line_end: int
    embedding: list[float]  # 384 dims

# --- Chunking ---
def chunk_markdown(file_path: Path) -> list[dict]:
    """Split markdown into ~500-token chunks at section boundaries.

    - Split on ## headings first (natural boundaries)
    - If a section > CHUNK_SIZE, split on paragraphs (double newline)
    - If still too large, split on sentences
    - Each chunk carries: source_file, section (heading text), line_start, line_end
    - Adjacent chunks overlap by CHUNK_OVERLAP tokens
    """

def chunk_csv(file_path: Path) -> list[dict]:
    """Convert CSV rows to natural language chunks.

    - Read CSV with csv.DictReader
    - Group rows into chunks of ~10 rows each
    - Convert each row to prose: "Compound {id} ({smiles}): MW={mw}, LogP={logp}..."
    - Section = f"rows_{start}-{end}"
    - Each chunk carries source_file, section, line_start, line_end
    """

# --- Index management ---
def get_kb_dir() -> Path:
    """Return experiments/<exp>/knowledge_base/ (uses MAIN root, not worktree)."""

def _build_index(kb_dir: Path) -> dict:
    """Scan all files in kb_dir, chunk them, embed, build BM25 vocab.

    Returns index dict:
    {
        "chunks": {
            chunk_id: {
                "doc_id": str,
                "source_file": str,
                "section": str,
                "content": str,
                "line_start": int,
                "line_end": int,
                "embedding": [float],   # 384 dims
                "token_freqs": {str: int}  # for BM25
            }
        },
        "bm25_doc_count": int,
        "bm25_avg_dl": float,          # avg document length
        "bm25_df": {str: int},         # document frequency per term
        "created_at": str,
        "updated_at": str
    }
    """

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25.
    Strip punctuation, lowercase, split on whitespace."""
```

### Key decisions:
- Reuse `_locked_index` pattern from literature.py (import it or duplicate)
- Reuse fastembed model cache pattern (`_model: ClassVar`)
- Index lives at `experiments/<exp>/knowledge_base/index.json`
- Scan `**/*.md` and `**/*.csv` under knowledge_base/ (skip index.json)
- No new dependencies — csv module from stdlib for CSV

### Test: `tests/test_knowledge_base.py`
- `test_chunk_markdown_splits_on_headings`
- `test_chunk_csv_converts_rows_to_prose`
- `test_build_index_creates_embeddings_and_bm25_vocab`

---

## Phase 2: BM25 + Hybrid Retrieval with RRF
**Assignee:** Sonnet agent
**File:** `src/pharma_agents/tools/knowledge_base.py` (extend)

### Functions to implement:

```python
def _bm25_score(query_tokens: list[str], chunk: dict, index: dict) -> float:
    """Okapi BM25 scoring for a single chunk.

    score = Σ IDF(qi) * (tf * (k1+1)) / (tf + k1 * (1 - b + b * dl/avgdl))

    where:
    - IDF(qi) = log((N - df + 0.5) / (df + 0.5) + 1)
    - tf = chunk["token_freqs"].get(token, 0)
    - dl = sum(chunk["token_freqs"].values())
    - avgdl = index["bm25_avg_dl"]
    - N = index["bm25_doc_count"]
    - df = index["bm25_df"].get(token, 0)
    """

def _dense_search(query_embedding: list[float], index: dict, top_k: int = 20) -> list[tuple[str, float]]:
    """Cosine similarity search. Returns [(chunk_id, score), ...] sorted desc."""

def _sparse_search(query_tokens: list[str], index: dict, top_k: int = 20) -> list[tuple[str, float]]:
    """BM25 search. Returns [(chunk_id, score), ...] sorted desc."""

def _rrf_merge(rankings: list[list[tuple[str, float]]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    For each chunk across all rankings:
        rrf_score = Σ 1 / (k + rank_in_list_i)

    Returns merged ranking sorted by rrf_score desc.
    """

def hybrid_search(query: str, index: dict, top_k: int = 5) -> list[dict]:
    """Full hybrid retrieval pipeline.

    1. Embed query (fastembed)
    2. Tokenize query
    3. Dense search → top 20
    4. Sparse search → top 20
    5. RRF merge → top_k results
    6. Return list of {chunk_id, source_file, section, content, rrf_score}
    """
```

### Test: `tests/test_knowledge_base.py` (extend)
- `test_bm25_score_ranks_relevant_chunk_higher`
- `test_rrf_merge_combines_two_rankings`
- `test_hybrid_search_returns_attributed_results`

---

## Phase 3: CrewAI Tools
**Assignee:** Sonnet agent
**Files:**
- `src/pharma_agents/tools/knowledge_base.py` (add tool classes)
- `src/pharma_agents/tools/__init__.py` (add exports)

### Tools to implement:

```python
class KnowledgeQueryTool(BaseTool):
    """Search the internal knowledge base for relevant information."""
    name: str = "query_knowledge_base"
    description: str = (
        "Search the internal knowledge base (reports, assay data, SOPs, safety docs) "
        "for information relevant to your hypothesis. Returns top results with source attribution. "
        "Use specific queries like 'BBB physicochemical property ranges' or "
        "'historical model performance ensemble methods'."
    )
    top_k: int = 5
    _model: ClassVar = None
    _index: ClassVar = None  # cached index

    def _run(self, query: str) -> str:
        """
        1. Load/cache index from get_kb_dir() / "index.json"
        2. If no index exists, auto-build from files in kb_dir
        3. Run hybrid_search(query, index, top_k)
        4. Format results:
           --- [Source: internal_reports/bbb_model_benchmark_2024.md § Model Comparison] ---
           Content here (first 300 chars)...
           Score: 0.87
        """

class KnowledgeIngestTool(BaseTool):
    """Rebuild the knowledge base index from source documents."""
    name: str = "rebuild_knowledge_base"
    description: str = (
        "Rebuild the knowledge base index from all documents in the knowledge_base directory. "
        "Use this after adding new documents."
    )

    def _run(self, _input: str = "") -> str:
        """
        1. Call _build_index(get_kb_dir())
        2. Save to index.json (with locking)
        3. Return summary: "Indexed N chunks from M documents"
        """
```

### Update `__init__.py`:
```python
from .knowledge_base import KnowledgeIngestTool, KnowledgeQueryTool, get_kb_dir
```

---

## Phase 4: Wire Into Crew + Update Prompts
**Assignee:** Sonnet agent
**Files:**
- `src/pharma_agents/crew.py` — add KnowledgeQueryTool to hypothesis_agent tools
- `src/pharma_agents/tasks.yaml` — update hypothesis_task to mention knowledge base
- `src/pharma_agents/agents.yaml` — update hypothesis_agent backstory

### crew.py changes:
```python
# Add import
from pharma_agents.tools import KnowledgeQueryTool

# In hypothesis_agent():
tools=[
    ReadTrainPyTool(),
    LiteratureQueryTool(),
    KnowledgeQueryTool(),       # ← NEW
    SkillDiscoveryTool(),
    ...
]
```

### tasks.yaml changes (hypothesis_task):
Add after the literature query section:
```yaml
ALSO: Query the internal knowledge base for relevant information:
- Use query_knowledge_base with queries like "{property} physicochemical ranges"
  or "historical model performance" or "validation requirements"
- The knowledge base contains internal reports, assay data, SOPs, and safety docs
- Cross-reference literature findings with internal data
```

### agents.yaml changes (hypothesis_agent backstory):
Add mention of internal knowledge base access.

---

## Phase 5: Documentation Updates
**Assignee:** Me (Claude Opus) — not Sonnet
**Files:**
- `docs/ARCHITECTURE.md` — add Knowledge Base RAG section, update diagrams
- `docs/TOOLS.md` — add query_knowledge_base and rebuild_knowledge_base entries
- `README.md` — mention RAG knowledge base in features/architecture
- `docs/decisions.md` — ADR for hybrid retrieval choice (BM25 + dense + RRF)
- Save this plan to `docs/plans/rag-knowledge-base.md`

---

## Verification

1. **Unit tests pass:** `uv run pytest tests/test_knowledge_base.py -v`
2. **Index builds:** Run KnowledgeIngestTool against bbbp/knowledge_base/ → index.json created
3. **Hybrid search works:** Query "BBB physicochemical property ranges" → returns chunks from cns_drug_design_guidelines
4. **RRF improves over cosine-only:** Same query returns better results than pure dense search
5. **Integration:** Run a single iteration with `/run bbbp 1` — hypothesis agent uses query_knowledge_base
6. **Ruff clean:** `uv run ruff check src/pharma_agents/tools/knowledge_base.py`

## Dependencies

- No new packages needed
- Reuses: fastembed, numpy, csv (stdlib), math (stdlib), re (stdlib)
- Follows existing patterns from literature.py
