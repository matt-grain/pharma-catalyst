# Knowledge Base RAG — Technical Reference

## Overview

The Knowledge Base RAG provides the hypothesis agent with access to internal pharma documents
(reports, assay data, SOPs, safety reviews) via hybrid retrieval. It sits alongside the
literature RAG (arxiv papers) as a complementary data source.

Two tools work together in a **search → retrieve** pattern:
- `query_knowledge_base` — find relevant chunks across all documents
- `read_kb_source` — read the full source file when the agent needs complete data

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Hypothesis Agent                                                       │
│  "What assay data do we have for BBB permeability?"                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                          Step 1 │  query_knowledge_base("PAMPA assay BBB permeability")
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  INDEX LOOKUP (cached after first build)                                │
│                                                                         │
│  KnowledgeQueryTool._get_index()                                        │
│    1. ClassVar _index cache → hit? return immediately                   │
│    2. Load index.json from disk → found? cache & return                 │
│    3. Neither? Auto-build from source files → write index → cache       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  HYBRID SEARCH                                                          │
│                                                                         │
│  ┌─── PREPARE ───────────────────────────────────────────────────────┐  │
│  │  query_embedding = fastembed("PAMPA assay BBB permeability")      │  │
│  │  query_tokens = ["pampa", "assay", "bbb", "permeability"]         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── DENSE SEARCH (semantic) ───────────────────────────────────────┐  │
│  │  cosine_sim(query_emb, chunk_emb) for all 37 chunks               │  │
│  │  → top 20 by similarity                                           │  │
│  │                                                                   │  │
│  │  Catches semantic matches:                                        │  │
│  │    "MDCK-MDR1 permeability" matches even without "PAMPA" keyword  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── SPARSE SEARCH (BM25 lexical) ──────────────────────────────────┐  │
│  │  BM25(query_tokens, chunk) for all 37 chunks                      │  │
│  │  → top 20 by BM25 score                                           │  │
│  │                                                                   │  │
│  │  Catches exact term matches:                                      │  │
│  │    "pampa_pe: 8.2" found by exact token "pampa"                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── RRF MERGE ─────────────────────────────────────────────────────┐  │
│  │  rrf_score(chunk) = 1/(60 + rank_dense) + 1/(60 + rank_sparse)    │  │
│  │                                                                   │  │
│  │  Chunks ranked high in BOTH lists get the best combined score.    │  │
│  │  No score normalization needed — purely rank-based.               │  │
│  │                                                                   │  │
│  │  → top 5 returned to agent                                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SEARCH RESULTS (returned to agent)                                     │
│                                                                         │
│  Top 5 results for 'PAMPA assay BBB permeability':                      │
│                                                                         │
│  --- [Source: assay_data/bbb_pampa_assay_results.csv § rows_1-10] ---   │
│  Row 1: compound_id: SNF-001, smiles: CN1C=NC2=..., pampa_pe: 8.2,      │
│  classification: CNS+, mw: 194.19, logp: 0.16, ...                      │
│  Row 2: compound_id: SNF-002, ...                                       │
│  (CSV chunks shown in full — all 10 rows visible)                       │
│                                                                         │
│  --- [Source: internal_reports/bbb_model_benchmark_2024.md              │
│       § Dataset Characteristics] ---                                    │
│  Our internal dataset comprises 2,847 compounds tested in the           │
│  MDCK-MDR1 permeability assay, supplemented with PAMPA-BBB...           │
│  (markdown truncated at 500 chars — enough for context)                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │  Agent sees CSV data but wants the full 50-row file
                                 │
                          Step 2 │  read_kb_source("assay_data/bbb_pampa_assay_results.csv")
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  SOURCE RETRIEVAL                                                       │
│                                                                         │
│  ReadKnowledgeSourceTool._run("assay_data/bbb_pampa_assay_results.csv") │
│    1. Resolve path: kb_dir / source_file                                │
│    2. Security: verify path is within kb_dir (no traversal)             │
│    3. Read full file (up to 5000 chars)                                 │
│    4. Return complete content                                           │
│                                                                         │
│  [Source: assay_data/bbb_pampa_assay_results.csv]                       │
│                                                                         │
│  compound_id,smiles,pampa_pe,classification,mw,logp,tpsa,hbd,hba,notes  │
│  SNF-001,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,8.2,CNS+,194.19,0.16,...          │
│  SNF-002,CC(C)CC1=CC=C(C=C1)C(C)C(=O)O,12.5,CNS+,206.28,3.97,...        │
│  ... (all 50 rows)                                                      │
│  SNF-050,CC(C)NCC(O)C1=CC=C(NS(C)(=O)=O)C=C1,0.8,CNS-,272.34,...        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
                    Agent now has full assay data to
                    inform its hypothesis (property ranges,
                    compound distributions, class balance)
```

## Indexing Pipeline (One-Time)

Runs automatically on first query, or manually via CLI:
```bash
uv run python -m pharma_agents.ingest_kb -e bbbp
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│  _build_index(kb_dir)                                                   │
│                                                                         │
│  1. SCAN: glob **/*.md and **/*.csv under knowledge_base/               │
│                                                                         │
│  2. CHUNK each file:                                                    │
│     ┌──────────────────────────────────────────────────────────────┐    │
│     │ Markdown (.md):                                              │    │
│     │   - Extract doc title from YAML frontmatter or # heading     │    │
│     │   - Split on ## headings (natural section boundaries)        │    │
│     │   - If section > 500 words → split on paragraphs             │    │
│     │   - If still too large → split at word boundaries            │    │
│     │   - Apply 50-word overlap between adjacent chunks            │    │
│     ├──────────────────────────────────────────────────────────────┤    │
│     │ CSV (.csv):                                                  │    │
│     │   - Read with csv.DictReader                                 │    │
│     │   - Group rows in batches of 10                              │    │
│     │   - Convert each row to natural language:                    │    │
│     │     "Row 1: compound_id: SNF-001, smiles: ..., mw: 194"      │    │
│     └──────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  3. CONTEXTUAL RETRIEVAL (Anthropic cookbook approach):                 │
│     Before embedding, prepend document context to each chunk:           │
│                                                                         │
│     Raw content:  "TPSA < 90, LogP 1-3, HBD ≤ 3..."                     │
│     Embed text:   "[From: CNS Drug Design Guidelines]                   │
│                    BBB-Specific Property Ranges:                        │
│                    TPSA < 90, LogP 1-3, HBD ≤ 3..."                     │
│                                                                         │
│     → embedding captures BOTH local detail AND global doc context       │
│     → stored content stays clean (no prefix in search results)          │
│                                                                         │
│  4. EMBED: batch fastembed (BAAI/bge-small-en-v1.5, 384 dims)           │
│     model.embed([embed_text_1, embed_text_2, ...])                      │
│                                                                         │
│  5. BM25 VOCABULARY: for each chunk                                     │
│     - _tokenize(content) → lowercase, strip punctuation, len > 1        │
│     - token_freqs = Counter(tokens)                                     │
│     - Global: bm25_doc_count, bm25_avg_dl, bm25_df per token            │
│                                                                         │
│  6. WRITE: index.json (full replacement, not incremental)               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Index Schema

```json
{
  "chunks": {
    "bbb_model_benchmark_2024::chunk_0": {
      "doc_id": "bbb_model_benchmark_2024",
      "source_file": "internal_reports/bbb_model_benchmark_2024.md",
      "section": "Executive Summary",
      "content": "This report presents a systematic benchmarking...",
      "line_start": 12,
      "line_end": 25,
      "embedding": [0.032, -0.018, ...],   // 384 floats
      "token_freqs": {"report": 2, "benchmarking": 1, ...}
    },
    "bbb_pampa_assay_results::chunk_3": {
      "doc_id": "bbb_pampa_assay_results",
      "source_file": "assay_data/bbb_pampa_assay_results.csv",
      "section": "rows_31-40",
      "content": "Row 31: compound_id: SNF-031, ...",
      ...
    }
  },
  "bm25_doc_count": 37,
  "bm25_avg_dl": 105.7,
  "bm25_df": {"model": 15, "bbb": 12, "prediction": 10, ...},
  "created_at": "2026-03-18T...",
  "updated_at": "2026-03-18T..."
}
```

## Why Hybrid: BM25 + Dense + RRF

| Technique | What it solves | Example |
|-----------|---------------|---------|
| **BM25** (sparse) | Exact term matching | "XGBoost" finds chunks with that exact word |
| **Dense** (semantic) | Meaning similarity | "tree ensemble" finds "gradient boosting" chunks |
| **RRF** (fusion) | Merging incompatible score scales | Rank-based: `1/(60+rank)` per list, no normalization |
| **Contextual retrieval** | Chunks losing parent doc context | Embed with `[From: doc title]` prefix |

### RRF Merge Example

```
Query: "ensemble model performance"

Dense ranking:                    Sparse (BM25) ranking:
  #1  benchmark § Model Comp.      #1  benchmark § Key Findings
  #2  benchmark § Key Findings     #2  benchmark § Recommendations
  #3  benchmark § Recommendations  #3  history_csv § rows_11-20

RRF scores (k=60):
  benchmark § Key Findings:    1/(60+2) + 1/(60+1) = 0.0325  ← BEST
  benchmark § Model Comp.:     1/(60+1) + 0         = 0.0164
  benchmark § Recommendations: 1/(60+3) + 1/(60+2) = 0.0321
  history_csv § rows_11-20:    0         + 1/(60+3) = 0.0159
```

A chunk ranked high in **both** lists gets the best combined score.

## Knowledge Base Structure

```
experiments/<experiment>/knowledge_base/
├── index.json                      # Auto-generated, do not edit
├── .gitignore                      # Excludes *.lock files
├── internal_reports/               # Benchmarks, guidelines
│   ├── bbb_model_benchmark_2024.md
│   └── cns_drug_design_guidelines_v3.md
├── assay_data/                     # Experimental measurements
│   ├── bbb_pampa_assay_results.csv
│   └── historical_model_performance.csv
├── safety_docs/                    # Risk assessments, regulatory
│   └── bbb_penetration_safety_review.md
└── sops/                           # Standard operating procedures
    └── ml_model_validation_sop.md
```

Each experiment has its own knowledge base with domain-specific content.
Add new documents → run `uv run python -m pharma_agents.ingest_kb -e <exp>` to rebuild.

## Tools Reference

### `query_knowledge_base`
**Used by:** Hypothesis Agent
**Input:** Search query string
**Output:** Top 5 chunks with source attribution. CSV chunks shown in full, markdown truncated at 500 chars.

### `read_kb_source`
**Used by:** Hypothesis Agent
**Input:** `source_file` path from search results (e.g., `"assay_data/bbb_pampa_assay_results.csv"`)
**Output:** Full file content (up to 5000 chars). Path traversal guard prevents reading outside kb_dir.

### Typical Agent Workflow
```
1. query_knowledge_base("historical model performance toxicity")
   → sees chunk from history CSV with partial rows
   → sees chunk from benchmark report with key findings

2. read_kb_source("assay_data/historical_model_performance.csv")
   → gets all 20 rows of model iteration history
   → uses this to avoid repeating failed approaches

3. query_knowledge_base("validation requirements SOP")
   → gets SOP requirements: ROC-AUC > 0.85, stratified 5-fold
   → incorporates these constraints into proposal
```
