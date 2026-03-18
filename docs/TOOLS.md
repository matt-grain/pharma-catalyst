# Tool Reference

All custom tools available to pharma-agents agents.

---

## Core Tools (ML Pipeline)

### `read_train_py`
**Used by:** Hypothesis Agent, Model Agent, Evaluator Agent
**Source:** `src/pharma_agents/tools/training.py`

Reads `train.py` content in three modes:

| Mode | Input | Output |
|------|-------|--------|
| **Full** (default) | `"read"` or `"full"` | Full file with line numbers (`   1 | code...`) |
| **Outline** | `"outline"` | Function/class signatures with line numbers — use first on large files |
| **Line range** | `"lines 20-50"` | Only the specified lines with line numbers |

**Workflow tip:** Use `outline` first to find where to edit, then `lines N-M` to read just that section. This minimizes context pollution in agent conversation history.

```
Input: "read" | "outline" | "lines N-M"
Output: File content with line numbers, or structural outline
```

---

### `edit_train_py`
**Used by:** Model Agent
**Source:** `src/pharma_agents/tools/training.py`

Makes targeted edits to `train.py` by replacing a specific text snippet. Preferred over `write_train_py` for small changes — avoids f-string double-encoding issues and reduces context pollution.

```
Input: old_text (exact text to find), new_text (replacement)
Output: "Successfully edited train.py: replaced N lines with M lines" or error
```

**Safety features:**
- Rejects if `old_text` not found — shows similar lines as hints (extracts key identifier for fuzzy matching)
- Rejects ambiguous matches (>1 occurrence) — asks for more context
- Blocks dangerous patterns in `new_text`: `os.system()`, `subprocess.run()`, `eval()`, `exec()`, etc.

---

### `search_train_py`
**Used by:** Model Agent
**Source:** `src/pharma_agents/tools/training.py`

Searches `train.py` for a text pattern (supports regex). Returns matching lines with line numbers. Use before `edit_train_py` to find exact text to replace.

```
Input: Pattern string (regex or literal)
Output: Matching lines with line numbers, e.g. "  L  26: score = 0.85"
```

**Features:**
- Case-insensitive by default
- Falls back to literal search if regex is invalid
- Shows match count

---

### `write_train_py`
**Used by:** Model Agent
**Source:** `src/pharma_agents/tools/training.py`

Writes complete content to `train.py`, overwriting existing code. Use `edit_train_py` for small changes instead.

**Validation (applied before writing):**
- Strips markdown code fences (```` ```python ... ``` ````) — common LLM output artifact
- Fixes double-encoded escape sequences (`\\n` → `\n`, `\\t` → `\t`) — common when code passes through JSON serialization twice
- Rejects content shorter than 50 characters
- Requires a `def train` function
- Requires at least one `import` statement
- Blocks dangerous patterns: `os.system()`, `subprocess.run()`, `eval()`, `exec()`, `shutil.rmtree()`, etc.
- Warns if feature computation detected without inf/NaN handling

```
Input: Full Python code (string)
Output: "Successfully wrote X characters to train.py" or validation error
```

---

### `code_check`
**Used by:** Model Agent
**Source:** `src/pharma_agents/tools/training.py`

Runs ruff linter on `train.py` to check for syntax/style errors.

```
Input: (none)
Output: "OK - No linting errors" or list of errors to fix
```

**Caching:** Disabled via `cache_function = lambda: False`. This ensures the agent always sees fresh linting results after writing new code.

---

### `run_train_py`
**Used by:** Evaluator Agent
**Source:** `src/pharma_agents/tools/training.py`

Executes `train.py` and returns the validation metric.

```
Input: (none)
Output: "RMSE: 0.6532" or "ROC_AUC: 0.9286" (depending on experiment)
Timeout: 180 seconds
```

**Caching:** Disabled via `cache_function = lambda: False` — prevents CrewAI's anti-loop protection from blocking retries when the agent calls `run_train_py("run")` multiple times with the same input.

---

### `install_package`
**Used by:** Model Agent
**Source:** `src/pharma_agents/tools/training.py`

Installs a Python package via `uv add`. Only whitelisted ML/data science packages are allowed.

```
Input: Package name (e.g., "lightgbm", "xgboost")
Output: "Successfully installed 'lightgbm'. You can now import it." or error
Limit: 3 installs per iteration
Timeout: 120 seconds
```

**Allowed packages:** `lightgbm`, `xgboost`, `catboost`, `scikit-learn`, `sklearn`, `pandas`, `numpy`, `scipy`, `rdkit`, `deepchem`, `torch`, `pytorch`, `tensorflow`, `keras`, `molfeat`, `descriptastorus`, `mordred`, `pubchempy`, `chembl-webresource-client`

**Caching:** Disabled — tracks installed packages per iteration via ClassVar counter (reset between iterations).

---

## Literature Tools (Archivist)

### `search_arxiv`
**Used by:** Archivist Agent
**Source:** `src/pharma_agents/tools/arxiv.py`

Searches arxiv.org for recent papers on a topic.

```
Input: Search query (e.g., "ADMET prediction graph neural network")
Output: List of paper IDs with titles and abstracts (max 10)
Limit: 8 searches per iteration
Rate limit: 3 seconds between requests (arxiv recommendation)
```

**Caching:** Disabled — always performs fresh searches.

Example output:
```
Found 10 papers for 'ADMET prediction':

- **2401.12345** (2024-01-15): Graph Neural Networks for ADMET...
  We present a novel approach using message-passing...
```

---

### `fetch_arxiv_paper`
**Used by:** Archivist Agent
**Source:** `src/pharma_agents/tools/arxiv.py`

Fetches paper content — tries alphaxiv overview first, then alphaxiv full text, then arxiv abstract page via markdown.new.

```
Input: Paper ID (e.g., "2401.12345" or full arxiv URL)
Output: Structured paper content (markdown)
Limit: 10 papers per iteration
Rate limit: 1 second between requests
Retries: 2 attempts per URL
```

**Caching:** Disabled — always fetches fresh content.

**Why alphaxiv?** Faster than PDF, structured markdown optimized for LLM consumption.

---

### `store_paper`
**Used by:** Archivist Agent
**Source:** `src/pharma_agents/tools/literature.py`

Stores a paper summary with embeddings for semantic search. Accepts JSON or raw markdown (auto-extracts paper ID and summary).

```
Input: JSON with {paper_id, title, summary, key_methods[]} OR raw markdown content
Output: "Stored paper X with embedding (384 dims)"
```

Creates:
- `literature/papers/<paper_id>.md` — Markdown summary
- `literature/papers/<paper_id>_full.md` — Full content (if available)
- `literature/index.json` — Index with embeddings

**Embedding model:** fastembed `BAAI/bge-small-en-v1.5` (384 dims)

---

### `query_literature`
**Used by:** Hypothesis Agent
**Source:** `src/pharma_agents/tools/literature.py`

Semantic search over the local literature database using cosine similarity.

```
Input: Search query (e.g., "gradient boosting molecular features")
Output: Top 5 matching papers with similarity scores and summaries
```

**Note:** Caches the embedding model across calls for performance (ClassVar `_model`).

Example output:
```
Top 5 papers for 'gradient boosting molecular features':

**[0.87] 2401.12345**: XGBoost for Molecular Property Prediction
  Gradient boosting with Morgan fingerprints achieves...

**[0.82] 2312.54321**: Feature Engineering for ADMET
  Combining physicochemical descriptors with...
```

---

### `fetch_more_papers`
**Used by:** Hypothesis Agent
**Source:** `src/pharma_agents/tools/literature.py`

Searches arxiv and fetches fresh papers on-demand when the hypothesis agent needs more ideas.

```
Input: Specific topic (e.g., "attention mechanisms for molecules")
Output: Summary of papers found and stored
Limit: 2 calls per iteration
```

**Caching:** Disabled — always fetches fresh content.

Workflow:
1. Searches arxiv for papers matching query
2. Fetches top 3 papers via alphaxiv
3. Stores them in the literature database with embeddings
4. Returns summary of what was stored

**Use case:** When `query_literature` doesn't have enough relevant papers or the agent is stuck and needs fresh ideas.

---

## Scientific Data Tools

### `search_pubmed`
**Used by:** Archivist Agent
**Source:** `src/pharma_agents/tools/tooluniverse.py`

Searches PubMed for peer-reviewed biomedical papers via NCBI E-utilities (esearch + esummary + efetch for abstracts).

```
Input: Search query (e.g., "BBBP prediction machine learning")
Output: List of papers with PMID, title, authors, journal, and abstract
Limit: 3 searches per iteration
Rate limit: 0.34s between requests (3 req/s without NCBI_API_KEY)
```

**Environment:** Set `NCBI_API_KEY` to boost rate limit from 3 to 10 req/s (free key from NCBI).

---

### `lookup_compound`
**Used by:** Hypothesis Agent
**Source:** `src/pharma_agents/tools/tooluniverse.py`

Looks up real experimental properties for a molecule from PubChem PUG REST API. Resolves SMILES or compound names to CID, then fetches molecular properties.

```
Input: SMILES string (e.g., "CCO") or compound name (e.g., "aspirin")
Output: MW, LogP (XLogP3), TPSA, HBD, HBA, rotatable bonds, canonical SMILES
Limit: 5 lookups per iteration
Rate limit: 0.5s between requests
```

**Resolution strategy:** Tries SMILES → PubChem CID first (heuristic: contains `()=#[]` or short uppercase start), then falls back to name lookup.

---

### `validate_experimental`
**Used by:** Evaluator Agent
**Source:** `src/pharma_agents/tools/tooluniverse.py`

Validates model predictions against experimental data from PubChem. Looks up real property values for a list of molecules.

```
Input: JSON with "smiles_list" (up to 10 SMILES) and "property" (e.g., "logP", "mw", "tpsa")
Output: ASCII table: SMILES | Experimental Value | Source, with found/total count
Limit: 3 calls per iteration
```

**Property mapping:**
| Input | PubChem Field |
|-------|---------------|
| `logP` | XLogP |
| `mw` | MolecularWeight |
| `tpsa` | TPSA |
| `hbd` | HBondDonorCount |
| `hba` | HBondAcceptorCount |

---

## Knowledge Base Tools (Hybrid RAG)

### `query_knowledge_base`
**Used by:** Hypothesis Agent
**Source:** `src/pharma_agents/tools/knowledge_base.py`

Hybrid search over the internal knowledge base using BM25 (sparse) + dense embeddings + Reciprocal Rank Fusion. Returns top results with source file and section attribution.

```
Input: Search query (e.g., "BBB physicochemical property ranges")
Output: Top 5 matching chunks with source attribution and RRF scores
```

**Retrieval pipeline:**
1. Embed query with fastembed (BGE-small-en-v1.5)
2. Tokenize query for BM25
3. Dense search → top 20 by cosine similarity
4. Sparse search → top 20 by BM25 score
5. RRF merge (k=60) → final top 5

**Auto-builds index** on first query if `index.json` doesn't exist. Indexes all `.md` and `.csv` files in the knowledge base directory.

Example output:
```
Top 3 results for 'BBB physicochemical property ranges':

--- [Source: internal_reports/cns_drug_design_guidelines_v3.md § BBB-Specific Property Ranges] (score: 0.0312) ---
The following ranges are derived from analysis of our internal CNS compound library...

--- [Source: safety_docs/bbb_penetration_safety_review.md § Model Validation Requirements] (score: 0.0305) ---
Performance thresholds (per SOP-ML-VAL-003): ROC-AUC >= 0.85...
```

**Output formatting:**
- CSV chunks shown **in full** (already bounded at 10 rows per chunk — agent needs complete data)
- Markdown chunks truncated at 500 chars (enough for context, use `read_kb_source` for full content)

**Supported document formats:**
- Markdown (`.md`) — chunked at `## ` heading boundaries with overlap
- CSV (`.csv`) — rows converted to natural language, batched in groups of 10

**Rebuild index (CLI, not an agent tool):**
```bash
uv run python -m pharma_agents.ingest_kb -e bbbp
```
Use this after adding new documents to the knowledge base directory. The index also auto-builds on first query if missing.

---

### `read_kb_source`
**Used by:** Hypothesis Agent
**Source:** `src/pharma_agents/tools/knowledge_base.py`

Read the full content of a knowledge base source file. Use after `query_knowledge_base` when the agent needs the complete document or dataset (especially CSV files).

```
Input: source_file path from a search result (e.g., "assay_data/bbb_pampa_assay_results.csv")
Output: Full file content (up to 5000 chars)
```

**Security:** Path traversal guard — rejects any path that resolves outside the knowledge base directory.

**Typical workflow:**
1. `query_knowledge_base("PAMPA assay results")` → sees partial CSV rows in results
2. `read_kb_source("assay_data/bbb_pampa_assay_results.csv")` → gets all 50 rows

See [docs/RAG.md](RAG.md) for the full retrieval flow diagram.

---

## Skill Tools

### `discover_skills`
**Used by:** Hypothesis Agent
**Source:** `src/pharma_agents/tools/skills.py`

Lists available skills with descriptions, filtered by keyword. Parses YAML frontmatter from skill files to extract name and description.

```
Input: Keyword to filter (e.g., "drug", "compound", "molecular") or "all"
Output: List of matching skill names with truncated descriptions
```

**Search scope:** Scans three locations:
1. `.claude/skills/scientific/*.md` — flat scientific skills (rdkit, deepchem, etc.)
2. `.claude/skills/*.md` — root-level skills
3. `.claude/skills/*/SKILL.md` — directory-based skills (tooluniverse-*)

**Typical workflow:** Agent calls `discover_skills("drug")` → sees 16 matching skills → calls `load_skill("tooluniverse-drug-research")` to load the one it needs.

---

### `load_skill`
**Used by:** Hypothesis Agent
**Source:** `src/pharma_agents/tools/skills.py`

Loads a skill by name to get domain knowledge, code examples, or workflow guides. Use `discover_skills` first to find available skills.

```
Input: Skill name (e.g., "rdkit", "tooluniverse-drug-research")
Output: Full skill content (max 8000 chars, truncated if longer)
Limit: 3 skills per iteration
```

**Search paths (tried in order):**
1. `.claude/skills/scientific/<name>.md`
2. `.claude/skills/<name>.md`
3. `.claude/skills/<name>/SKILL.md`

**Available skills include:** rdkit, deepchem, datamol, molfeat, pytdc, chembl-database, pubchem-database, literature-review, plus 15 tooluniverse-* workflow playbooks (drug-research, literature-deep-research, chemical-compound-retrieval, chemical-safety, drug-repurposing, target-research, etc.)

---

## Per-Iteration Counter Resets

All tools with ClassVar counters are reset at the start of each iteration via `_reset_tool_counters()` in `main.py`. This ensures rate limits apply **per iteration**, not across the entire run.

Tools with `reset_counters()` classmethod:
- `AlphaxivTool` — resets `_papers_fetched`, `_last_fetch`
- `ArxivSearchTool` — resets `_searches_done`, `_last_search`
- `FetchMorePapersTool` — resets `_calls_done`
- `InstallPackageTool` — resets `_packages_installed`
- `SkillLoaderTool` — resets `_skills_loaded`
- `PubMedSearchTool` — resets `_calls_done`, `_last_call`
- `CompoundLookupTool` — resets `_calls_done`, `_last_call`
- `ExperimentalValidationTool` — resets `_calls_done`

---

## Tool Constraints Summary

| Tool | Rate Limit | Max Per Iteration | Caching |
|------|------------|-------------------|---------|
| `read_train_py` | — | — | default (cached) |
| `edit_train_py` | — | — | default (cached) |
| `search_train_py` | — | — | default (cached) |
| `write_train_py` | — | — | default (cached) |
| `code_check` | — | — | **disabled** |
| `run_train_py` | — | — | **disabled** |
| `install_package` | — | 3 installs | **disabled** |
| `search_arxiv` | 3s interval | 8 searches | **disabled** |
| `fetch_arxiv_paper` | 1s interval | 10 papers | **disabled** |
| `store_paper` | — | — | default (cached) |
| `query_literature` | — | — | default (cached) |
| `fetch_more_papers` | — | 2 calls | **disabled** |
| `search_pubmed` | 0.34s interval | 3 searches | **disabled** |
| `lookup_compound` | 0.5s interval | 5 lookups | **disabled** |
| `validate_experimental` | 0.3s/molecule | 3 calls | **disabled** |
| `query_knowledge_base` | — | — | default (cached) |
| `read_kb_source` | — | — | default (cached) |
| `discover_skills` | — | — | default (cached) |
| `load_skill` | — | 3 skills | default (cached) |

---

## Adding New Tools

New tools should:

1. Inherit from `crewai.tools.BaseTool`
2. Define `name`, `description`, and `_run()` method
3. Include rate limiting for external APIs
4. **Disable caching** for state-dependent tools:
   ```python
   cache_function: object = lambda _args, _result: False  # type: ignore[assignment]
   ```
   **Warning:** `cache_function: None = None` does NOT disable caching — CrewAI treats `None` as "use default" which caches. You must provide a callable that returns `False`.
5. If the tool uses ClassVar counters, add a `reset_counters()` classmethod and register it in `main.py:_reset_tool_counters()`

Example:
```python
class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something useful. Input: X. Output: Y."
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    _calls_done: ClassVar[int] = 0

    @classmethod
    def reset_counters(cls) -> None:
        cls._calls_done = 0

    def _run(self, query: str) -> str:
        # Implementation
        return "result"
```

---

*Tools are defined in `src/pharma_agents/tools/` — split by domain: `training.py`, `arxiv.py`, `literature.py`, `knowledge_base.py`, `skills.py`, `tooluniverse.py`*
