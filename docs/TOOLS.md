# Tool Reference

All custom tools available to pharma-agents agents.

---

## Core Tools (ML Pipeline)

### `read_train_py`
**Used by:** Hypothesis Agent, Model Agent, Evaluator Agent
**Source:** `src/pharma_agents/tools/training.py`

Reads the current content of `train.py`.

```
Input: (none)
Output: Full Python code of train.py
```

---

### `write_train_py`
**Used by:** Model Agent
**Source:** `src/pharma_agents/tools/training.py`

Writes complete content to `train.py`, overwriting existing code.

**Validation (applied before writing):**
- Strips markdown code fences (```` ```python ... ``` ````) — common LLM output artifact
- Rejects content shorter than 50 characters
- Requires a `def train` function
- Requires at least one `import` statement
- Blocks dangerous patterns: `os.system()`, `subprocess.run()`, `eval()`, `exec()`, `shutil.rmtree()`, etc.

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

**Caching:** Disabled — always runs fresh training.

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

## Knowledge Tools

### `load_skill`
**Used by:** Hypothesis Agent
**Source:** `src/pharma_agents/tools/skills.py`

Loads a scientific skill for domain knowledge and code examples.

```
Input: Skill name (e.g., "rdkit", "deepchem", "molfeat")
Output: Skill content with best practices and examples (max 8000 chars)
Limit: 3 skills per iteration
```

**Available skills:**
| Skill | Description |
|-------|-------------|
| `rdkit` | Molecular featurization, fingerprints, descriptors |
| `deepchem` | Deep learning for drug discovery |
| `datamol` | Molecular data manipulation |
| `molfeat` | Molecular feature extraction |
| `pytdc` | Therapeutics Data Commons datasets |
| `chembl-database` | ChEMBL bioactivity data access |
| `pubchem-database` | PubChem compound database |
| `literature-review` | Systematic literature review methodology |

---

## Per-Iteration Counter Resets

All tools with ClassVar counters are reset at the start of each iteration via `_reset_tool_counters()` in `main.py`. This ensures rate limits apply **per iteration**, not across the entire run.

Tools with `reset_counters()` classmethod:
- `AlphaxivTool` — resets `_papers_fetched`, `_last_fetch`
- `ArxivSearchTool` — resets `_searches_done`, `_last_search`
- `FetchMorePapersTool` — resets `_calls_done`
- `InstallPackageTool` — resets `_packages_installed`
- `SkillLoaderTool` — resets `_skills_loaded`

---

## Tool Constraints Summary

| Tool | Rate Limit | Max Per Iteration | Caching |
|------|------------|-------------------|---------|
| `read_train_py` | — | — | default (cached) |
| `write_train_py` | — | — | default (cached) |
| `code_check` | — | — | **disabled** |
| `run_train_py` | — | — | **disabled** |
| `install_package` | — | 3 installs | **disabled** |
| `search_arxiv` | 3s interval | 8 searches | **disabled** |
| `fetch_arxiv_paper` | 1s interval | 10 papers | **disabled** |
| `store_paper` | — | — | default (cached) |
| `query_literature` | — | — | default (cached) |
| `fetch_more_papers` | — | 2 calls | **disabled** |
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

*Tools are defined in `src/pharma_agents/tools/` — split by domain: `training.py`, `arxiv.py`, `literature.py`, `skills.py`, `evaluate.py`*
