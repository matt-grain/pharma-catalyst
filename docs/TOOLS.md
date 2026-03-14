# Tool Reference

All custom tools available to pharma-catalyst agents.

---

## Core Tools (ML Pipeline)

### `read_train_py`
**Used by:** Hypothesis Agent, Model Agent, Evaluator Agent

Reads the current content of `train.py`.

```
Input: (none)
Output: Full Python code of train.py
```

---

### `write_train_py`
**Used by:** Model Agent

Writes complete content to `train.py`, overwriting existing code.

```
Input: Full Python code (string)
Output: "Successfully wrote X characters to train.py"
```

---

### `code_check`
**Used by:** Model Agent

Runs ruff linter on `train.py` to check for syntax/style errors.

```
Input: (none)
Output: "OK - No linting errors" or list of errors to fix
```

**Note:** Caching disabled - always checks fresh file state.

---

### `run_train_py`
**Used by:** Evaluator Agent

Executes `train.py` and returns the validation metric.

```
Input: (none)
Output: "RMSE: 0.6532" or "ROC_AUC: 0.9286" (depending on experiment)
Timeout: 60 seconds
```

---

## Literature Tools (Archivist)

### `search_arxiv`
**Used by:** Archivist Agent

Searches arxiv.org for recent papers on a topic.

```
Input: Search query (e.g., "ADMET prediction graph neural network")
Output: List of paper IDs with titles and abstracts (max 10)
Limit: 5 searches per run
```

Example output:
```
Found 10 papers for 'ADMET prediction':

- **2401.12345** (2024-01-15): Graph Neural Networks for ADMET...
  We present a novel approach using message-passing...
```

---

### `fetch_arxiv_paper`
**Used by:** Archivist Agent

Fetches paper summary from alphaxiv.org as markdown.

```
Input: Paper ID (e.g., "2401.12345" or full arxiv URL)
Output: Structured AI-generated overview (markdown)
Limit: 10 papers per run
Rate limit: 1 second between requests
```

**Why alphaxiv?** Faster than PDF, optimized for LLM consumption.

---

### `store_paper`
**Used by:** Archivist Agent

Stores a paper summary with embeddings for semantic search.

```
Input: JSON with {paper_id, title, summary, key_methods[]}
Output: "Stored paper X with embedding (384 dims)"
```

Creates:
- `literature/papers/<paper_id>.md` - Markdown summary
- `literature/index.json` - Index with embeddings

**Embedding model:** fastembed `BAAI/bge-small-en-v1.5` (4.8ms/query)

---

### `query_literature`
**Used by:** Hypothesis Agent

Semantic search over the local literature database.

```
Input: Search query (e.g., "gradient boosting molecular features")
Output: Top 5 matching papers with summaries
```

Example output:
```
Top 5 papers for 'gradient boosting molecular features':

**[0.87] 2401.12345**: XGBoost for Molecular Property Prediction
  Gradient boosting with Morgan fingerprints achieves...

**[0.82] 2312.54321**: Feature Engineering for ADMET
  Combining physicochemical descriptors with...
```

---

## Knowledge Tools

### `load_skill`
**Used by:** Hypothesis Agent

Loads a scientific skill for domain knowledge and code examples.

```
Input: Skill name (e.g., "rdkit", "deepchem", "molfeat")
Output: Skill content with best practices and examples
Limit: 3 skills per run
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

## Tool Constraints Summary

| Tool | Rate Limit | Max Per Run |
|------|------------|-------------|
| `search_arxiv` | - | 5 searches |
| `fetch_arxiv_paper` | 1s interval | 10 papers |
| `store_paper` | - | - |
| `query_literature` | - | - |
| `load_skill` | - | 3 skills |
| `code_check` | no cache | - |

---

## Adding New Tools

New tools should:

1. Inherit from `crewai.tools.BaseTool`
2. Define `name`, `description`, and `_run()` method
3. Include rate limiting for external APIs
4. Disable caching if state-dependent: `cache_function: None = None`

Example:
```python
class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something useful. Input: X. Output: Y."

    def _run(self, input: str) -> str:
        # Implementation
        return "result"
```

---

*Tools are defined in `src/pharma_agents/tools/custom_tools.py`*
