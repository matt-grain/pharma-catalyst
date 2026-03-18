# Architecture

## Overview

Pharma-agents is an autonomous multi-agent system that iteratively improves molecular property prediction models. Built on CrewAI, it uses a crew of specialized agents that collaborate in a sequential process: research literature, propose hypotheses, implement changes, and evaluate results.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Sequential Pipeline | CrewAI 0.100+ |
| Adversarial Review | AG2 (AutoGen) 0.9+ |
| LLM | Gemini Flash (configurable) |
| Embeddings | fastembed (BGE-small-en-v1.5, 384 dims) |
| ML Framework | scikit-learn, RDKit, XGBoost, LightGBM, TF/Keras... |
| Package Manager | uv |
| Linting | ruff |
| PubMed/PubChem/ChEMBL | Direct REST API wrappers (NCBI E-utilities, PUG REST) |
| Version Control | Git (worktrees for isolation) |

## Project Structure

```
pharma-agents/
├── src/pharma_agents/
│   ├── main.py              # Entry point, run loop, worktree management
│   ├── crew.py              # CrewAI crew & agent definitions
│   ├── review_panel.py      # AutoGen expert review panel (GroupChat)
│   ├── review_config.py     # Review panel config loader & constants
│   ├── review_agents.yaml   # Review panel agent definitions (names + prompts)
│   ├── memory.py            # AgentMemory persistence (what worked/failed)
│   ├── report.py            # HTML report generation
│   ├── agents.yaml          # Agent configurations
│   ├── tasks.yaml           # Task prompts & workflows
│   └── tools/               # Custom CrewAI tools
│       ├── __init__.py
│       ├── arxiv.py         # ArxivSearchTool, AlphaxivTool
│       ├── knowledge_base.py # KnowledgeQueryTool (hybrid BM25+dense RAG)
│       ├── literature.py    # LiteratureStoreTool, LiteratureQueryTool
│       ├── training.py      # ReadTrainPy, WriteTrainPy, CodeCheck, RunTrainPy
│       ├── skills.py        # SkillDiscoveryTool, SkillLoaderTool
│       └── tooluniverse.py  # PubMedSearch, CompoundLookup, ExperimentalValidation
├── experiments/
│   └── <experiment>/        # e.g., bbbp/
│       ├── baseline.json    # Baseline metric & config
│       ├── baseline_train.py
│       ├── train.py         # Modified by agents
│       ├── memory.json      # Cross-run agent memory
│       └── knowledge_base/  # Internal docs for RAG (reports, assays, SOPs)
│       └── literature/      # Fetched papers + embeddings
├── skills/                  # Scientific skills (rdkit, deepchem, etc.)
└── docs/
    ├── ARCHITECTURE.md      # This file
    └── agents.md            # Agent profiles & workflows
```

## Dual-Framework Architecture: CrewAI + AutoGen

The system deliberately uses two agent frameworks, each for what it does best:

- **CrewAI** handles the sequential ML research pipeline (archivist → hypothesis → implement → evaluate). CrewAI excels at task-based workflows with clear hand-offs.
- **AutoGen (AG2)** handles the adversarial expert review panel. AutoGen's `GroupChat` enables multi-perspective debate where agents build on each other's arguments — a fundamentally different interaction pattern from sequential tasks.

### Review Panel Flow (default)

```
[CrewAI Hypothesis Crew]
     │
     ▼  HypothesisOutput (Pydantic)
[AutoGen Review Panel: Statistician + Medicinal Chemist + Devil's Advocate → Moderator]
     │
     ▼  ReviewVerdict (approved / revised / rejected)
[CrewAI Implementation Crew: implement_task → evaluate_task]
     or
[Skip iteration if rejected — feedback stored in memory]
```

The review panel catches bad hypotheses before committing to 3-5 minutes of model training.
Use `--no-review` to bypass the panel and use the legacy single-crew flow.

## CrewAI Integration

### Crew Definition

```python
@CrewBase
class PharmaAgentsCrew:
    agents_config = "agents.yaml"
    tasks_config = "tasks.yaml"

    @agent
    def hypothesis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["hypothesis_agent"],
            llm=get_llm(),
            tools=[ReadTrainPyTool(), LiteratureQueryTool(),
                   SkillDiscoveryTool(), SkillLoaderTool(),
                   CompoundLookupTool()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.hypothesis_agent(), self.model_agent(), self.evaluator_agent()],
            tasks=[self.hypothesis_task(), self.implement_task(), self.evaluate_task()],
            process=Process.sequential,
        )
```

### Process Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHARMA-AGENTS WORKFLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   START RUN     │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │         First run OR stuck?          │
                    └──────────────────┬──────────────────┘
                           YES │              │ NO
                               ▼              │
                    ┌──────────────────┐      │
                    │    ARCHIVIST     │      │
                    │  (async/parallel)│      │
                    │   - search arxiv │      │
                    │   - fetch papers │      │
                    │   - store + embed│      │
                    └────────┬─────────┘      │
                             │                │
                             └────────┬───────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEQUENTIAL CREW PROCESS                              │
│                                                                             │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐          │
│  │   HYPOTHESIS    │   │     MODEL       │   │   EVALUATOR     │          │
│  │     AGENT       │ → │     AGENT       │ → │     AGENT       │          │
│  ├─────────────────┤   ├─────────────────┤   ├─────────────────┤          │
│  │ query_literature│   │ read_train_py   │   │ run_train_py    │          │
│  │ read_train_py   │   │ write_train_py  │   │ validate_experi.│          │
│  │ discover_skills │   │ code_check      │   │ compare baseline│          │
│  │ load_skill      │   │ install_package │   │ KEEP or REVERT  │          │
│  │ lookup_compound │   │                 │   │                 │          │
│  │ fetch_more_papers│  │                 │   │                 │          │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘          │
│         │                      │                      │                    │
│         ▼                      ▼                      ▼                    │
│     PROPOSAL             train.py modified      score comparison           │
│     + REASONING          + linted               + recommendation           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────┐
                           │  UPDATE MEMORY   │
                           │  - what worked   │
                           │  - what failed   │
                           │  - best score    │
                           └────────┬─────────┘
                                    │
                           ┌────────┴────────┐
                           │ More iterations? │
                           └────────┬────────┘
                          YES │          │ NO
                              │          ▼
                              │   ┌──────────────┐
                              │   │ GENERATE     │
                              │   │ REPORT       │
                              │   └──────────────┘
                              │
                              └───► (loop back to HYPOTHESIS)
```

## Tools Architecture

### Custom Tools

| Tool | Module | Purpose |
|------|--------|---------|
| `ArxivSearchTool` | arxiv.py | Search arxiv API for recent papers |
| `AlphaxivTool` | arxiv.py | Fetch paper summaries from alphaxiv (markdown) |
| `LiteratureStoreTool` | literature.py | Store papers with fastembed embeddings |
| `LiteratureQueryTool` | literature.py | Semantic search over literature |
| `FetchMorePapersTool` | literature.py | On-demand paper fetching when stuck |
| `ReadTrainPyTool` | training.py | Read train.py (constrained to one file) |
| `WriteTrainPyTool` | training.py | Write train.py (full file replacement) |
| `CodeCheckTool` | training.py | Run ruff linter on train.py |
| `RunTrainPyTool` | training.py | Execute training, extract metric |
| `InstallPackageTool` | training.py | Install whitelisted ML packages via uv |
| `SkillDiscoveryTool` | skills.py | List available skills filtered by keyword |
| `SkillLoaderTool` | skills.py | Load scientific skills (rdkit patterns, etc.) |
| `KnowledgeQueryTool` | knowledge_base.py | Hybrid BM25+dense search over internal docs |
| `KnowledgeIngestTool` | knowledge_base.py | Rebuild KB index from source documents |
| `PubMedSearchTool` | tooluniverse.py | Search PubMed + Semantic Scholar |
| `CompoundLookupTool` | tooluniverse.py | PubChem/ChEMBL compound properties |
| `ExperimentalValidationTool` | tooluniverse.py | Validate predictions vs experimental data |

### Why Custom Tools?

1. **Safety**: Constrained file access (only train.py, not arbitrary paths)
2. **Domain-specific**: Alphaxiv/arxiv integration, fastembed for embeddings
3. **Whitelisted installs**: Only approved ML packages can be installed
4. **Metric extraction**: Parse training output for specific metric format

## Memory System

```
experiments/<exp>/memory.json
├── runs/
│   └── <run_id>/
│       ├── experiments[]     # Per-iteration results
│       │   ├── hypothesis
│       │   ├── reasoning
│       │   ├── result (success/failure)
│       │   ├── score_before / score_after
│       │   └── improvement_pct
│       ├── best_score
│       ├── consecutive_failures
│       └── conclusion        # PROGRESS_CONTINUING, STAGNANT, etc.
└── global_best_score
```

**Memory informs agents:**
- "What Worked" section: successful hypotheses to build on
- "What Failed" section: approaches to avoid
- Stagnation detection: triggers exploration mode (archivist re-runs)

## Git Worktree Isolation

Each run operates in an isolated git worktree:

```
.worktrees/
└── bbbp/
    └── run_001/           # Isolated copy of repo
        └── experiments/
            └── bbbp/
                └── train.py   # Agent modifies this
```

**Benefits:**
- Changes don't affect main repo until committed
- Easy revert: just delete worktree
- Parallel runs possible (different worktrees)

## Literature Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    LITERATURE PIPELINE                           │
│                                                                  │
│  ┌────────────┐     ┌────────────┐     ┌────────────────────┐   │
│  │ ArxivSearch│ ──► │ Alphaxiv   │ ──► │ LiteratureStore    │   │
│  │   Tool     │     │   Tool     │     │      Tool          │   │
│  └────────────┘     └────────────┘     └────────────────────┘   │
│       │                   │                     │                │
│       ▼                   ▼                     ▼                │
│   paper IDs        markdown content      fastembed (384 dims)   │
│   + abstracts      (full or summary)     + index.json           │
│                                                                  │
│                         Storage:                                 │
│  experiments/<exp>/literature/                                   │
│  ├── index.json           # Paper metadata + embeddings         │
│  └── papers/                                                     │
│      ├── 2107.06773v2.md      # Summary markdown                │
│      └── 2107.06773v2_full.md # Full content (if alphaxiv)      │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ LiteratureQueryTool                                        │ │
│  │   - Embeds query with fastembed                            │ │
│  │   - Cosine similarity search                               │ │
│  │   - Returns top-k relevant papers                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Knowledge Base RAG (Hybrid Retrieval)

Internal document retrieval using BM25 + dense embeddings with Reciprocal Rank Fusion.
Ingests markdown reports, CSV assay data, SOPs, and safety docs — simulating a pharma
company's internal knowledge base.

```
┌──────────────────────────────────────────────────────────────────┐
│                  KNOWLEDGE BASE RAG                               │
│                                                                   │
│  Ingest:  md/csv → chunk (500 words, 50 overlap)                 │
│           → embed (fastembed BGE-small, 384d) + BM25 vocab        │
│           → store in index.json                                   │
│                                                                   │
│  Query:   query → embed + tokenize                                │
│           → dense top-20 (cosine) + sparse top-20 (BM25)          │
│           → RRF merge (k=60)                                      │
│           → return top-5 with source attribution                  │
│                                                                   │
│  Storage:                                                         │
│  experiments/<exp>/knowledge_base/                                 │
│  ├── index.json                  # Chunk index + BM25 stats       │
│  ├── internal_reports/*.md       # Benchmarks, guidelines          │
│  ├── assay_data/*.csv            # PAMPA-BBB, model history        │
│  ├── safety_docs/*.md            # Risk assessments                │
│  └── sops/*.md                   # Validation procedures           │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ KnowledgeQueryTool (hypothesis agent)                         │ │
│  │   - Hybrid search: BM25 sparse + dense cosine                 │ │
│  │   - Reciprocal Rank Fusion (k=60)                             │ │
│  │   - Returns top-k chunks with source § section attribution    │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ KnowledgeIngestTool (archivist agent)                         │ │
│  │   - Scans kb_dir for .md and .csv files                       │ │
│  │   - Chunks at section/paragraph boundaries                    │ │
│  │   - Batch embeds + builds BM25 vocabulary                     │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Why Hybrid Retrieval?

The system combines four complementary retrieval techniques:

| Technique | What it solves | How it works |
|-----------|---------------|--------------|
| **BM25** (sparse) | Exact term matching — drug names, specific metrics, acronyms | Classic term-frequency scoring with IDF weighting (Robertson, 1994). Query "XGBoost ROC-AUC" finds chunks containing those exact tokens. |
| **Dense embeddings** (semantic) | Meaning similarity — "similar concept, different words" | fastembed (BGE-small, 384d) encodes query and chunks into vectors. Cosine similarity finds semantically related content even without keyword overlap. |
| **Reciprocal Rank Fusion** (RRF) | Merging two ranked lists on incompatible scales | BM25 scores (0-15) and cosine scores (0-1) can't be averaged. RRF ignores scores entirely — it uses **rank positions**: `rrf_score = 1/(k + rank_in_BM25) + 1/(k + rank_in_dense)` with k=60. A chunk ranked high in both lists gets the best combined score. No normalization needed. (Cormack et al., 2009) |
| **Contextual retrieval** | Chunks losing parent document context after splitting | Each chunk's embedding is computed from `"[From: {doc_title}] {section}: {content}"` instead of raw content alone. The embedding vector captures both local detail and global document context. (Anthropic cookbook, 2024) |

Without hybrid retrieval, a query like "SOP validation thresholds" would fail on dense-only search (too specific) or BM25-only search (misses semantic matches about "performance requirements"). Together, they cover both angles.

#### Example: how RRF merges rankings

```
Query: "ensemble model performance"

BM25 ranking:                    Dense ranking:
  #1  benchmark § Key Findings     #1  benchmark § Model Comparison
  #2  benchmark § Recommendations  #2  benchmark § Key Findings
  #3  history_csv § rows_11-20     #3  benchmark § Recommendations

RRF merge (k=60):
  benchmark § Key Findings:    1/(60+1) + 1/(60+2) = 0.0328  ← best in both
  benchmark § Recommendations: 1/(60+2) + 1/(60+3) = 0.0320
  benchmark § Model Comparison: 0      + 1/(60+1)  = 0.0164
```

## Scientific Data Integration (PubMed, PubChem, ChEMBL)

Direct REST API wrappers for biomedical literature and compound databases.
No heavy SDK dependencies — just urllib calls to public APIs.

```
┌──────────────────────────────────────────────────────────────────┐
│                   SCIENTIFIC DATA TOOLS                          │
│                                                                  │
│  Agent Tool Wrappers (CrewAI BaseTool)                          │
│  ├── PubMedSearchTool ──► NCBI E-utilities (esearch/esummary)   │
│  ├── CompoundLookupTool ──► PubChem PUG REST API                │
│  └── ExperimentalValidationTool ──► PubChem property lookups    │
│                                                                  │
│  Skill Playbooks (.claude/skills/tooluniverse-*/SKILL.md)       │
│  └── 15 curated pharma workflows loaded via SkillLoaderTool     │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Direct REST over SDK | ToolUniverse SDK has heavy transitive deps (torch, chemprop); REST is zero-dep |
| PubChem PUG REST for compounds | Free, no API key, comprehensive molecular properties |
| 15 curated ToolUniverse skills | Workflow playbooks for agents, loaded via SkillLoaderTool |

## Configuration

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `PHARMA_EXPERIMENT` | Active experiment name | Required |
| `LLM_MODEL` | LLM model string | `gemini/gemini-3-flash-preview` |
| `GOOGLE_API_KEY` | Gemini API key | Required |
| `PHARMA_EXPERIMENTS_DIR` | Override experiments path | (auto-detected) |
| `NCBI_API_KEY` | PubMed rate boost (3→10 req/s) | Optional |
| `PHARMA_KB_DIR` | Override knowledge base path | (auto-detected) |

### Baseline Config

```json
// experiments/<exp>/baseline.json
{
  "metric": "ROC_AUC",
  "score": 0.8951,
  "direction": "higher_is_better",
  "property": "blood-brain barrier penetration"
}
```

## Guardrails & Safety Architecture

Autonomous agents modifying and executing code require defense-in-depth. The system implements
guardrails across seven layers, from code-level constraints to organizational review gates.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        GUARDRAIL LAYERS                                │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ L1. CODE SAFETY        Dangerous pattern blocking, code linting  │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ L2. FILE ISOLATION     train.py only, path traversal guard       │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ L3. DEPENDENCY CONTROL Whitelisted packages, max 3 installs/run  │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ L4. EXECUTION LIMITS   Timeouts, iteration caps, rate limiting   │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ L5. ADVERSARIAL REVIEW 5-expert panel gates implementation       │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ L6. GIT ISOLATION      Worktrees, auto-revert, baseline lock     │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ L7. ADAPTIVE MEMORY    Stagnation detection, exploration mode    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### L1. Code Safety — Dangerous Pattern Blocking

Agents can write Python code to `train.py`. All writes and edits are scanned for dangerous patterns
before hitting disk.

| Blocked Pattern | Risk |
|-----------------|------|
| `os.system(`, `subprocess.*` | Arbitrary command execution |
| `eval(`, `exec(` | Code injection |
| `shutil.rmtree(`, `os.remove(` | Filesystem destruction |
| `__import__(` | Dynamic import bypass |

**Config:** `src/pharma_agents/tool_defaults.yaml` (global) + `experiments/<exp>/tool_config.yaml` (per-experiment override). Loaded via `tool_config.py`.

**Additional write-time validation** (WriteTrainPyTool):
- Minimum 50 chars (reject empty files)
- Must contain `def train` (core contract)
- Must have `import` statements
- Warns if feature computation detected without inf/NaN handling
- Auto-fixes double-encoded escape sequences from JSON serialization

**Linting:** CodeCheckTool runs `ruff check` after every write. Agent must fix errors before finishing.

### L2. File Isolation — Constrained Access

Agents can **only** read and modify `train.py` in the current experiment directory. No other file
access is possible through agent tools.

- `ReadTrainPyTool`, `WriteTrainPyTool`, `EditTrainPyTool`, `SearchTrainPyTool` — all resolve to `get_experiments_dir() / "train.py"`
- Path traversal guard: rejects `PHARMA_EXPERIMENTS_DIR` values containing `..`
- `baseline_train.py` is **never modified** — it's the immutable reference
- Literature and knowledge base directories use the **main** experiments root (not the worktree), so agent code changes can't corrupt research data

### L3. Dependency Control — Package Whitelist

Agents can install packages via `InstallPackageTool`, but only from an approved list.

- **Global whitelist:** `tool_defaults.yaml` — 28 packages across core, ML, deep learning, cheminformatics, and visualization categories
- **Per-experiment override:** `experiments/<exp>/tool_config.yaml` — adds experiment-specific packages (e.g., `admet-ai` for clintox). Overrides **merge** with defaults (additive, not replacing).
- **Max installs per run:** 3 — prevents dependency explosion
- **Deduplication:** Same package can't be installed twice in one run

### L4. Execution Limits — Timeouts, Iteration Caps, Rate Limiting

Every agent and tool has bounded execution:

| Agent | Max Iterations | Max Time | Purpose |
|-------|---------------|----------|---------|
| Archivist | 20 | 10 min | Network I/O heavy |
| Hypothesis | 15 | 5 min | LLM reasoning |
| Model (implementation) | 40 | 10 min | Code fix cycles |
| Evaluator | 10 | 5 min | Includes training |

**Training timeout:** `run_train_py` kills the subprocess after 180 seconds. Prevents infinite loops, excessive hyperparameter sweeps.

**Tool rate limits** (per iteration, reset via `_reset_tool_counters()`):

| Tool | Limit | Rate |
|------|-------|------|
| Arxiv search | 8/iter | 3s interval |
| Paper fetch | 30/iter | 1s interval |
| Fetch more papers | 2/iter | — |
| PubMed search | 5/iter | 0.34s interval |
| Compound lookup | 3/iter | 0.5s interval |
| Skill loading | 3/iter | — |
| Package install | 3/run | — |

**LLM rate limiting** (LoggingLLM wrapper):
- Free tier: 4s minimum gap between calls, 10 RPM max
- Paid tier: 30 RPM max
- 429 errors: up to 3 retries with API-parsed or exponential backoff
- Permanent errors (auth, model not found): abort entire run immediately

**Content truncation** — prevents token bloat in agent context:
- Hypothesis text: 200 chars max
- Reasoning text: 300 chars max
- Paper summaries: 1,000 chars max
- PDF content: 15,000 chars / 8 pages max

### L5. Adversarial Review — Expert Panel Gating

Before any hypothesis reaches implementation, it faces a 5-expert review panel (AG2 GroupChat):

| Panelist | Focus |
|----------|-------|
| Statistician | Sample size, overfitting, statistical validity |
| Medicinal Chemist | SAR, feature relevance, chemical plausibility |
| Devil's Advocate | Data leakage, compute risk, blind spots |
| Team Memory Analyst | Novelty check, history of what worked/failed |
| Pharma Ethics Reviewer | Interpretability, bias, regulatory alignment |

The **Moderator** synthesizes the debate into a structured verdict:
- `approved` → proceed to implementation
- `revised` → implementation uses revised proposal
- `rejected` → skip iteration, feedback stored in memory for next hypothesis

**Fallback:** If the review panel fails (API error), it defaults to `approved` with low confidence (0.2) rather than blocking the pipeline. Rate-limited partial debates attempt to salvage the moderator's verdict.

### L6. Git Isolation — Worktrees & Auto-Revert

Each run operates in a **separate git worktree** — a physical directory with its own branch, completely isolated from `main`.

- **On start:** `.worktrees/<exp>/run_XXX/` created from main
- **Baseline committed:** Initial state locked in the branch
- **On improvement:** Changes committed with metric and improvement percentage
- **On failure:** `git checkout train.py` reverts to last good state
- **On crew error:** Auto-revert before next iteration
- **Main branch:** Never touched during runs — only modified via explicit `promote` command

### L7. Adaptive Memory — Stagnation Detection & Circuit Breakers

The memory system detects when agents are stuck and adapts:

| Condition | Threshold | Response |
|-----------|-----------|----------|
| Per-run stuck | 3+ consecutive failures | Exploration mode triggered |
| Global stagnation | 5 experiments without improvement | Archivist re-runs, radical ideas prompt injected |
| Local optimum | 2+ failures OR < 1% avg improvement | Run concludes, notes left for future runs |

**Exploration mode** injects a strong prompt override: *"GLOBAL STAGNATION... You MUST propose something RADICALLY DIFFERENT"* — preventing agents from getting trapped in local optima.

**Run conclusions** are classified and persisted as "Notes from Previous Researchers" for future runs:
- `PROGRESS_CONTINUING` — keep iterating
- `LOCAL_OPTIMUM` — try different architectures
- `STUCK` — exploration mode required

### What's NOT Implemented (Future Work)

| Gap | Risk | Mitigation Path |
|-----|------|----------------|
| **Full sandboxing** | train.py runs in the host Python process | Containerized execution (Docker) or subprocess with restricted permissions |
| **Budget caps** | No token/cost ceiling per run | Track LLM token usage in LoggingLLM, abort when budget exceeded |
| **Output validation** | Agent metric reports can hallucinate | Already mitigated: Python evaluation is the source of truth, not agent text. Could add schema validation on HypothesisOutput fields. |
| **Human-in-the-loop checkpoint** | No mandatory human approval gate | Add `--confirm` flag requiring human approval after review panel and before implementation |
| **Prompt injection defense** | Arxiv/PubMed content could contain adversarial text | Sanitize external content before injecting into agent prompts |
| **Audit logging** | Logs exist but no structured audit trail | Structured JSON event log for compliance (who did what, when, why) |

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CrewAI for pipeline, AutoGen for review | Each framework used for its strength (see ADR in decisions.md) |
| Sequential process (not parallel) | Each agent depends on previous output |
| fastembed over OpenAI embeddings | No API key required, faster, local |
| Worktrees over branches | Full file isolation, easy cleanup |
| Constrained file tools | Safety - agents can only modify train.py |
| Whitelisted package installs | Prevent arbitrary dependency injection |
| Memory persistence | Learn from past runs, avoid repeating failures |
| Hybrid RAG (BM25+dense+RRF) | Best of sparse and dense retrieval, no new deps |
| YAML tool config over hardcoded lists | Experiment-specific overrides without code changes |

---

*Architecture documentation for pharma-agents v0.18+*
