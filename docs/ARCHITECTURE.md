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

---

*Architecture documentation for pharma-agents v0.17+*
