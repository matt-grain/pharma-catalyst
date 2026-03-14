# Agent Profiles

Meet the pharma-catalyst crew - autonomous agents working together to improve molecular property prediction.

---

## The Archivist

```
┌─────────────────────────────────────────────────────────────────┐
│                    FIRST RUN (or exploration)                   │
│                                                                 │
│  ┌──────────────┐    search     ┌──────────────┐               │
│  │   ARCHIVIST  │ ───────────►  │    arxiv     │               │
│  │     AGENT    │               │     API      │               │
│  └──────┬───────┘               └──────────────┘               │
│         │                                                       │
│         │ fetch papers                                          │
│         ▼                                                       │
│  ┌──────────────┐    fallback   ┌──────────────┐               │
│  │   alphaxiv   │ ◄──────────── │  arxiv abs   │               │
│  │  (full text) │   if 404      │ (via md.new) │               │
│  └──────┬───────┘               └──────────────┘               │
│         │                                                       │
│         │ store + embed (fastembed BGE-small 384d)              │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────┐               │
│  │ experiments/<exp>/literature/                │               │
│  │   ├── index.json        (embeddings + meta)  │               │
│  │   └── papers/                                │               │
│  │       ├── 2107.06773v2.md      (summary)     │               │
│  │       └── 2107.06773v2_full.md (full text)   │               │
│  └──────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                         │
                         │ persists for all runs
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SUBSEQUENT RUNS                              │
│                                                                 │
│  ┌──────────────┐  semantic search  ┌──────────────┐           │
│  │  HYPOTHESIS  │ ◄───────────────► │  LITERATURE  │           │
│  │    AGENT     │   (cosine sim)    │      DB      │           │
│  └──────────────┘                   └──────────────┘           │
│                                                                 │
│  Query: "gradient boosting molecular fingerprints"              │
│  Returns: Top 5 papers with similarity scores                   │
└─────────────────────────────────────────────────────────────────┘
```

**Role:** Literature Research Archivist
**Mission:** Gather recent (2023-2025) research papers on molecular ML

**Tools:**
| Tool | Description |
|------|-------------|
| `search_arxiv` | Search arxiv API for papers on ADMET, GNNs, fingerprints |
| `fetch_arxiv_paper` | Get paper summaries via alphaxiv (or arxiv fallback) |
| `store_paper` | Save with fastembed embeddings for semantic search |

**When Active:** First run of an experiment, or exploration mode (stuck/stagnant)

---

## The Scientist

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYPOTHESIS AGENT                             │
│                                                                 │
│     ┌─────────────────────────────────────────────────┐        │
│     │                INPUTS                           │        │
│     │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │        │
│     │  │  MEMORY   │  │ LITERATURE│  │   SKILLS  │   │        │
│     │  │ (what     │  │ (recent   │  │ (rdkit,   │   │        │
│     │  │  worked/  │  │  papers)  │  │  deepchem)│   │        │
│     │  │  failed)  │  │           │  │           │   │        │
│     │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │        │
│     │        │              │              │         │        │
│     │        └──────────────┼──────────────┘         │        │
│     │                       │                        │        │
│     └───────────────────────┼────────────────────────┘        │
│                             ▼                                  │
│                   ┌───────────────────┐                        │
│                   │    REASONING      │                        │
│                   │  ┌─────────────┐  │                        │
│                   │  │ Literature  │  │                        │
│                   │  │ + Memory +  │  │                        │
│                   │  │ Domain      │  │                        │
│                   │  │ Knowledge   │  │                        │
│                   │  └─────────────┘  │                        │
│                   └─────────┬─────────┘                        │
│                             │                                  │
│                             ▼                                  │
│                   ┌───────────────────┐                        │
│                   │     OUTPUT        │                        │
│                   │  ┌─────────────┐  │                        │
│                   │  │ PROPOSAL    │  │                        │
│                   │  │ + REASONING │  │                        │
│                   │  │ + CHANGE    │  │                        │
│                   │  └─────────────┘  │                        │
│                   └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

**Role:** Research Scientist specializing in molecular ML
**Mission:** Propose improvements to the ML pipeline

**Tools:**
| Tool | Description |
|------|-------------|
| `read_train_py` | Understand current implementation |
| `query_literature` | Semantic search over recent papers |
| `load_skill` | Load scientific skills (rdkit, deepchem patterns) |
| `fetch_more_papers` | Request fresh papers when stuck |

**Strategy:** Combine memory of past experiments + literature insights to propose novel approaches

---

## The Engineer

```
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL AGENT                                │
│                                                                 │
│         PROPOSAL                                                │
│            │                                                    │
│            ▼                                                    │
│  ┌───────────────────────────────────────────┐                 │
│  │  1. READ current train.py                 │                 │
│  └───────────────────────────────────────────┘                 │
│            │                                                    │
│            ▼                                                    │
│  ┌───────────────────────────────────────────┐                 │
│  │  2. INSTALL packages if needed            │                 │
│  │     (lightgbm, xgboost, catboost...)      │                 │
│  └───────────────────────────────────────────┘                 │
│            │                                                    │
│            ▼                                                    │
│  ┌───────────────────────────────────────────┐                 │
│  │  3. WRITE modified train.py               │                 │
│  │  ┌─────────────────────────────────────┐  │                 │
│  │  │  def train():                       │  │                 │
│  │  │    # Morgan FP (2048 bits, r=3)     │  │                 │
│  │  │    # + MACCS keys                   │  │                 │
│  │  │    # + Physicochemical descriptors  │  │                 │
│  │  │    model = XGBClassifier(...)       │  │                 │
│  │  │    return roc_auc                   │  │                 │
│  │  └─────────────────────────────────────┘  │                 │
│  └───────────────────────────────────────────┘                 │
│            │                                                    │
│            ▼                                                    │
│  ┌───────────────────────────────────────────┐                 │
│  │  4. CODE CHECK (ruff linter)              │                 │
│  │     ┌─────────┐                           │                 │
│  │     │ ruff ✓  │ ◄──── fix errors ─────┐   │                 │
│  │     │ pass    │                       │   │                 │
│  │     └─────────┘                       │   │                 │
│  │         │                             │   │                 │
│  │         │ errors?                     │   │                 │
│  │         └─────────────────────────────┘   │                 │
│  └───────────────────────────────────────────┘                 │
│            │                                                    │
│            ▼                                                    │
│        READY FOR EVALUATION                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Role:** ML Engineer
**Mission:** Implement proposed changes in train.py

**Tools:**
| Tool | Description |
|------|-------------|
| `read_train_py` | Get current code |
| `write_train_py` | Write modified code (full file) |
| `code_check` | Validate with ruff linter |
| `install_package` | Install ML packages (whitelisted: lightgbm, xgboost, catboost, etc.) |

**Constraint:** Must pass linting before finishing

---

## The Evaluator

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATOR AGENT                              │
│                                                                 │
│  ┌───────────────────────────────────────────┐                 │
│  │         RUN TRAINING                      │                 │
│  │  $ python train.py                        │                 │
│  │  > ROC_AUC: 0.9418                        │                 │
│  └───────────────────────────────────────────┘                 │
│                       │                                         │
│                       ▼                                         │
│  ┌───────────────────────────────────────────┐                 │
│  │         COMPARISON                        │                 │
│  │  ┌─────────────────────────────────────┐  │                 │
│  │  │  Score:     0.9418                  │  │                 │
│  │  │  Baseline:  0.8951                  │  │                 │
│  │  │  ────────────────────────────────   │  │                 │
│  │  │  Change:    +5.2%                   │  │                 │
│  │  │  Direction: higher_is_better        │  │                 │
│  │  └─────────────────────────────────────┘  │                 │
│  └───────────────────────────────────────────┘                 │
│                       │                                         │
│                       ▼                                         │
│  ┌───────────────────────────────────────────┐                 │
│  │         DECISION                          │                 │
│  │                                           │                 │
│  │    improved?                              │                 │
│  │       │                                   │                 │
│  │       ├── YES ──► ✅ KEEP                 │                 │
│  │       │           (commit changes)        │                 │
│  │       │                                   │                 │
│  │       └── NO ───► ❌ REVERT               │                 │
│  │                   (git reset)             │                 │
│  │                                           │                 │
│  └───────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

**Role:** QA Scientist
**Mission:** Run training and evaluate objectively

**Tools:**
| Tool | Description |
|------|-------------|
| `read_train_py` | Review implementation |
| `run_train_py` | Execute and extract metric |

**Output:** KEEP (if improved) or REVERT (if worse/equal)

---

## Crew Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHARMA-AGENTS WORKFLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ INITIALIZATION                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                              │
│  │  Create  │ ►  │  Load    │ ►  │  Check   │                              │
│  │ worktree │    │  memory  │    │literature│                              │
│  └──────────┘    └──────────┘    └──────────┘                              │
│                                        │                                    │
│                         ┌──────────────┴──────────────┐                     │
│                         │  Literature exists?         │                     │
│                         └──────────────┬──────────────┘                     │
│                                NO │          │ YES                          │
│                                   ▼          │                              │
│                         ┌──────────────┐     │                              │
│                         │  ARCHIVIST   │     │                              │
│                         │  (gather     │     │                              │
│                         │   papers)    │     │                              │
│                         └──────┬───────┘     │                              │
│                                │             │                              │
│                                └──────┬──────┘                              │
│                                       ▼                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ ITERATION LOOP (for each iteration)                                        │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SEQUENTIAL PROCESS                           │   │
│  │                                                                     │   │
│  │  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐        │   │
│  │  │  HYPOTHESIS   │   │     MODEL     │   │   EVALUATOR   │        │   │
│  │  │    AGENT      │ → │     AGENT     │ → │     AGENT     │        │   │
│  │  ├───────────────┤   ├───────────────┤   ├───────────────┤        │   │
│  │  │ • literature  │   │ • read code   │   │ • run train   │        │   │
│  │  │ • memory      │   │ • install pkg │   │ • compare     │        │   │
│  │  │ • skills      │   │ • write code  │   │ • recommend   │        │   │
│  │  │               │   │ • lint check  │   │               │        │   │
│  │  └───────────────┘   └───────────────┘   └───────────────┘        │   │
│  │         │                   │                    │                 │   │
│  │         ▼                   ▼                    ▼                 │   │
│  │     PROPOSAL           train.py             KEEP/REVERT           │   │
│  │                        modified                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        POST-ITERATION                               │   │
│  │                                                                     │   │
│  │   KEEP? ──► git commit ──► update memory ──► update baseline       │   │
│  │   REVERT? ──► git reset ──► log failure ──► continue               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPLETION                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                              │
│  │  Merge   │ ►  │ Generate │ ►  │  Cleanup │                              │
│  │ worktree │    │  report  │    │ worktree │                              │
│  └──────────┘    └──────────┘    └──────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Memory System

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT MEMORY                                 │
│                                                                 │
│  experiments/<exp>/memory.json                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  {                                                        │ │
│  │    "runs": {                                              │ │
│  │      "1": {                                               │ │
│  │        "best_score": 0.9300,                              │ │
│  │        "experiments": [                                   │ │
│  │          {                                                │ │
│  │            "hypothesis": "Add physicochemical...",        │ │
│  │            "result": "success",                           │ │
│  │            "improvement_pct": 3.9                         │ │
│  │          }                                                │ │
│  │        ],                                                 │ │
│  │        "conclusion": "PROGRESS_CONTINUING"                │ │
│  │      },                                                   │ │
│  │      "2": { ... }                                         │ │
│  │    },                                                     │ │
│  │    "global_best_score": 0.9418                            │ │
│  │  }                                                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Used by Hypothesis Agent:                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  "What Worked" ──► Build on successful approaches         │ │
│  │  "What Failed" ──► Avoid repeating mistakes               │ │
│  │  "Best Score"  ──► Track progress                         │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

*Built with CrewAI. Powered by Gemini. Optimized by curiosity.*
