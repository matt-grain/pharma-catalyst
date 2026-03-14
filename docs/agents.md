# Agent Profiles

Meet the pharma-catalyst crew - autonomous agents working together to improve molecular property prediction.

---

## The Archivist

```
┌─────────────────────────────────────────────────────────────┐
│                    FIRST RUN (or exploration)               │
│  ┌──────────────┐                                           │
│  │  Archivist   │ ──► alphaxiv skill ──► papers as MD      │
│  │    Agent     │ ──► web search ──► recent techniques     │
│  └──────┬───────┘                                           │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────┐                   │
│  │ experiments/<exp>/literature/        │                   │
│  │   ├── papers/                        │  (MD summaries)   │
│  │   ├── embeddings.db                  │  (vector search)  │
│  │   └── index.json                     │  (what's fetched) │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ query relevant papers
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    NORMAL RUNS                              │
│  ┌──────────────┐    context     ┌──────────────┐          │
│  │  Hypothesis  │ ◄───────────── │ Literature   │          │
│  │    Agent     │   (RAG-style)  │     DB       │          │
│  └──────────────┘                └──────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Role:** Literature Research Archivist
**Mission:** Gather recent (2023-2025) research papers on molecular ML

**Tools:**
- `search_arxiv` - Find papers on ADMET, GNNs, fingerprints
- `fetch_arxiv_paper` - Get paper summaries as markdown via alphaxiv
- `store_paper` - Save with embeddings for semantic search

**When Active:** First run of an experiment, or exploration mode

---

## The Scientist

```
    ╔═══════════════════════════════════════╗
    ║         RESEARCH SCIENTIST            ║
    ╠═══════════════════════════════════════╣
    ║                                       ║
    ║           🧪     💡     🧬           ║
    ║            \     |     /             ║
    ║             \    |    /              ║
    ║              ╔═══════╗               ║
    ║              ║ HYPO- ║               ║
    ║              ║ THESIS║               ║
    ║              ╚═══════╝               ║
    ║                 ↓                    ║
    ║         ┌─────────────┐              ║
    ║         │ PROPOSAL    │              ║
    ║         │ + REASONING │              ║
    ║         └─────────────┘              ║
    ║                                       ║
    ╚═══════════════════════════════════════╝
```

**Role:** Research Scientist specializing in molecular ML
**Mission:** Propose improvements to the ML pipeline

**Tools:**
- `read_train_py` - Understand current implementation
- `query_literature` - Search recent papers for techniques
- `load_skill` - Load scientific skills (rdkit, deepchem, etc.)

**Strategy:** Combine memory of past experiments + literature insights

---

## The Engineer

```
    ╔═══════════════════════════════════════╗
    ║            ML ENGINEER                ║
    ╠═══════════════════════════════════════╣
    ║                                       ║
    ║      ┌──────────────────────┐        ║
    ║      │  def train():        │        ║
    ║      │    # Morgan FP       │        ║
    ║      │    # + Descriptors   │  ←──┐  ║
    ║      │    model.fit(X, y)   │     │  ║
    ║      │    return rmse       │     │  ║
    ║      └──────────────────────┘     │  ║
    ║               ↓                   │  ║
    ║         ┌──────────┐              │  ║
    ║         │ ruff ✓   │──── fix ─────┘  ║
    ║         │ pyright ✓│                 ║
    ║         └──────────┘                 ║
    ║                                       ║
    ╚═══════════════════════════════════════╝
```

**Role:** ML Engineer
**Mission:** Implement proposed changes in train.py

**Tools:**
- `read_train_py` - Get current code
- `write_train_py` - Write modified code
- `code_check` - Validate with ruff

**Constraint:** Must pass linting before finishing

---

## The Evaluator

```
    ╔═══════════════════════════════════════╗
    ║           QA SCIENTIST                ║
    ╠═══════════════════════════════════════╣
    ║                                       ║
    ║         ┌───────────────┐            ║
    ║         │  RUN TRAINING │            ║
    ║         └───────┬───────┘            ║
    ║                 ↓                    ║
    ║     ┌───────────────────────┐        ║
    ║     │  RMSE: 0.6532         │        ║
    ║     │  Baseline: 1.3175     │        ║
    ║     │  ═══════════════════  │        ║
    ║     │  IMPROVEMENT: 50.4%   │        ║
    ║     │  ✅ KEEP              │        ║
    ║     └───────────────────────┘        ║
    ║                                       ║
    ╚═══════════════════════════════════════╝
```

**Role:** QA Scientist
**Mission:** Run training and evaluate objectively

**Tools:**
- `read_train_py` - Review implementation
- `run_train_py` - Execute and get metric

**Output:** KEEP (improved) or REVERT (worse)

---

## Crew Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                      FIRST RUN ONLY                            │
│  ┌──────────────┐                                              │
│  │  ARCHIVIST   │ → arxiv → alphaxiv → embeddings → literature/│
│  └──────────────┘                                              │
└────────────────────────────────┬────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EACH ITERATION                               │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  SCIENTIST   │ →  │  ENGINEER    │ →  │  EVALUATOR   │     │
│  │  (propose)   │    │  (implement) │    │  (evaluate)  │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                   │               │
│         ↓                   ↓                   ↓               │
│     literature/         train.py            KEEP/REVERT        │
│     + memory.json       + code_check        + commit/revert    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Built with CrewAI. Powered by Gemini. Inspired by curiosity.* 🧬
