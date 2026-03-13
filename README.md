# pharma-catalyst

**Self-improving AI agents for molecular property prediction.**

Built with CrewAI, featuring persistent cross-run memory, automatic stuck detection, and exploration mode to escape local optima. Demonstrated on ESOL solubility prediction: **1.32 → 0.65 RMSE (50.4% improvement)** in 6 autonomous iterations.

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — applied to drug discovery ML.

![Python](https://img.shields.io/badge/python-3.12+-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-multi--agent-purple)
![License](https://img.shields.io/badge/license-MIT-green)

## The Idea

Give an AI agent crew a molecular ML task. Let them iterate autonomously. They propose changes, implement them, evaluate results, keep improvements, discard failures. You wake up to a better model.

## Architecture

```
                    +------------------+
                    |    Human (You)   |
                    |  Defines strategy|
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |      Crew        |
                    |    (CrewAI)      |
                    +--------+---------+
                             |
               +-------------+-------------+
               |                           |
               v                           v
        +-------------+             +-------------+
        | Hypothesis  |             |   Model     |
        |   Agent     | ─────────►  |   Agent     |
        +-------------+             +-------------+
        Research Scientist           ML Engineer
               │                           │
               │                           v
               │                    +-------------+
               │                    | code_check  |
               │                    | (ruff+pyright)
               │                    +-------------+
               │                           │
               v                           v
        +------------------------------------------+
        |           Python Evaluation              |
        |  (run_training → compare → commit/revert)|
        +------------------------------------------+
                             │
                             v
                    +------------------+
                    |  Agent Memory    |
                    | (experiments/    |
                    |  memory.json)    |
                    +------------------+
```

**Sequential process:**
1. Hypothesis Agent proposes an improvement (informed by memory)
2. Model Agent implements it in `train.py` (validates with ruff+pyright)
3. Python runs training, compares RMSE (no LLM hallucination possible)
4. If improved → commit + save to memory. If worse → revert + save failure to memory.
5. Next iteration sees what worked and what failed.

## Target Task

**ESOL Solubility Prediction** - a classic ADMET benchmark.

- ~1,128 molecules
- Predict aqueous solubility (logS)
- Metric: RMSE (lower = better)
- Baseline: ~1.32 RMSE with RandomForest + Morgan fingerprints

CPU-friendly. No GPU required. Trains in <1 second.

## Quick Start

```bash
# Install dependencies
uv sync

# Set your LLM API key (Gemini example)
export GOOGLE_API_KEY="your-key-here"

# Download ESOL dataset (one-time)
uv run python -m pharma_agents.data.fetch

# Run baseline training
uv run python -m pharma_agents.tools.train

# Run the agent crew (5 iterations by default)
uv run python -m pharma_agents.main
```

## Git Workflow

Each run uses an **isolated worktree**, keeping `main` completely untouched.

```
pharma-catalyst/
  │
  ├── main branch (baseline, never touched during runs)
  │
  └── .worktrees/
        ├── run_001/ ──► isolated copy, agents iterate here
        │     └── branch: run/001
        │
        ├── run_002/ ──► another isolated copy
        │     └── branch: run/002
        │
        └── ...
```

**Why worktrees?**
- No stash/conflict issues when switching runs
- Main stays clean - no accidental commits
- Can run multiple experiments in parallel
- Clean discard: just remove the worktree

### Commands

```bash
# Start a new run (creates worktree + branch)
uv run python -m pharma_agents.main

# Promote a successful run to main (makes it the new baseline)
uv run python -m pharma_agents.promote 1

# Discard a cancelled/stuck run (removes worktree, memory entry, branch)
uv run python -m pharma_agents.discard 2
uv run python -m pharma_agents.discard 2 --keep-logs  # keep log files

# Reset train.py to baseline manually
uv run python -m pharma_agents.tools.reset

# Compare runs
git diff run/001..run/002 -- experiments/train.py
```

### What happens during a run

1. **Worktree creation**: `.worktrees/run_XXX/` created from `main`
2. **Reset**: `train.py` reset to `baseline_train.py` (in worktree)
3. **Baseline commit**: Initial state committed to `run/XXX` branch
4. **Iterations**: Agents propose/implement/evaluate (all in worktree)
   - Improvement → commit to branch
   - No improvement → revert changes
5. **Summary**: Final RMSE and git log displayed
6. **Logs**: Saved to `experiments/run_XXX/`

### Promoting a run

When a run produces good results, promote it to main:

```bash
uv run python -m pharma_agents.promote 1
```

This will:
1. Checkout `main`
2. Merge `run/001` into `main`
3. Copy `train.py` → `baseline_train.py` (new baseline)
4. Commit the updated baseline

Future runs will start from this new baseline.

## Project Structure

```
pharma-catalyst/
├── src/pharma_agents/           # Library code (generic, reusable)
│   ├── crew.py                  # CrewAI crew definition
│   ├── memory.py                # Persistent agent memory
│   ├── agents.yaml              # Agent configs (roles, goals, backstories)
│   ├── tasks.yaml               # Task definitions
│   ├── main.py                  # Entry point - runs iterations
│   ├── promote.py               # Promote run branch to main
│   ├── discard.py               # Discard cancelled/stuck runs
│   ├── tools/
│   │   ├── custom_tools.py      # WriteTrainPyTool, CodeCheckTool
│   │   ├── evaluate.py          # Fixed evaluation harness
│   │   └── reset.py             # Reset train.py to baseline
│   └── data/
│       ├── fetch.py             # Download dataset
│       └── esol.csv             # ESOL dataset (1,128 molecules)
├── experiments/                 # Shared experiment data (stays in main)
│   ├── baseline.json            # Baseline config (metric, score, direction)
│   ├── baseline_train.py        # BASELINE code (never modified by agents)
│   ├── memory.json              # Persistent agent memory (shared)
│   ├── run_001/                 # Logs for run 1
│   └── run_002/                 # Logs for run 2
├── .worktrees/                  # Isolated run environments (gitignored)
│   ├── run_001/                 # Worktree for run 1
│   └── run_002/                 # Worktree for run 2
├── .env                         # API keys (GOOGLE_API_KEY, LLM_MODEL)
├── pyproject.toml
└── README.md
```

This separation allows swapping experiments (different models, metrics, datasets) by replacing the `experiments/` folder contents.

## Agent Memory

Agents learn across runs via persistent memory (`experiments/memory.json`).

```json
{
  "runs": {
    "1": {
      "run_id": 1,
      "start_time": "2026-03-13T11:52:00",
      "experiments": [
        {
          "iteration": 1,
          "hypothesis": "Combine HistGradientBoosting with physicochemical descriptors",
          "reasoning": "Building on successful approaches from memory",
          "result": "success",
          "rmse_before": 1.3175,
          "rmse_after": 0.7061,
          "improvement_pct": 46.4,
          "insight": "Memory-informed decisions outperform blind exploration"
        }
      ],
      "best_rmse": 0.6532,
      "consecutive_failures": 0,
      "conclusion": "LOCAL_OPTIMUM",
      "conclusion_detail": "HistGradientBoosting + descriptors maximized. Try different architectures."
    }
  },
  "global_best_rmse": 0.6532,
  "key_learnings": [
    "Combine fingerprints (local) with descriptors (global)",
    "HistGradientBoosting outperforms RandomForest with good features"
  ]
}
```

**What gets remembered:**
- **What worked** - with reasoning and improvement percentage
- **What failed** - with reasoning and WHY it failed
- **Key learnings** - actionable insights for future runs

The Hypothesis Agent sees this context and can:
- Build on successful approaches
- Avoid repeating failures
- Combine winning strategies

### Run Conclusions & Notes for Future Researchers

Each run ends with a conclusion that guides future agents:

| Conclusion | Meaning | Guidance |
|------------|---------|----------|
| `LOCAL_OPTIMUM` | Reached diminishing returns | Try different architectures, GNN features, ensembles |
| `PROGRESS_CONTINUING` | Still improving when stopped | Continue same direction, more iterations |
| `STUCK` | Multiple consecutive failures | Exploration mode required |

Future runs see **"Notes from Previous Researchers"**:
```
### Notes from Previous Researchers
- **Run 0** [PROGRESS_CONTINUING]: 18.4% improvement with descriptors. More iterations could help.
- **Run 1** [LOCAL_OPTIMUM]: 50.4% improvement. HistGradientBoosting + descriptors maximized.
  Future runs should try: different model architectures, GNN-based features, or ensembles.
```

### Stuck Detection & Exploration Mode

The system detects when agents are stuck and triggers exploration mode:

**Two-level detection:**
1. **Per-run stuck**: 3+ consecutive failures in the current run
2. **Global stagnation**: No improvement in the last 5 experiments across all runs

**When stuck, agents see:**
```
### EXPLORATION MODE REQUIRED
**GLOBAL STAGNATION: No improvement in recent experiments.
Current approaches have been exhausted.**

You MUST propose something RADICALLY DIFFERENT from what's been tried.
Look at 'What Failed' and 'What Worked' - then think outside that box.
Consider: different model families, different feature representations,
simplification instead of complexity, or entirely new scientific angles.
```

This prevents agents from getting trapped in local optima and encourages creative exploration of the solution space.

## Hallucinations vs Truth

LLMs can hallucinate numbers in their reports. We handle this:

| Component | Can Hallucinate? | Source of Truth |
|-----------|------------------|-----------------|
| Agent proposals | Yes (harmless) | Just text suggestions |
| Agent RMSE reports | Yes (dangerous) | **Python evaluation** |
| Improvement decisions | No | `run_training()` in Python |
| Memory records | No | Actual measured RMSE values |

**The memory is the pudding** - agents may claim "WORSE" when it's actually better, but:
1. Python runs the actual training
2. Python compares actual RMSE values
3. Python decides commit vs revert
4. Memory records what actually happened

Agent text output is for observability, not decisions.

## Safety Features

- **Baseline preservation**: `baseline_train.py` is never modified
- **Branch isolation**: Each run on its own branch
- **Fixed evaluation**: Python evaluation harness (no LLM involvement)
- **Code validation**: ML Engineer runs ruff+pyright before finishing
- **Experiment logging**: Every run logged with timestamps
- **Timeout**: Training must complete in 60 seconds
- **Revert on failure**: Bad changes don't persist
- **Git audit trail**: Every improvement is a commit
- **Memory persistence**: Learnings survive across runs

## Example Results

**Run 001** (5 iterations with memory):
- Baseline RMSE: 1.3175
- Final RMSE: 0.6532
- Improvement: **50.4%**

**Iteration progression:**
1. Combined HistGradientBoosting + physicochemical descriptors → 46% improvement
2. Fine-tuned learning rate → +0.2%
3. Added aromatic proportion feature → +1.5%
4. Tried regularization (failed, reverted)
5. Optimized feature combination → +5.9%

**Key insight:** The agent learned from memory that MolLogP and MolWt are critical (from seeded experiments), then strategically combined HistGradientBoosting with all successful descriptors. Memory-informed decisions outperformed blind exploration.

## Why This Architecture

**The molecule is a prop. The agentic capability is the show.**

This demonstrates:
- Multi-agent orchestration (CrewAI)
- Autonomous iteration on ML code
- Clear metrics and feedback loops
- Production patterns (logging, git, reproducibility)
- Pharma domain awareness (ADMET, solubility, Delaney paper)

The same architecture scales to larger ADMET tasks with GPU.

## Framework Choice

**Why CrewAI over LangGraph?**

> LangGraph is powerful but adds complexity I didn't need for this pattern. My use case is a sequential crew with clear roles: Researcher proposes, Engineer implements, QA evaluates. CrewAI's role-based model mapped directly to that design. I optimized for clarity and iteration speed over flexibility I wouldn't use.

## License

MIT
