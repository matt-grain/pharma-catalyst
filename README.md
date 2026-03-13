# pharma-agents

**Autonomous multi-agent system for molecular property prediction.**

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) - but for drug discovery ML.

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
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
   +-------------+    +-------------+    +-------------+
   | Hypothesis  |    |   Model     |    |  Evaluator  |
   |   Agent     |    |   Agent     |    |   Agent     |
   +-------------+    +-------------+    +-------------+
   Research Scientist   ML Engineer      QA Scientist
```

**Sequential process:**
1. Hypothesis Agent proposes an improvement
2. Model Agent implements it in `train.py`
3. Evaluator Agent runs training, reports RMSE
4. If improved → commit. If worse → revert. Repeat.

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

Each run creates its own branch, keeping `main` clean at baseline.

```
main (baseline RMSE: 1.3175)
  │
  ├── run/001 ──► agents iterate, commits improvements
  │     │
  │     └── [promote] ──► merges to main, updates baseline
  │
  ├── run/002 ──► fresh start from baseline (or new baseline)
  │
  └── run/003 ──► ...
```

### Commands

```bash
# Start a new run (creates branch run/XXX, resets to baseline)
uv run python -m pharma_agents.main

# Promote a successful run to main (makes it the new baseline)
uv run python -m pharma_agents.promote 1

# Reset train.py to baseline manually
uv run python -m pharma_agents.tools.reset

# Compare runs
git diff run/001..run/002 -- src/pharma_agents/tools/train.py
```

### What happens during a run

1. **Branch creation**: `run/XXX` branch created from `main`
2. **Reset**: `train.py` reset to `baseline_train.py`
3. **Baseline commit**: Initial state committed
4. **Iterations**: Agents propose/implement/evaluate
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
pharma-agents/
├── src/pharma_agents/
│   ├── crew.py              # CrewAI crew definition
│   ├── agents.yaml          # Agent configs (roles, goals, backstories)
│   ├── tasks.yaml           # Task definitions
│   ├── main.py              # Entry point - runs iterations
│   ├── promote.py           # Promote run branch to main
│   ├── tools/
│   │   ├── baseline_train.py  # BASELINE (never modified by agents)
│   │   ├── train.py           # Working copy (agents modify this)
│   │   ├── evaluate.py        # Fixed evaluation harness
│   │   └── reset.py           # Reset train.py to baseline
│   └── data/
│       ├── fetch.py         # Download ESOL dataset
│       └── esol.csv         # Dataset (1,128 molecules)
├── experiments/
│   ├── run_001/             # Logs for run 1
│   │   └── *.log
│   ├── run_002/             # Logs for run 2
│   └── ...
├── .env                     # API keys (GOOGLE_API_KEY)
├── pyproject.toml
└── README.md
```

## Safety Features

- **Baseline preservation**: `baseline_train.py` is never modified
- **Branch isolation**: Each run on its own branch
- **Fixed evaluation**: `evaluate.py` is never modified (prevents gaming)
- **Experiment logging**: Every run logged with timestamps
- **Timeout**: Training must complete in 60 seconds
- **Revert on failure**: Bad changes don't persist
- **Git audit trail**: Every improvement is a commit

## Example Results

**Run 001** (first iteration):
- Baseline RMSE: 1.3175
- Final RMSE: 1.0747
- Improvement: **18.43%**

The agent proposed adding physicochemical descriptors (MolLogP, MolWt, NumRotatableBonds, HeavyAtomCount) to the Morgan fingerprints, correctly identifying that solubility depends on global molecular properties not captured by local structural fingerprints alone.

See `experiments/run_001/run_001.md` for detailed analysis.

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
