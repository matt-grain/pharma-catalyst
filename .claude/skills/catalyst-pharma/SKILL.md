---
name: catalyst-pharma
description: Expert in pharma-agents project - autonomous multi-agent system for molecular ML optimization. Use when working on experiments, understanding the crew architecture, or debugging agent runs.
---

# Pharma Catalyst Expert

You are an expert in the **pharma-agents** project - an autonomous multi-agent system for molecular property prediction optimization.

## Project Overview

This system uses CrewAI to run autonomous ML optimization loops:
1. **Hypothesis Agent** - Proposes improvements based on past experiments
2. **Model Agent** - Implements changes to train.py
3. **Evaluator Agent** - Runs training and reports results

## Key Architecture

### Experiment Structure
```
experiments/
├── bbbp/                    # Blood-Brain Barrier Penetration (classification)
│   ├── baseline.json        # Score, metric, direction config
│   ├── baseline_train.py    # Reference baseline (never modified)
│   ├── train.py             # Working copy (agents modify this)
│   └── memory.json          # Experiment learnings
├── solubility/              # ESOL solubility (regression)
│   └── ...
```

### Metric Configuration (baseline.json)
```json
{
  "score": 0.8951,
  "metric": "ROC_AUC",
  "direction": "higher_is_better"  // or "lower_is_better" for RMSE
}
```

### Git Workflow
- Each run creates an isolated worktree: `.worktrees/<experiment>/run_XXX/`
- Branch naming: `run/<experiment>/<number>` (e.g., `run/bbbp/001`)
- Successful improvements are committed; failures are reverted

## Key Files

| File | Purpose |
|------|---------|
| `src/pharma_agents/main.py` | Entry point, iteration loop |
| `src/pharma_agents/crew.py` | Agent definitions |
| `src/pharma_agents/memory.py` | Persistent learning, metric helpers |
| `src/pharma_agents/tools/custom_tools.py` | WriteTrainPy, RunTrainPy, CodeCheck |
| `experiments/<name>/baseline.json` | Experiment config |

## Commands

- `/run <experiment> <iterations>` - Start an optimization run
- `/discard <run> -e <experiment>` - Remove a failed/stuck run
- `/promote <run> -e <experiment>` - Promote successful run as new baseline

## Important Patterns

### Direction-Aware Comparison
```python
from pharma_agents.memory import is_better, compute_improvement_pct

if is_better(new_score, old_score):  # Handles both higher/lower is better
    print(f"Improved by {compute_improvement_pct(old_score, new_score):.1f}%")
```

### Environment Variables
- `PHARMA_EXPERIMENT` - Which experiment to run (bbbp, solubility)
- `PHARMA_EXPERIMENTS_DIR` - Override experiments path (for worktrees)
- `MAX_ITERATIONS` - Number of optimization iterations

## Debugging Tips

1. **FileReadTool errors** - Check that train.py exists in the worktree
2. **Metric hardcoding** - Use `get_metric_name()`, `is_better()` not raw comparisons
3. **Run numbering** - Per-experiment, check branches with `git branch -a | grep run/<exp>/`
