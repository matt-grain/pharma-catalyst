---
description: Start an autonomous ML optimization run
---

# Run Experiment

Start an autonomous ML optimization run for a molecular property prediction task.

## Arguments

- `experiment`: Which experiment to run (`bbbp` or `solubility`, default: bbbp)
- `iterations`: Number of improvement iterations (default: 5)

## Examples

```
/run bbbp 5
/run solubility 3
```

## What It Does

1. Creates an isolated git worktree for the run
2. Resets train.py to baseline
3. Runs N iterations of the agent crew (Hypothesis → Implement → Evaluate)
4. Commits improvements, reverts failures
5. Saves learnings to memory.json

$ARGUMENTS

```bash
uv run python -m pharma_agents.main $ARGUMENTS
```
