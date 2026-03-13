---
description: Promote a successful run as the new baseline
---

# Promote Run

Promote a successful run as the new baseline.

## Arguments

- `run_number`: Required. The run number to promote
- `--experiment`, `-e`: Experiment name (default: bbbp)

## Examples

```
/promote 2 -e bbbp
/promote 4 -e solubility
```

## What It Does

1. Merges the run branch into main
2. Copies the improved train.py to baseline_train.py
3. Updates baseline.json with the new score
4. Commits the updated baseline

$ARGUMENTS

```bash
uv run python -m pharma_agents.promote $ARGUMENTS
```
