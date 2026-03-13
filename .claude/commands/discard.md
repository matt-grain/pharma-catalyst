---
description: Remove a cancelled, stuck, or failed run
---

# Discard Run

Remove a cancelled, stuck, or failed run completely.

## Arguments

- `run_number`: Required. The run number to discard (e.g., 1, 2, 3)
- `--experiment`, `-e`: Experiment name (default: bbbp)
- `--keep-logs`: Keep log files instead of deleting them

## Examples

```
/discard 1 -e bbbp
/discard 3 -e solubility --keep-logs
```

## What It Does

1. Removes the git worktree (`.worktrees/<experiment>/run_XXX/`)
2. Removes the run entry from memory.json
3. Deletes the git branch (`run/<experiment>/XXX`)
4. Optionally deletes log files

$ARGUMENTS

```bash
uv run python -m pharma_agents.discard $ARGUMENTS
```
