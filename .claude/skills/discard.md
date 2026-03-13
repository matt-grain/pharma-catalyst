# Discard Run

Remove a cancelled, stuck, or failed run completely.

## Usage

```bash
/discard <run_number> [experiment] [--keep-logs]
```

## Arguments

- `run_number`: Required. The run number to discard (e.g., 1, 2, 3)
- `experiment`: Optional. Experiment name (default: bbbp)
- `--keep-logs`: Optional. Keep log files instead of deleting them

## Examples

```bash
# Discard BBBP run 1
/discard 1 bbbp

# Discard solubility run 3, keep logs
/discard 3 solubility --keep-logs
```

## What It Does

1. Removes the git worktree (`.worktrees/<experiment>/run_XXX/`)
2. Removes the run entry from memory.json
3. Deletes the git branch (`run/<experiment>/XXX`)
4. Optionally deletes log files

## Command

```bash
uv run python -m pharma_agents.discard {{run_number}} --experiment {{experiment}}
```
