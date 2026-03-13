# Promote Run

Promote a successful run as the new baseline.

## Usage

```bash
/promote <run_number> [experiment]
```

## Arguments

- `run_number`: Required. The run number to promote
- `experiment`: Optional. Experiment name (default: bbbp)

## Examples

```bash
# Promote BBBP run 2 as new baseline
/promote 2 bbbp

# Promote solubility run 4
/promote 4 solubility
```

## What It Does

1. Merges the run branch into main
2. Copies the improved train.py to baseline_train.py
3. Updates baseline.json with the new score
4. Commits the updated baseline

## Command

```bash
PHARMA_EXPERIMENT={{experiment}} uv run python -m pharma_agents.promote {{run_number}}
```
