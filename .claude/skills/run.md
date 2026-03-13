# Run Experiment

Start an autonomous ML optimization run for a molecular property prediction task.

## Usage

```bash
/run [experiment] [iterations]
```

## Arguments

- `experiment`: Which experiment to run (`bbbp` or `solubility`, default: bbbp)
- `iterations`: Number of improvement iterations (default: 5)

## Examples

```bash
# Run BBBP with 5 iterations
/run bbbp 5

# Run solubility with 3 iterations
/run solubility 3

# Quick test with 1 iteration
/run bbbp 1
```

## What It Does

1. Creates an isolated git worktree for the run
2. Resets train.py to baseline
3. Runs N iterations of the agent crew (Hypothesis → Implement → Evaluate)
4. Commits improvements, reverts failures
5. Saves learnings to memory.json

## Command

```bash
PHARMA_EXPERIMENT={{experiment}} MAX_ITERATIONS={{iterations}} uv run python -m pharma_agents.main
```
