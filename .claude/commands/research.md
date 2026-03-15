---
description: Run archivist to build/refresh the literature database before a run
---

# Literature Research

Run the Archivist agent standalone to build or refresh the literature database.
Use this before a run so you (or the agents) have recent papers available.

## Arguments

- `--experiment`, `-e`: Experiment name (default: bbbp)

## Examples

```
/research -e bbbp
/research -e clintox
```

## What It Does

1. Runs the Archivist agent (only) for the given experiment
2. Searches arxiv for recent papers on the experiment's property
3. Fetches and stores papers with embeddings in `experiments/<name>/literature/`
4. Reports how many papers were added

No worktree, no training, no model changes. Safe to run anytime.

Browse results in `experiments/<name>/literature/papers/`.

$ARGUMENTS

```bash
uv run python -m pharma_agents.research $ARGUMENTS
```
