---
description: Generate HTML report with charts for a run
---

# Generate Report

Generate an interactive HTML report with Plotly charts from memory.json.

## Arguments

- `--experiment`, `-e`: Experiment name (default: bbbp)
- `--run`, `-r`: Specific run ID (default: all runs)
- `--open`: Open in browser after generation

## Examples

```
/report -e bbbp
/report -e bbbp -r 2 --open
/report -e solubility --open
```

## What It Does

Generates `report.html` in each run folder with:
- Score progression chart
- Before/After comparison
- Success rate pie chart
- Improvement per iteration
- Experiment details table

$ARGUMENTS

```bash
uv run python -m pharma_agents.report $ARGUMENTS
```
