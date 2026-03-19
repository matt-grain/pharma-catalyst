## Agent Memory

Agents learn across runs via persistent memory (`experiments/memory.json`).

```json
{
  "runs": {
    "1": {
      "run_id": 1,
      "start_time": "2026-03-13T11:52:00",
      "experiments": [
        {
          "iteration": 1,
          "hypothesis": "Combine HistGradientBoosting with physicochemical descriptors",
          "reasoning": "Building on successful approaches from memory",
          "result": "success",
          "rmse_before": 1.3175,
          "rmse_after": 0.7061,
          "improvement_pct": 46.4,
          "insight": "Memory-informed decisions outperform blind exploration"
        }
      ],
      "best_score": 0.6532,
      "consecutive_failures": 0,
      "conclusion": "LOCAL_OPTIMUM",
      "conclusion_detail": "HistGradientBoosting + descriptors maximized. Try different architectures."
    }
  },
  "global_best_score": 0.9477,
  "key_learnings": [],
  "last_updated": "2026-03-18T15:06:58.363430"
}
```

**What gets remembered:**
- **What worked** - with reasoning and improvement percentage
- **What failed** - with reasoning and WHY it failed
- **Key learnings** - actionable insights for future runs

The Hypothesis Agent sees this context and can:
- Build on successful approaches
- Avoid repeating failures
- Combine winning strategies

### Run Conclusions & Notes for Future Researchers

Each run ends with a conclusion that guides future agents:

| Conclusion | Meaning | Guidance |
|------------|---------|----------|
| `LOCAL_OPTIMUM` | Reached diminishing returns | Try different architectures, GNN features, ensembles |
| `PROGRESS_CONTINUING` | Still improving when stopped | Continue same direction, more iterations |
| `STUCK` | Multiple consecutive failures | Exploration mode required |

Future runs see **"Notes from Previous Researchers"**:
```
### Notes from Previous Researchers
- **Run 0** [PROGRESS_CONTINUING]: 18.4% improvement with descriptors. More iterations could help.
- **Run 1** [LOCAL_OPTIMUM]: 50.4% improvement. HistGradientBoosting + descriptors maximized.
  Future runs should try: different model architectures, GNN-based features, or ensembles.
```
