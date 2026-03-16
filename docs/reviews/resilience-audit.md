# Resilience Audit: pharma-agents

**Date:** 2026-03-14
**Scope:** Full "what can go wrong" analysis of autonomous agent execution
**Context:** CrewAI multi-agent system where LLM-powered agents write and execute Python code autonomously. Real failure already observed: agent got stuck when generated `train.py` imported `lightgbm` which was not installed.

---

## Summary of Findings

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 4 | Can cause silent data corruption, infinite loops, or complete run failure |
| MAJOR | 8 | Will cause failures under realistic conditions that require manual intervention |
| MINOR | 5 | Reduce robustness but unlikely to cause outright failure |

---

## CRITICAL Findings

### C1. No `max_execution_time` on Any Agent -- Runaway Agent Risk

**Files:** `src/pharma_agents/crew.py` lines 94-151
**Problem:** None of the four agents have `max_execution_time` set. CrewAI's `max_iter` limits the number of tool-calling iterations, but a single LLM call that hangs (API timeout, rate limit backoff, network issue) will block forever. The `model_agent` has `max_iter=40` -- if each iteration involves a slow LLM call, a single crew run could take hours with no timeout.

The `hypothesis_agent` and `evaluator_agent` have no `max_iter` set at all, inheriting the default of 20, but still have no wall-clock timeout.

**Real failure scenario:** Google Gemini API returns 429 rate limit errors. CrewAI's internal retry logic retries indefinitely with exponential backoff. The agent hangs for hours consuming no tokens but blocking the entire sequential pipeline.

**Fix:**
```python
@agent
def model_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["model_agent"],
        llm=get_llm(temperature=0.3),
        tools=[...],
        max_iter=40,
        max_execution_time=600,  # 10 minutes hard wall-clock limit
        verbose=True,
    )

@agent
def evaluator_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["evaluator_agent"],
        llm=get_llm(),
        tools=[ReadTrainPyTool(), RunTrainPyTool()],
        max_iter=10,    # Evaluator should not need many iterations
        max_execution_time=300,  # 5 minutes
        verbose=True,
    )
```

Recommended values:
- `archivist_agent`: `max_execution_time=600` (network I/O, 10 min)
- `hypothesis_agent`: `max_execution_time=300` (5 min, mostly LLM reasoning)
- `model_agent`: `max_execution_time=600` (10 min, code fix cycles)
- `evaluator_agent`: `max_execution_time=300` (5 min, includes 60s training timeout)

---

### C2. ClassVar Counters Never Reset Between Runs -- State Leaks Across Iterations

**Files:**
- `src/pharma_agents/tools/arxiv.py` lines 26-27: `_papers_fetched`, `_last_fetch`
- `src/pharma_agents/tools/arxiv.py` lines 137-138: `_searches_done`, `_last_search`
- `src/pharma_agents/tools/literature.py` line 298: `_calls_done`
- `src/pharma_agents/tools/training.py` line 163: `_packages_installed`
- `src/pharma_agents/tools/skills.py` line 27: `_skills_loaded`

**Problem:** All rate-limiting counters use `ClassVar` (class-level state) and are never reset. The `main.py` `run()` function creates a new `PharmaAgentsCrew()` per iteration call but this does NOT reset `ClassVar` state. After the first run exhausts the archivist's 10 papers and 8 searches, ALL subsequent runs in the same process will immediately hit the limit.

`InstallPackageTool._packages_installed` is a `ClassVar[list[str]]` that accumulates across ALL iterations. After 3 packages are installed, no more can be installed for the entire process lifetime -- even across separate crew runs.

**Real failure scenario:** User runs `run(iterations=5)`. Iteration 1 archivist does 8 searches and fetches 10 papers. Iteration 2 needs fresh literature (exploration mode triggered) but every search/fetch call immediately returns "limit reached". The archivist task silently produces no results.

**Fix:** Add a reset mechanism and call it before each crew run.

```python
# In each tool class, add a classmethod:
@classmethod
def reset_counters(cls):
    cls._papers_fetched = 0
    cls._last_fetch = 0.0

# In main.py run(), before each iteration:
from .tools.arxiv import AlphaxivTool, ArxivSearchTool
from .tools.literature import FetchMorePapersTool
from .tools.training import InstallPackageTool
from .tools.skills import SkillLoaderTool

AlphaxivTool.reset_counters()
ArxivSearchTool.reset_counters()
FetchMorePapersTool.reset_counters()
InstallPackageTool.reset_counters()
SkillLoaderTool.reset_counters()
```

---

### C3. `RunTrainPyTool` 60-Second Timeout Is Insufficient for ML Training

**Files:**
- `src/pharma_agents/tools/training.py` lines 130-134
- `src/pharma_agents/tools/evaluate.py` line 57

**Problem:** Both `RunTrainPyTool._run()` and `run_training()` use a 60-second timeout. The agents are encouraged to install and use packages like `xgboost`, `catboost`, `torch`, `tensorflow`, and `deepchem` (all in the `ALLOWED_PACKAGES` whitelist). Training a neural network or gradient boosting model with cross-validation on ~2000 molecules can easily exceed 60 seconds, especially:
- First-time `import torch` can take 10-15 seconds
- `catboost` with default iterations (1000) on 2000 samples: ~30-60s
- Any hyperparameter grid search: minutes
- `deepchem` graph neural networks: minutes

The agent has no way to know about the timeout constraint. The task description says "Complete in under 60 seconds" only in the baseline_train.py docstring, not in the task prompt.

**Real failure scenario:** Agent proposes switching to XGBoost with 500 estimators. Code passes `code_check`. Training starts, takes 90 seconds, gets killed. Agent sees "Error: Training timed out (>60s)" but has no context for WHY or how to fix it. It may retry the same approach, burning iterations.

**Fix:**
1. Increase timeout to 180 seconds in both locations
2. Add timeout constraint to the `implement_task` description in `tasks.yaml`
3. Add timeout information to the error message

```python
# training.py RunTrainPyTool
timeout=180,  # Allow up to 3 minutes for complex models

# evaluate.py run_training
def run_training(timeout_seconds: int = 180) -> ExperimentResult:
```

```yaml
# tasks.yaml implement_task
Rules:
  - Training MUST complete within 180 seconds (3 minutes)
  - Avoid grid search with many combinations
  - Keep model complexity reasonable (e.g., n_estimators <= 500)
```

---

### C4. No Crew-Level Error Handling for LLM API Failures

**File:** `src/pharma_agents/main.py` lines 449-463

**Problem:** The crew kickoff is wrapped in a bare `except Exception` that catches everything, reverts git, and `continue`s to the next iteration. This is correct for catching errors, but there is no distinction between:
- Transient LLM API errors (429, 503) -- should retry after a delay
- Permanent configuration errors (invalid API key, wrong model name) -- should abort immediately
- Agent logic errors (max iterations reached) -- should continue to next iteration
- Budget/quota exhaustion -- should abort the entire run

Currently, if the API key is invalid, the system will loop through all N iterations, each failing immediately, wasting time and producing no useful output.

**Fix:**
```python
except Exception as e:
    error_msg = str(e)
    logger.error(f"Crew error: {e}")
    git_revert_changes(worktree_path)

    # Distinguish transient vs permanent failures
    if any(term in error_msg.lower() for term in [
        "api key", "authentication", "unauthorized", "403",
        "invalid model", "model not found",
    ]):
        logger.error("PERMANENT ERROR - aborting all iterations")
        break
    elif any(term in error_msg.lower() for term in [
        "rate limit", "429", "quota", "resource exhausted",
    ]):
        logger.warning("Rate limit hit - waiting 60s before retry")
        import time
        time.sleep(60)
    # else: transient error, continue to next iteration
    continue
```

---

## MAJOR Findings

### M1. `promote()` Function Has Multiple Unhandled Failure Modes

**File:** `src/pharma_agents/main.py` lines 607-679

**Problems:**
1. **Line 644:** `promote()` references `experiments_dir / "train.py"` and `experiments_dir / "baseline_train.py"` but `experiments_dir` is set to `project_root / "experiments"` WITHOUT the experiment name. After the merge, the experiment-specific subdirectory is not targeted. This will copy the wrong file or fail with FileNotFoundError.

2. **Line 655:** `baseline_json.read_text()` will crash with `FileNotFoundError` if baseline.json does not exist at the expected path (which it won't, since the path is wrong -- see above).

3. **Line 651:** `run_training()` is called after merge, but `PHARMA_EXPERIMENTS_DIR` environment variable may still point to a now-deleted worktree.

4. **Line 614:** Branch name is `f"run/{run_number:03d}"` but `git_create_worktree()` creates branches as `f"run/{experiment}/{run_number:03d}"`. The promote function will always fail to find the branch.

**Fix:** The `promote()` function needs the experiment name, and paths need to be corrected:

```python
def promote(run_number: int, experiment: str | None = None) -> None:
    experiment = experiment or get_experiment_name()
    branch_name = f"run/{experiment}/{run_number:03d}"
    experiments_dir = project_root / "experiments" / experiment
    # ... rest of function using experiment-aware paths
```

---

### M2. `WriteTrainPyTool` Has No Content Validation -- Agent Can Write Empty or Non-Python Files

**File:** `src/pharma_agents/tools/training.py` lines 53-70

**Problem:** The `WriteTrainPyTool` writes whatever string the agent provides directly to `train.py` with zero validation. Known failure modes:
- Agent writes an empty string (deletes file content)
- Agent writes partial code (truncated by context window)
- Agent writes markdown-wrapped code (```python ... ```)
- Agent writes code that imports `os.system("rm -rf /")` or similar

The `CodeCheckTool` catches syntax errors AFTER the write, but between the write and the check, the file is in a potentially broken state. If the process crashes between write and check, the experiment is left with garbage.

**Fix:**
```python
def _run(self, content: str) -> str:
    content = content.strip()

    # Strip markdown code fences (common LLM mistake)
    if content.startswith("```python"):
        content = content[len("```python"):].strip()
    if content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    # Basic validation
    if len(content) < 50:
        return "Error: Content too short. Provide the COMPLETE train.py file."
    if "def train" not in content:
        return "Error: train.py must contain a 'def train' function."
    if "import " not in content:
        return "Error: train.py must have import statements."

    # Check for dangerous patterns
    dangerous = ["os.system", "subprocess.run", "subprocess.call",
                 "shutil.rmtree", "__import__", "eval(", "exec("]
    for pattern in dangerous:
        if pattern in content:
            return f"Error: Dangerous pattern '{pattern}' detected. Not allowed."

    train_path = get_experiments_dir() / "train.py"
    # ... write file
```

---

### M3. `LiteratureStoreTool` Instantiates `TextEmbedding` Model on Every Call

**File:** `src/pharma_agents/tools/literature.py` lines 119-120, 153

**Problem:** Every call to `store_paper` creates a new `TextEmbedding("BAAI/bge-small-en-v1.5")` instance. The `LiteratureQueryTool` correctly caches the model (line 222-231), but `LiteratureStoreTool` does not. Loading the embedding model takes 2-5 seconds. If the archivist stores 10 papers, that is 20-50 seconds of unnecessary model loading.

Worse, multiple `TextEmbedding` instances may consume significant memory (each loads the model into RAM).

**Fix:** Share the cached model from `LiteratureQueryTool`:

```python
class LiteratureStoreTool(BaseTool):
    _model: ClassVar = None

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from fastembed import TextEmbedding
            cls._model = TextEmbedding("BAAI/bge-small-en-v1.5")
        return cls._model

    def _run(self, paper_input: str) -> str:
        model = self._get_model()
        # ... use model
```

---

### M4. Metric Parsing Is Fragile -- Agent Can Fool the Evaluation Harness

**File:** `src/pharma_agents/tools/evaluate.py` lines 116-125

**Problem:** The metric extraction logic looks for any line containing the metric name with a colon:

```python
metric_line = [line for line in output.split("\n") if metric in line]
score = float(metric_line[-1].split(":")[-1].strip())
```

If the agent's `train.py` prints debug output like `"Loading ROC_AUC model: 0.99"` before the actual result, the parser picks up the LAST line containing the metric name. If the agent prints the metric string multiple times, the parser takes the last one -- which could be a training score, not validation score.

More critically: if the agent's code prints `"ROC_AUC: 0.9999"` as a hardcoded string (gaming the metric), the evaluation harness accepts it as truth.

**Real failure scenario:** Agent writes `print(f"ROC_AUC: {0.99}")` as a debug line before the actual evaluation. The harness picks up 0.99 instead of the real score.

**Fix:** Use a structured output protocol:

```python
# In evaluate.py, look for a specific marker:
RESULT_MARKER = "###RESULT###"

# Parse only lines with the marker
for line in output.split("\n"):
    if line.startswith(RESULT_MARKER):
        score = float(line.split(":")[-1].strip())
        break

# In baseline_train.py and task prompt, require:
# print(f"###RESULT### ROC_AUC: {score:.4f}")
```

Alternatively, write the score to a JSON file and read that instead of parsing stdout.

---

### M5. `git_reset_train_to_baseline` Crashes If Baseline File Is Missing

**File:** `src/pharma_agents/main.py` lines 177-187

**Problem:** `shutil.copy(baseline, train)` will raise `FileNotFoundError` if `baseline_train.py` does not exist at the expected path. This function is called at the start of every run (line 366), so a missing baseline file immediately crashes the entire run with no helpful error message.

The `validate_experiment()` function checks for `baseline.json` but NOT for `baseline_train.py`.

**Fix:**
```python
def git_reset_train_to_baseline(repo_path: Path) -> None:
    experiment_name = get_experiment_name()
    experiments_dir = repo_path / "experiments" / experiment_name
    baseline = experiments_dir / "baseline_train.py"
    train = experiments_dir / "train.py"

    if not baseline.exists():
        raise FileNotFoundError(
            f"Baseline training script not found: {baseline}\n"
            f"Each experiment needs a baseline_train.py file."
        )
    shutil.copy(baseline, train)
```

Also update `validate_experiment()` to check for `baseline_train.py`:
```python
def validate_experiment(experiment: str | None = None) -> None:
    # ... existing checks ...
    baseline_train = exp_dir / "baseline_train.py"
    if not baseline_train.exists():
        raise SystemExit(
            f"Experiment '{exp_name}' is missing baseline_train.py.\n"
            f"Create experiments/{exp_name}/baseline_train.py with the baseline model."
        )
```

---

### M6. `run_training()` Is Called Twice Per Iteration -- Redundant and Inconsistent

**File:** `src/pharma_agents/main.py` lines 457 and 467

**Problem:** The evaluator agent already runs `train.py` via `RunTrainPyTool` (line 457, inside the crew). Then `main.py` calls `run_training()` again independently (line 467) to get the "official" score. These two executions can produce different results because:
1. The agent may have modified `train.py` between the tool call and the end of the crew run (e.g., if the model_agent delegation re-triggers)
2. Random seeds may produce slightly different results
3. The agent's `RunTrainPyTool` has a 60s timeout but `run_training()` also has 60s -- if training takes 55s, the second call may timeout while the first succeeded

This creates confusion: the agent may report "ROC_AUC: 0.92 - KEEP" but main.py's evaluation says 0.88 and reverts.

**Fix:** Trust only the main.py evaluation (which is the correct approach -- Python is source of truth). But add a comment explaining this, and consider removing the `RunTrainPyTool` from the evaluator agent entirely, having Python handle evaluation:

```python
# Option A: Remove RunTrainPyTool from evaluator, run evaluation in Python only
# Option B: Keep both but document the discrepancy risk

# If keeping both, at minimum log the discrepancy:
if eval_result.success and eval_result.rmse is not None:
    logger.info(f"Python evaluation: {metric} = {eval_result.rmse:.4f}")
    logger.info("(This is the authoritative score, not the agent's report)")
```

---

### M7. No `max_rpm` Set on Any Agent or Crew -- API Rate Limit Vulnerability

**Files:** `src/pharma_agents/crew.py` lines 94-203

**Problem:** Neither individual agents nor the crew have `max_rpm` configured. The `model_agent` with `max_iter=40` can make rapid-fire LLM calls during code fix cycles, potentially hitting API rate limits. Google Gemini's free tier has strict RPM limits (typically 15-60 RPM depending on the model).

When rate limits are hit without `max_rpm`, CrewAI may not handle the 429 gracefully -- it depends on the LiteLLM retry configuration, which is not explicitly set here.

**Fix:**
```python
@crew
def crew(self) -> Crew:
    return Crew(
        agents=[...],
        tasks=[...],
        process=Process.sequential,
        verbose=True,
        max_rpm=30,  # Conservative limit for Gemini API
    )
```

---

### M8. `compute_improvement_pct` Division by Zero When Baseline Score Is Zero

**File:** `src/pharma_agents/memory.py` lines 89-94

**Problem:**
```python
def compute_improvement_pct(old_score: float, new_score: float) -> float:
    direction = get_metric_direction()
    if direction == "higher_is_better":
        return ((new_score - old_score) / old_score) * 100
    return ((old_score - new_score) / old_score) * 100
```

If `old_score` is 0 (which could happen for accuracy-like metrics at initialization), this raises `ZeroDivisionError`.

Similarly in `evaluate.py` line 81:
```python
improvement_pct = ((baseline_score - score) / baseline_score) * 100
```

**Fix:**
```python
def compute_improvement_pct(old_score: float, new_score: float) -> float:
    if old_score == 0:
        return 0.0 if new_score == 0 else float('inf')
    # ... rest of function
```

---

## MINOR Findings

### m1. `TeeStream` File Handle Leak on Exception During `__init__`

**File:** `src/pharma_agents/main.py` lines 33-36

**Problem:** If `open(log_file, "a")` succeeds but the `TeeStream` object is never properly used (e.g., exception between construction and context manager entry), the file handle leaks. Also, `__del__` is unreliable for cleanup in CPython.

**Fix:** Use the context manager pattern consistently. The existing `capture_stdout_to_log` handles this correctly in the `finally` block, so this is low risk.

---

### m2. `SkillLoaderTool._skills_loaded` Is a Mutable ClassVar Default

**File:** `src/pharma_agents/tools/skills.py` line 27

**Problem:** `_skills_loaded: ClassVar[list[str]] = []` is a mutable default shared across all instances and never reset (same issue as C2, but lower impact since skills are read-only).

**Fix:** Same as C2 -- add a reset method.

---

### m3. `LLM_MODEL` Default Points to Preview Model

**File:** `src/pharma_agents/crew.py` line 72

**Problem:**
```python
model = os.getenv("LLM_MODEL", "gemini/gemini-3-flash-preview")
```

The default model is a "preview" model that may be deprecated or removed at any time. If the user does not set `LLM_MODEL` in their `.env` file, the system will silently break when Google removes the preview endpoint.

**Fix:** Default to a stable model name, or fail explicitly if no model is configured:

```python
model = os.getenv("LLM_MODEL")
if not model:
    raise EnvironmentError(
        "LLM_MODEL environment variable must be set. "
        "Example: LLM_MODEL=gemini/gemini-2.0-flash"
    )
```

---

### m4. No Validation That `GOOGLE_API_KEY` Is Set Before Starting

**File:** `src/pharma_agents/crew.py` lines 70-77

**Problem:** `os.getenv("GOOGLE_API_KEY")` returns `None` if the key is not set. This `None` is passed to `LoggingLLM(api_key=None)`. The error will only surface when the first LLM call is made -- deep inside a crew run -- producing a confusing error message.

**Fix:** Validate at startup:

```python
def get_llm(temperature: float = 0.7) -> LLM:
    model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY environment variable is required. "
            "Get a key at https://aistudio.google.com/apikey"
        )
    return LoggingLLM(model=model, api_key=api_key, temperature=temperature)
```

---

### m5. `parse_hypothesis_from_log` Only Captures the LAST Proposal

**File:** `src/pharma_agents/main.py` lines 248-297

**Problem:** The function iterates all lines and overwrites `hypothesis`/`reasoning` on every match. It captures the LAST `PROPOSAL:` line in the log. If an earlier agent iteration produced a proposal that was then revised, the log contains both. The function returns the last one -- which is usually correct, but not guaranteed.

This is fragile compared to using the crew's structured output. The `HypothesisOutput` Pydantic model already exists (line 32-38) and is set as `output_pydantic` on the hypothesis_task (line 166), but `main.py` does not use it -- it falls back to regex parsing of log files.

**Fix:** Use the crew result's Pydantic output instead of parsing logs:

```python
# After crew.kickoff():
result = crew.crew().kickoff(inputs=inputs)

# Access structured output from hypothesis task
hypothesis_output = result.tasks_output[0].pydantic  # HypothesisOutput
if hypothesis_output:
    hypothesis = hypothesis_output.proposal
    reasoning = hypothesis_output.reasoning
else:
    # Fallback to log parsing
    hypothesis, reasoning = parse_hypothesis_from_log(log_file)
```

---

## Systemic Risks

### S1. Agent Can Create Infinite Import Loops

The `model_agent` can write arbitrary Python that imports from the project's own modules. If it writes `from pharma_agents.tools import ...` in `train.py`, this creates a circular import or unintended coupling. The `CodeCheckTool` (ruff) will not catch this -- it is a runtime error only.

**Mitigation:** Add to the `implement_task` prompt: "train.py must be self-contained. Never import from pharma_agents."

### S2. Worktree Accumulation

Each run creates a git worktree in `.worktrees/`. These are never cleaned up automatically. Over many runs, this can consume significant disk space (each worktree is a full working copy). The `discard.py` script removes them manually, but there is no automatic cleanup of old worktrees.

**Mitigation:** Add worktree cleanup at the end of each run, or add a max-worktree check at the start.

### S3. Memory.json Grows Unboundedly

`AgentMemory` appends every experiment to `memory.json` and never prunes old data. After hundreds of experiments, the `format_for_prompt()` method generates increasingly long context that can exceed the LLM context window. The method does limit to last 10 successes and 10 failures, but each entry can be multi-line.

**Mitigation:** Add a max-entries parameter to `format_for_prompt()` and consider archiving old runs.

### S4. fastembed Model Download on First Run

`LiteratureStoreTool` and `LiteratureQueryTool` use `TextEmbedding("BAAI/bge-small-en-v1.5")`. On first run, this downloads the model (~50MB). If the network is unavailable or slow, the tool call hangs or fails inside the agent's execution, producing a confusing error.

**Mitigation:** Add a startup check that downloads the model before the crew starts:

```python
# In main.py run(), before crew kickoff:
try:
    from fastembed import TextEmbedding
    TextEmbedding("BAAI/bge-small-en-v1.5")  # Ensures model is cached
except Exception as e:
    logger.warning(f"Could not pre-load embedding model: {e}")
```

---

## Recommended Priority Order

1. **C1** -- Add `max_execution_time` to all agents (prevents hangs)
2. **C2** -- Reset ClassVar counters between iterations (prevents silent tool lockout)
3. **C4** -- Distinguish transient vs permanent API errors (prevents wasted loops)
4. **C3** -- Increase training timeout and communicate it to agents (prevents false failures)
5. **M2** -- Add content validation to WriteTrainPyTool (prevents garbage writes)
6. **M4** -- Fix metric parsing / use structured output markers (prevents score gaming)
7. **M7** -- Add `max_rpm` to crew (prevents rate limit crashes)
8. **m4** -- Validate API key at startup (prevents confusing late errors)
9. **m3** -- Use stable model default (prevents silent breakage)
10. **M1** -- Fix promote() function paths (currently broken)
11. **M5** -- Validate baseline_train.py exists (prevents confusing crash)
12. **M3** -- Cache embedding model in LiteratureStoreTool (performance)
13. **M6** -- Remove redundant double-evaluation (clarity)
14. **M8** -- Handle division by zero in improvement calculation (edge case)

---

## Verification Checklist

After applying fixes, verify each item:

- [ ] Run with expired/invalid API key -- should fail fast with clear message
- [ ] Run 3+ iterations -- ClassVar counters should reset between iterations
- [ ] Agent writes XGBoost with 1000 estimators -- should not timeout prematurely
- [ ] Agent writes empty train.py -- should be rejected by WriteTrainPyTool
- [ ] Agent writes `print("ROC_AUC: 0.99")` as debug -- should not fool evaluator
- [ ] Kill process mid-run, restart -- should recover cleanly
- [ ] Run promote with correct experiment -- should work end to end
- [ ] Memory.json with 100+ experiments -- format_for_prompt should not exceed context
