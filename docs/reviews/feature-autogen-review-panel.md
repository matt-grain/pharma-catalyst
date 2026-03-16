# Code Review: feature/autogen-review-panel

**Date:** 2026-03-15
**Reviewer:** Claude Code (automated)
**Branch:** `feature/autogen-review-panel` vs `main`
**Scope:** Python files only — 9 files changed, ~3 275 lines reviewed

---

## Files Reviewed

| File | Lines | Status |
|---|---|---|
| `src/pharma_agents/review_panel.py` | 331 | NEW |
| `src/pharma_agents/review_config.py` | 49 | NEW |
| `src/pharma_agents/report.py` | 651 | NEW |
| `src/pharma_agents/log_utils.py` | 31 | NEW |
| `src/pharma_agents/tools/training.py` | 453 | MODIFIED |
| `src/pharma_agents/crew.py` | 377 | MODIFIED |
| `src/pharma_agents/main.py` | 942 | MODIFIED |
| `src/pharma_agents/memory.py` | 521 | MODIFIED |
| `tests/test_training_tools.py` | 355 | NEW |
| `tests/test_review_panel.py` | 260 | NEW |
| `tests/test_crew_split.py` | 48 | NEW |

---

## CRITICAL Issues

### CRIT-1 — `TeeStream` opens a file in `__init__` with no guaranteed close path

**File:** `main.py:37`

```python
class TeeStream:
    def __init__(self, original_stream, log_file: Path):
        self.log_file = open(log_file, "a", encoding="utf-8")
```

The file handle is opened in `__init__` and closed in `close()`, which is called by `capture_stdout_to_log`'s `finally` block. However, the `__del__` fallback at line 64 is not reliable: CPython may run it, but PyPy or when exceptions occur inside the `finally` branch may not. More critically, if `TeeStream.__init__` raises after `open()` returns (impossible here, but structurally fragile), the handle leaks.

The deeper problem is that the `open()` call in `__init__` means the class cannot be constructed safely in a context where the path is wrong without leaking a file descriptor. The correct pattern is a context manager that opens inside `__enter__`:

```python
@contextmanager
def capture_stdout_to_log(log_file: Path):
    with open(log_file, "a", encoding="utf-8") as fh:
        tee = TeeStream(sys.stdout, fh)  # accept file object, not path
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            yield
        finally:
            sys.stdout = old_stdout
```

**Impact:** File descriptor leak on abnormal exit paths. Low probability under normal operation, but the pattern is structurally wrong.

---

### CRIT-2 — `subprocess.run` without `shell=False` and unvalidated working directory

**File:** `main.py:108,109,110,161,162,205,225,243,858,859,895,900`

All `subprocess.run` calls use list-form commands (correct — no shell injection via the command itself). However, the `cwd` parameter in several calls is derived from user-controlled or computed paths without being resolved to an absolute path first, and `git checkout <path>` at line 243 uses a `Path` object that contains the `experiment_name` string sourced from `PHARMA_EXPERIMENT` env var.

```python
# main.py:243 — experiment_name comes from os.environ
train_py = repo_path / "experiments" / experiment_name / "train.py"
subprocess.run(["git", "checkout", str(train_py)], ...)
```

If `PHARMA_EXPERIMENT` is set to `../../etc/passwd`, the path passed to `git checkout` traverses outside the repo. While `git checkout` would refuse to write to that path, the `cwd` in `git_revert_changes` and several other functions is `repo_path` which is correctly derived from `__file__`, so actual traversal is limited. But the `PHARMA_EXPERIMENTS_DIR` override (memory.py:33) is used directly as a `Path` with no validation against the project root:

```python
override = os.environ.get("PHARMA_EXPERIMENTS_DIR")
if override:
    return Path(override)
```

**Impact:** An attacker who can set environment variables can redirect tool reads/writes to arbitrary filesystem paths. In a CI/CD or container context this is exploitable.

**Fix:** Validate that the resolved path is under the project root before using it:

```python
def get_experiments_dir() -> Path:
    override = os.environ.get("PHARMA_EXPERIMENTS_DIR")
    if override:
        resolved = Path(override).resolve()
        # Guard: must be inside the project tree
        project_root = Path(__file__).parent.parent.parent.resolve()
        resolved.relative_to(project_root)  # raises ValueError if outside
        return resolved
    return get_experiments_root() / get_experiment_name()
```

---

### CRIT-3 — `promote()` reads `baseline_json` without existence check, crashes with unhandled `FileNotFoundError`

**File:** `main.py:883`

```python
existing_config = json.loads(baseline_json.read_text())
```

There is no guard for the case where `baseline_json` does not yet exist in the run branch (e.g., the first promote, or a partially initialized experiment). The function will raise `FileNotFoundError` that propagates to the caller uncaught, producing an unhandled exception with no cleanup. The git repo is left in a checked-out state on `main` without the expected files committed.

**Fix:** Add an existence check and either raise a descriptive error or initialize a sensible default before proceeding.

---

## MAJOR Issues

### MAJ-1 — `run()` is a 550-line monolith violating SRP

**File:** `main.py:390–826`

The `run()` function is 437 lines. It combines:
- Git repository management (init, worktrees, branches)
- Logging setup
- Crew orchestration (3 distinct flow paths)
- Experiment evaluation and memory recording
- Report generation
- Progress display

This violates SRP severely and makes the function untestable in isolation. The git functions are already extracted (good), but the main loop body should be further decomposed. Suggested extraction:
- `_run_iteration(crew, inputs, use_review_panel, ...)  -> IterationResult`
- `_record_iteration_result(result, memory, ...)  -> None`
- A dataclass `IterationResult` to carry the score, verdict, hypothesis, etc.

The function also has cyclomatic complexity well above 10 (multiple nested `if` branches inside a `for` loop with exception handling).

---

### MAJ-2 — Bare `except Exception` swallows unexpected errors in three places

**File:** `main.py:287,384,647`

```python
# main.py:287
except Exception as e:
    logger.debug(f"Could not extract hypothesis from crew result: {e}")

# main.py:384
except Exception as e:
    logger.debug(f"Could not parse hypothesis from log: {e}")

# main.py:647
except Exception as e:
    logger.error(f"Crew error: {e}")
    git_revert_changes(worktree_path)
    ...
    continue
```

The last one (line 647) is particularly dangerous: it catches and continues past all exceptions from the crew, including `KeyboardInterrupt` (which is an `Exception` in Python 2 but not 3, so technically safe), `MemoryError`, `SystemExit`, and genuine programming errors like `AttributeError` or `NameError` that would hide bugs during development.

The first two swallow parsing errors silently with only a DEBUG log, meaning a structural change to CrewAI's output format would silently produce wrong hypothesis data in memory with no visible signal.

**Fix:** Catch specific exceptions (`json.JSONDecodeError`, `AttributeError`, `KeyError` etc.) and let genuinely unexpected exceptions propagate, or at minimum log them at ERROR level with a traceback.

---

### MAJ-3 — `LoggingLLM._call_count` and `_last_call_time` are class-level state — not thread-safe and bleeds between tests

**File:** `crew.py:46–47`

```python
class LoggingLLM(LLM):
    _call_count: int = 0
    _last_call_time: float = 0.0
```

These are mutable class variables. In tests they persist between test runs unless explicitly reset. In concurrent scenarios (even if CrewAI uses threading internally), they are racy without a lock. The increment at line 75 (`LoggingLLM._call_count += 1`) is not atomic.

**Fix:** Use `threading.Lock` around reads/writes, or use `threading.local()` for per-thread counters. Reset in a test fixture if coverage of the retry logic is needed.

---

### MAJ-4 — `report.py` has an XSS vulnerability in HTML generation

**File:** `report.py:263,264`

```python
<td title="{tooltip}">{review_badge} {conf_str}</td>
<td title="{exp.get("hypothesis", "")}">{hypothesis}</td>
```

`tooltip` contains `review_fb` and `concerns` which come from LLM-generated text inserted into an HTML attribute without escaping. A model that returns a hypothesis or feedback string containing `"` or `<script>` tags will break the HTML structure or inject JavaScript if the report is opened in a browser.

Similarly, `hypothesis` at line 231 is truncated but not HTML-escaped before insertion into the table cell.

**Fix:** Use `html.escape()` on all user/LLM-generated strings before inserting into HTML:

```python
import html

tooltip = html.escape(review_fb)
hypothesis_cell = html.escape(hypothesis)
```

---

### MAJ-5 — `memory.py` uses `Optional[float]` truthiness check that fails for `score_after = 0.0`

**File:** `memory.py:240–241`

```python
improvement = None
if score_after and score_before:
    improvement = compute_improvement_pct(score_before, score_after)
```

If `score_after` is `0.0` (a legitimate score in some normalized metrics), `if score_after` evaluates to `False` and improvement is never computed. The same pattern appears at line 261:

```python
if score_after and is_better(score_after, run_memory.best_score):
```

**Fix:** Use explicit `is not None` checks:

```python
if score_after is not None and score_before != 0:
    improvement = compute_improvement_pct(score_before, score_after)
```

---

### MAJ-6 — `_build_llm_config()` returns `dict` without type narrowing; `Any` propagates silently

**File:** `review_panel.py:63`

```python
def _build_llm_config() -> dict:
```

The return type is `dict` (equivalent to `dict[Any, Any]`), which means the caller gets no type safety. The `config_entry` inside the function is annotated as `dict` but mutated with `.update()` from a `dict[str, dict]` value. Pyright in strict mode would flag several untyped dict accesses downstream.

**Fix:** Use `dict[str, object]` or a TypedDict:

```python
from typing import TypedDict

class _LLMConfig(TypedDict):
    config_list: list[dict[str, object]]
    temperature: float
```

---

### MAJ-7 — `review_config.py` reloads YAML from disk on every call — no caching

**File:** `review_config.py:21,26,34,41,46`

```python
def _load_agents_config() -> dict[str, Any]:
    return yaml.safe_load(_AGENTS_YAML.read_text(encoding="utf-8"))
```

Every call to `get_agent_config()`, `get_agent_keys_ordered()`, and `get_max_rounds()` triggers a disk read and YAML parse. In `run_review_panel`, `get_agent_keys_ordered()` and `get_max_rounds()` are both called, and `get_max_rounds()` internally calls `get_agent_keys_ordered()` which calls `_load_agents_config()` again — three disk reads for one panel invocation. `get_max_rounds()` then calls `get_agent_keys_ordered()` again at line 48:

```python
def get_max_rounds() -> int:
    return len(get_agent_keys_ordered()) + 1
```

And `get_agent_keys_ordered()` calls `_load_agents_config()` (another two disk reads inside `run_review_panel`). This is a DRY violation and unnecessary I/O.

**Fix:** Cache with `functools.lru_cache` or a module-level `_AGENTS_CONFIG: dict | None = None` pattern.

---

### MAJ-8 — `CodeCheckTool` and `RunTrainPyTool` use deferred `import subprocess` inside `_run()`

**File:** `tools/training.py:329,373`

```python
def _run(self, argument: str = "check") -> str:
    import subprocess
    ...
```

Deferred imports inside functions are occasionally valid for circular import avoidance, but `subprocess` is stdlib with no circular dependency risk here. The `re` module is also imported inside `_run` and `_outline` (lines 64, 83) in `ReadTrainPyTool`. This is an anti-pattern: it makes the dependency graph invisible to static analysis and linters.

**Fix:** Move all stdlib imports to the module top level.

---

### MAJ-9 — `promote()` silently ignores `subprocess.run` failures for git add/commit

**File:** `main.py:895–903`

```python
subprocess.run(
    ["git", "add", str(baseline), str(baseline_json)],
    cwd=project_root,
    capture_output=True,
)
subprocess.run(
    ["git", "commit", "-m", f"Update baseline from {branch_name}"],
    cwd=project_root,
    capture_output=True,
)
```

Neither call checks `returncode`. If `git add` or `git commit` fails (e.g., due to a detached HEAD, locked index, or merge conflict), the function logs success anyway and the repo is left in an inconsistent state. Contrast with `git_commit_change()` which does check the return code.

**Fix:** Add `check=True` or explicit `returncode != 0` guards consistent with the rest of the git integration.

---

## MINOR Issues

### MIN-1 — Missing `from __future__ import annotations` in several files

**Files:** `main.py`, `memory.py`, `tools/training.py`, `report.py`

These files use `Optional[float]`, `tuple[str, str]`, `list[dict]` etc. Adding `from __future__ import annotations` would unify syntax with `review_panel.py` and `review_config.py` (which already have it) and allow the `X | None` union syntax throughout. `memory.py` still imports `from typing import Optional` (line 13) which is the pre-3.10 style.

---

### MIN-2 — `_parse_verdict_from_text` uses `# type: ignore[assignment]` for a valid narrowing

**File:** `review_panel.py:123`

```python
decision: Literal["approved", "revised", "rejected"] = "approved"
for keyword in ("rejected", "revised", "approved"):
    if keyword in text.lower():
        decision = keyword  # type: ignore[assignment]
        break
```

The `# type: ignore` is needed because `keyword` is typed as `str`, not the `Literal` union. This is fixable cleanly with a cast:

```python
from typing import cast, get_args
_DECISIONS = get_args(Literal["approved", "revised", "rejected"])
decision = cast(Literal["approved", "revised", "rejected"], keyword)
```

Or by iterating over the `Literal` type args directly.

---

### MIN-3 — `report.py` embeds `generate_from_memory` and a `__main__` block in a 651-line module

**File:** `report.py:538–651`

The module is 651 lines, well over the 200-line guideline. The `__main__` block (lines 613–651) and the `generate_from_memory` utility function (lines 538–610) belong in a separate CLI entry point or `report_cli.py` module. The report generation core (`generate_run_report`) is already self-contained and does not need the utility function to be co-located with it.

---

### MIN-4 — `main.py` has a duplicated import of `get_metric_direction`

**File:** `main.py:22,537,608,788`

`get_metric_direction` is imported at the top level (line 22) and then imported again twice inside the `run()` function body (lines 537 and 608). The deferred imports are unnecessary since there is no circular dependency issue:

```python
# line 537 — inside run()
from .memory import get_metric_direction, get_baseline_config

# line 608 — again inside run()
from .memory import get_metric_direction
```

---

### MIN-5 — `log_box()` does not account for emoji width in header centering

**File:** `log_utils.py:11–12`

```python
header = f" {emoji} {title} " if emoji else f" {title} "
pad = max(0, width - len(header) - 2)
```

`len()` counts Unicode code points, not terminal display columns. Emoji characters typically occupy 2 terminal columns. A box with an emoji in the header will be visually off-center by 1–2 characters. This is cosmetic but visible in side-by-side terminal output.

**Fix:** Use `wcwidth.wcswidth(header)` instead of `len(header)`, or strip emoji from the length calculation.

---

### MIN-6 — `main.py:466` uses `assert` for a post-condition check in production code

**File:** `main.py:466`

```python
assert baseline_result.rmse is not None
baseline_score: float = baseline_result.rmse
```

`assert` statements are removed by Python when run with the `-O` (optimize) flag. The comment above says "After success check, rmse is guaranteed to be set", but that guarantee should be expressed as a runtime check with a proper exception, not an assertion. The `run_training()` return type likely declares `rmse: float | None`, so the assert is doing type narrowing here.

**Fix:**

```python
if baseline_result.rmse is None:
    logger.error("Baseline training returned no score despite success=True")
    return
baseline_score: float = baseline_result.rmse
```

---

### MIN-7 — `memory.py:326` sums with a generator that silently skips `None` improvement values

**File:** `memory.py:326–328`

```python
avg_improvement = sum(
    e.improvement_pct for e in successes if e.improvement_pct
) / len(successes)
```

The `if e.improvement_pct` filter silently skips `None` and `0.0` values, but `len(successes)` still counts all successes. If some successful experiments have `improvement_pct=None` (e.g., because `score_after` was `0.0`, as noted in MAJ-5), the average is computed over fewer values but divided by the total count, producing an artificially low average. This will misclassify a run as `LOCAL_OPTIMUM` when it should be `PROGRESS_CONTINUING`.

**Fix:**

```python
valid_improvements = [e.improvement_pct for e in successes if e.improvement_pct is not None]
avg_improvement = sum(valid_improvements) / len(valid_improvements) if valid_improvements else 0.0
```

---

### MIN-8 — `test_review_panel.py` patches `autogen.*` at the wrong namespace

**File:** `tests/test_review_panel.py:175–177`

```python
@patch("autogen.GroupChatManager")
@patch("autogen.GroupChat")
@patch("autogen.ConversableAgent")
```

The production code imports `ConversableAgent`, `GroupChat`, `GroupChatManager` with `from autogen import ...` inside `run_review_panel()` at line 190. When the test patches `autogen.GroupChatManager`, it patches the original object on the `autogen` module — but `run_review_panel` binds the name locally inside the function scope via the deferred import. Whether this works depends on whether Python re-resolves `autogen.GroupChat` on each call into `run_review_panel` (it does, because the import is inside the function). The patches are technically correct here, but they would break immediately if the import were moved to the module level (which is the recommended fix for MAJ-8). The test should patch at the point of use:

```python
@patch("pharma_agents.review_panel.GroupChatManager")
```

This would require either keeping the deferred import or adding explicit module-level imports and adjusting the patch target.

---

### MIN-9 — `test_crew_split.py` uses nested `with patch()` — replace with `@patch` stacking

**File:** `tests/test_crew_split.py:14–26`

```python
with patch("pharma_agents.crew.get_llm", return_value=mock_llm):
    with patch("pharma_agents.crew.LoggingLLM", return_value=mock_llm):
        with patch.dict("os.environ", {...}):
```

Triple nesting at 3 levels in a fixture is hard to read. Prefer stacked `@patch` decorators on the fixture or use `pytest-mock`'s `mocker` fixture, which is already likely available given the `unittest.mock` usage elsewhere.

---

### MIN-10 — `report.py:62` silently replaces missing score with `baseline_score`

**File:** `report.py:55`

```python
score = exp.get("score_after") or exp.get("rmse") or baseline_score
```

If both `score_after` and `rmse` are `0.0` or `None`, this falls back to `baseline_score`, which will make a failed experiment look like it held steady at baseline on the chart — a misleading visualization. For failed experiments the score should either be omitted from the line chart or explicitly marked as `None` with a gap in the Plotly trace.

---

### MIN-11 — `WriteTrainPyTool` dangerous-pattern list is duplicated verbatim in `EditTrainPyTool`

**File:** `tools/training.py:164–176, 252–258`

The exact same list of 10 dangerous patterns appears in both `WriteTrainPyTool._run()` and `EditTrainPyTool._run()`. This violates DRY: adding a new dangerous pattern requires updating two places. Extract to a module-level constant:

```python
_DANGEROUS_PATTERNS: Final[tuple[str, ...]] = (
    "os.system(",
    "subprocess.run(",
    ...
)
```

---

## GOOD PRACTICES Observed

**GP-1 — `review_panel.py` has an explicit, well-documented rate-limiting strategy.**
The `_FREE_TIER_COOLDOWN` and `_FREE_TIER_DELAY` constants at lines 37–38 are named, documented with their rationale (RPM window, shared budget with CrewAI), and applied correctly. This is the right way to handle external API throttling constraints.

**GP-2 — `WriteTrainPyTool` validates LLM-generated code before writing.**
The validation chain (length check, `def train` presence, import check, dangerous pattern scan, inf/NaN warning) at lines 152–195 is a strong defense-in-depth approach for an LLM-driven code generation loop. The warning message for NaN handling is actionable.

**GP-3 — `ReviewVerdict` uses `Literal` for the decision field.**
Using `Literal["approved", "revised", "rejected"]` at line 44 gives compile-time exhaustiveness checking and clean runtime validation via Pydantic. This is the correct pattern for a constrained string enum when a full `StrEnum` would be overkill.

**GP-4 — `review_panel.py` implements a salvage path for partial debates.**
The logic at lines 276–303 that attempts to recover a moderator verdict from an incomplete rate-limited debate, and only re-raises if nothing is recoverable, is thoughtful resilience engineering. The distinction between "recoverable partial" and "unrecoverable" is explicit and logged.

**GP-5 — `AgentMemory._load()` handles a corrupted JSON file gracefully.**
Lines 163–165 catch `json.JSONDecodeError` and `ValueError` and start fresh rather than crashing. The specific exception types are named (not bare `except`), which is correct.

**GP-6 — `EditTrainPyTool` provides a contextual "similar lines" hint on mismatch.**
When `old_text` is not found, the tool at lines 225–239 extracts the likely identifier and shows the 5 nearest matching lines. This is excellent UX for an LLM tool because it gives the model enough context to self-correct without reading the whole file again.

**GP-7 — Test fixtures use `tmp_path` and clean up env vars.**
`test_training_tools.py:35–60` uses pytest's built-in `tmp_path` fixture (no manual `tempfile` management), sets env vars in the fixture body, and pops them in a `yield`-based teardown. This is the correct pytest pattern.

**GP-8 — `run_review_panel` reraises rate limit errors to the outer retry handler.**
The deliberate re-raise at line 300 (`raise debate_error`) when there is no salvageable verdict ensures the outer retry loop in `main.py` handles backoff rather than silently swallowing quota exhaustion with a fake approval. This is correct separation of concerns.

**GP-9 — `InstallPackageTool` uses an allowlist, not a blocklist.**
`ALLOWED_PACKAGES` at lines 10–30 is an explicit allowlist of ML/data science packages. This is the right security posture — blocklists are bypassable, allowlists are not.

**GP-10 — `log_utils.log_box` is a pure function with no side effects.**
The function returns a string and does not print or log directly, leaving the caller in control of output routing. This is the correct design for a utility used in both `print()` and `logger.info()` contexts.

---

## Summary Table

| Severity | Count | Items |
|---|---|---|
| CRITICAL | 3 | CRIT-1 (file handle), CRIT-2 (path traversal), CRIT-3 (unguarded read) |
| MAJOR | 9 | MAJ-1 through MAJ-9 |
| MINOR | 11 | MIN-1 through MIN-11 |
| Good Practices | 10 | GP-1 through GP-10 |

### Priority Order for Fixes

1. **CRIT-2** (path traversal via `PHARMA_EXPERIMENTS_DIR`) — security, exploitable via env
2. **CRIT-3** (unguarded `baseline_json.read_text()` in `promote()`) — crash on first promotion
3. **MAJ-4 / CRIT-1** (file handle management in `TeeStream`) — resource leak
4. **MAJ-5** (zero-score truthiness bug) — silent data corruption in memory
5. **MAJ-7** (silent average miscalculation) — affects run conclusion logic
6. **MAJ-4** (XSS in HTML report) — security if reports are shared internally
7. **MAJ-2** (bare `except Exception`) — hides bugs during development
8. **MAJ-1** (`run()` decomposition) — prerequisite for reliable testing
