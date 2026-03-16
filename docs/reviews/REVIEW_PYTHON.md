# Pharma-Agents: Python Code Quality Review

**Reviewed by:** Claude Opus (autonomous agent)
**Date:** 2026-03-14

## Executive Summary

This is a well-architected Python project demonstrating senior-level engineering practices for an autonomous multi-agent ML optimization system. The code shows strong domain knowledge in ML/cheminformatics and thoughtful design patterns. However, there are several issues that need addressing before presenting this as portfolio code.

---

## STRENGTHS (Senior-Level Demonstrations)

### 1. **Clean Module Organization**
- Clear separation: `tools/` (API interactions), `memory.py` (persistence), `crew.py` (orchestration), `main.py` (entry point)
- Tools split by domain: `arxiv.py`, `literature.py`, `training.py`, `skills.py`
- Proper `__init__.py` with explicit `__all__` exports
- Deprecated module with backwards compatibility (`custom_tools.py`)

### 2. **Type Annotations**
- Consistent use of modern Python 3.11+ syntax: `list[dict]`, `float | None`, `str | None`
- Dataclasses with proper type hints (`Experiment`, `RunMemory`, `ExperimentResult`)
- ClassVar used correctly for class-level state in tools

### 3. **Domain-Driven Design**
- `memory.py` encapsulates experiment history with smart methods (`is_stuck()`, `is_globally_stagnant()`, `format_for_prompt()`)
- Metric-agnostic design: `is_better()`, `compute_improvement_pct()` support both higher/lower-is-better
- Experiment validation with clear error messages

### 4. **Safety & Rate Limiting**
- `ALLOWED_PACKAGES` whitelist in `InstallPackageTool`
- Rate limiting with `max_papers_per_run`, `min_interval_seconds`
- Class-level counters (`_searches_done`, `_papers_fetched`) to enforce limits

### 5. **Error Handling Patterns**
- Context manager for stdout capture (`capture_stdout_to_log`)
- Graceful degradation in fetch tools (alphaxiv -> markdown.new fallback)
- Explicit error returns rather than silent failures

### 6. **Production Readiness**
- Loguru configuration with rotation
- JSONL experiment logging
- HTML report generation with Plotly
- Git worktree isolation for parallel runs

---

## ISSUES

### CRITICAL

#### 1. TeeStream file handle leak on exception ✅ FIXED
**File:** `/src/pharma_agents/main.py`, lines 29-56
**Status:** Fixed in commit `20509da` - Added `__del__` method for cleanup on garbage collection.

The `TeeStream.__init__` opens a file but if an exception occurs before `close()` is called, the file handle leaks.

```python
# BEFORE (current code):
def __init__(self, original_stream, log_file: Path):
    self.original = original_stream
    self.log_file = open(log_file, "a", encoding="utf-8")  # Leaks if never closed
    self._closed = False

# AFTER (fix):
def __init__(self, original_stream, log_file: Path):
    self.original = original_stream
    self._file_handle = open(log_file, "a", encoding="utf-8")
    self._closed = False

def __del__(self):
    if not self._closed:
        self.close()
```

Alternatively, use `atexit.register()` or make the context manager handle cleanup in `__exit__`.

#### 2. InstallPackageTool allows arbitrary code execution
**File:** `/src/pharma_agents/tools/training.py`, lines 151-206

While there's a whitelist, the `uv add` command modifies `pyproject.toml` permanently. A malicious agent could:
1. Install `tensorflow` (allowed) which has post-install hooks
2. The installed package persists across runs

**Mitigation:** Consider using `uv pip install --prefix` to a temporary venv, or require human approval.

#### 3. Path traversal in SkillLoaderTool ✅ FIXED
**File:** `/src/pharma_agents/tools/skills.py`, lines 27-63
**Status:** Fixed in commit `20509da` - Added regex validation to only allow `[a-z0-9-]+` characters.

```python
skill_name = skill_name.strip().lower()
# ...
skill_paths = [
    SKILLS_DIR / "scientific" / f"{skill_name}.md",  # skill_name not sanitized!
    SKILLS_DIR / f"{skill_name}.md",
]
```

An input like `../../../etc/passwd` or `..\\..\\secrets` could read arbitrary files.

```python
# AFTER (fix):
import re

def _run(self, skill_name: str) -> str:
    skill_name = skill_name.strip().lower()

    # Sanitize: only allow alphanumeric and hyphens
    if not re.match(r'^[a-z0-9-]+$', skill_name):
        return f"Error: Invalid skill name '{skill_name}'. Use only lowercase letters, numbers, and hyphens."

    # ... rest of function
```

---

### MAJOR

#### 4. Missing type annotations on several functions
**File:** `/src/pharma_agents/main.py`

```python
# BEFORE:
def parse_hypothesis_from_log(log_file: Path) -> tuple[str, str]:
    # ...
    def clean_terminal_chars(text: str) -> str:  # Good, but nested function
        # ...

# Functions missing return types or have incomplete signatures:
def git_init_if_needed(repo_path: Path) -> None:  # OK
def git_get_next_run_number(repo_path: Path, experiment: str) -> int:  # OK
def run(iterations: int = 10) -> None:  # OK
def promote(run_number: int) -> None:  # OK - but has bugs, see below
```

**Issue:** The `promote()` function references `eval_result.metric` and `eval_result.score` but `ExperimentResult.metric` is a string and `.score` exists. However, it prints `eval_result.rmse` which is an alias - inconsistent.

#### 5. Mutable class default in SkillLoaderTool ✅ FIXED
**File:** `/src/pharma_agents/tools/skills.py`, line 26
**Status:** Fixed in commit `20509da` - Changed to `ClassVar[list[str]]`.

```python
# BEFORE (bug - shared mutable default):
class SkillLoaderTool(BaseTool):
    _skills_loaded: list = []  # SHARED across all instances!

# AFTER (fix):
class SkillLoaderTool(BaseTool):
    _skills_loaded: ClassVar[list[str]] = []  # If intentionally shared
    # OR if per-instance:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._skills_loaded = []
```

#### 6. FetchMorePapersTool has fragile line parsing ✅ FIXED
**File:** `/src/pharma_agents/tools/literature.py`, lines 319-329
**Status:** Fixed in commit `20509da` - Changed to use `enumerate()` instead of `lines.index()`.

```python
for line in lines:
    if line.startswith("# "):
        title = line[2:].strip()
    elif line.startswith("## Summary") or line.startswith("## Abstract"):
        idx = lines.index(line)  # BUG: finds first occurrence, not current
        summary = " ".join(lines[idx + 1 : idx + 5]).strip()
        break
```

`lines.index(line)` returns the first occurrence, not the current index. Use `enumerate()`:

```python
# AFTER (fix):
for idx, line in enumerate(lines):
    if line.startswith("# "):
        title = line[2:].strip()
    elif line.startswith("## Summary") or line.startswith("## Abstract"):
        summary = " ".join(lines[idx + 1 : idx + 5]).strip()
        break
```

#### 7. Hardcoded magic numbers
**File:** `/src/pharma_agents/main.py`

- Line 280-283: `200`, `300` character truncation limits
- `timeout_seconds: int = 60` in multiple places

**File:** `/src/pharma_agents/tools/literature.py`

- Line 88: `[:1000]` summary truncation
- Line 107: `[:5]` key methods limit
- Line 180: `len(full_content) > 200` threshold

These should be constants at module level:

```python
# Constants for content limits
MAX_HYPOTHESIS_LENGTH = 200
MAX_REASONING_LENGTH = 300
MAX_SUMMARY_LENGTH = 1000
TRAINING_TIMEOUT_SECONDS = 60
```

---

### MINOR

#### 8. Inconsistent docstring style
Some functions have docstrings, others don't. The codebase mixes Google-style and imperative style:

```python
# memory.py - imperative (good)
def is_better(new_score: float, old_score: float) -> bool:
    """Check if new_score is better than old_score based on metric direction."""

# main.py - incomplete
def git_init_if_needed(repo_path: Path) -> None:
    """Initialize git repo if not already initialized."""
    # Missing Args/Returns sections
```

Standardize on Google-style for public APIs.

#### 9. Tests use `_run()` directly instead of public interface
**File:** `/tests/test_tools.py`

```python
# BEFORE:
tool = ArxivSearchTool()
result = tool._run("molecular property prediction")  # Private method!

# AFTER (better):
result = tool.run("molecular property prediction")  # If BaseTool.run() exists
# Or document that _run is the intended interface per CrewAI
```

#### 10. Test cleanup doesn't handle exceptions
**File:** `/tests/test_tools.py`, lines 39-44

```python
# Cleanup
if "PHARMA_EXPERIMENT" in os.environ:
    del os.environ["PHARMA_EXPERIMENT"]
```

If the test raises before cleanup, env vars persist. Use `pytest-env` or `monkeypatch` fixture.

#### 11. Missing `__all__` consistency
**File:** `/src/pharma_agents/tools/__init__.py` vs `/src/pharma_agents/tools/custom_tools.py`

`custom_tools.py` is missing `InstallPackageTool` from its re-exports but `__init__.py` includes it. This creates confusion for backwards compatibility.

---

## TEST COVERAGE GAPS

1. **No unit tests for `main.py`** - the core orchestration logic
2. **No tests for `memory.py`** - critical persistence layer
3. **No tests for `evaluate.py`** - the experiment harness
4. **No tests for `report.py`** - HTML generation
5. **Missing edge cases:**
   - What happens when `memory.json` is corrupted?
   - What if git commands fail?
   - What if arxiv returns malformed XML?

**Recommended additions:**

```python
# tests/test_memory.py
class TestAgentMemory:
    def test_load_empty_memory(self, tmp_path):
        """Memory initializes correctly from empty state."""

    def test_add_experiment_updates_best_score(self):
        """Successful experiments update best score tracking."""

    def test_finalize_run_detects_local_optimum(self):
        """Run with diminishing returns gets LOCAL_OPTIMUM conclusion."""

    def test_is_stuck_threshold(self):
        """is_stuck() respects consecutive failure threshold."""

    def test_corrupted_json_handling(self, tmp_path):
        """Gracefully handles malformed memory.json."""
```

---

## PERFORMANCE CONSIDERATIONS

1. **Embedding model loaded on every store/query** (`LiteratureStoreTool`, `LiteratureQueryTool`)
   - Consider caching the `TextEmbedding` model instance

2. **Full index loaded on every query** - OK for small DBs but won't scale
   - Consider FAISS or ChromaDB for larger literature databases

3. **Subprocess calls block** - OK for single-run but consider async for parallel experiments

---

## RECOMMENDED FIXES (Priority Order)

1. ✅ **Critical:** Fix path traversal in `SkillLoaderTool` — DONE
2. ✅ **Critical:** Add file handle cleanup to `TeeStream` — DONE
3. ✅ **Major:** Fix mutable default in `SkillLoaderTool._skills_loaded` — DONE
4. ✅ **Major:** Fix `lines.index()` bug in `FetchMorePapersTool` — DONE
5. ✅ **Major:** Add unit tests for `memory.py` and `evaluate.py` — DONE
6. ✅ **Minor:** Extract magic numbers to named constants — DONE
7. **Minor:** Standardize docstring format

---

## CONCLUSION

This codebase demonstrates **strong senior-level Python skills**:
- Clean architecture with clear separation of concerns
- Modern Python idioms (dataclasses, type hints, pathlib)
- Domain expertise in ML experimentation workflows
- Thoughtful safety measures (rate limiting, whitelisting)

The identified issues are fixable and don't undermine the overall design quality. The path traversal and mutable default bugs should be addressed before the interview, as they're the type of issue a code reviewer would flag immediately.
