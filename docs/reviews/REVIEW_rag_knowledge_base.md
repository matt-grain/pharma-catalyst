# Code Review: RAG Knowledge Base Feature

**Date:** 2026-03-18
**Reviewers:** CrewAI Expert Agent, Python/Opus Code Quality Agent
**Scope:** New files + modifications for hybrid RAG knowledge base

## Review Summary

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 1 | FIXED |
| WARN | 4 | FIXED |
| NOTE | 3 | Accepted (prototype-appropriate) |

## Issues Found & Resolution

### CRITICAL — `_locked_index` schema mismatch (FIXED)
**File:** `knowledge_base.py` lines 417-418, 459-460
**Problem:** `_locked_index` from literature.py initializes with `{"papers": {}}` when file is missing. KB index uses `{"chunks": {}}`. This left a ghost `papers` key in KB index files.
**Fix:** Replaced `_locked_index` usage with a simple `_write_index()` helper that does a full JSON write. KB index is always a full rebuild (not incremental like literature), so file locking is unnecessary.

### WARN — `ClassVar` on module-level variable (FIXED)
**File:** `knowledge_base.py` line 30
**Problem:** `_cached_model: ClassVar = None` — `ClassVar` is only meaningful inside a class body.
**Fix:** Changed to `_cached_model: object = None`.

### WARN — Duplicate model instantiation in `_build_index` (FIXED)
**File:** `knowledge_base.py` line 230
**Problem:** `_build_index` created a new `TextEmbedding()` instead of using `_get_cached_model()`.
**Fix:** Replaced with `model = _get_cached_model()`.

### WARN — `lru_cache` returns mutable dict (FIXED)
**File:** `tool_config.py` line 29
**Problem:** `_load_config()` returns a cached mutable dict. Direct mutation would corrupt the cache.
**Fix:** Added warning docstring. Public API (`get_allowed_packages`, `get_dangerous_patterns`) already returns copies (set/tuple).

### WARN — `.lstrip("# ")` strips characters not prefix (FIXED)
**File:** `knowledge_base.py` line 131
**Problem:** `line.lstrip("# ")` strips any combination of `#` and space chars. `## #Tags` would lose the inner `#`.
**Fix:** Changed to `re.sub(r"^#+\s*", "", line)` which strips only leading `#` sequences.

### NOTE — Double overlap logic (Accepted)
Overlap between paragraphs uses `_apply_overlap`, but within-paragraph splits use no overlap. Acceptable for prototype — chunks are rarely large enough to trigger within-paragraph splitting.

### NOTE — `_build_index` loads full index into memory (Accepted)
For ~1000 chunks x 384-dim embeddings = ~1.5MB JSON. Fine for prototype scale.

### NOTE — No encoding in `_locked_index` (Pre-existing)
Pre-existing issue in `literature.py`. Not introduced by this change.

## Files Reviewed

| File | Verdict |
|------|---------|
| `tools/knowledge_base.py` (new) | PASS after fixes |
| `tool_config.py` (new) | PASS |
| `tool_defaults.yaml` (new) | PASS |
| `ingest_kb.py` (new) | PASS |
| `crew.py` (modified) | PASS |
| `tools/__init__.py` (modified) | PASS |
| `tools/training.py` (modified) | PASS |
| `agents.yaml` (modified) | PASS |
| `tasks.yaml` (modified) | PASS |

## Test Results
- `tests/test_training_tools.py`: 21/21 passed
- End-to-end KB test: Index builds, hybrid search returns correct results, no ghost keys
