# Remaining Fixes Plan

**For:** Sonnet implementation
**Validated by:** Opus

This plan covers the remaining issues from the code reviews that haven't been fixed yet.

---

## Priority 1: Unit Tests (Major)

### Task 1.1: Add tests for `memory.py`

**File to create:** `tests/test_memory.py`

**Test cases to implement:**

```python
import pytest
from pathlib import Path
from pharma_agents.memory import AgentMemory, is_better, compute_improvement_pct

class TestMetricHelpers:
    """Test standalone metric helper functions."""

    def test_is_better_higher_is_better(self, monkeypatch):
        """When direction is higher_is_better, larger scores win."""
        monkeypatch.setenv("PHARMA_EXPERIMENT", "bbbp")  # Uses higher_is_better
        assert is_better(0.95, 0.90) is True
        assert is_better(0.85, 0.90) is False

    def test_is_better_lower_is_better(self, monkeypatch):
        """When direction is lower_is_better, smaller scores win."""
        monkeypatch.setenv("PHARMA_EXPERIMENT", "solubility")  # Uses lower_is_better
        assert is_better(0.5, 0.8) is True
        assert is_better(1.0, 0.8) is False

    def test_compute_improvement_pct(self):
        """Improvement percentage calculation."""
        # 10% improvement
        assert abs(compute_improvement_pct(0.80, 0.88) - 10.0) < 0.1


class TestAgentMemory:
    """Test AgentMemory class."""

    @pytest.fixture
    def memory_file(self, tmp_path):
        return tmp_path / "memory.json"

    def test_load_empty_memory(self, memory_file):
        """Memory initializes correctly from empty state."""
        memory = AgentMemory(memory_file)
        assert memory.runs == {}
        assert memory.global_best_score is None

    def test_add_experiment_creates_run(self, memory_file):
        """First experiment creates the run entry."""
        memory = AgentMemory(memory_file)
        memory.add_experiment(
            run=1,
            iteration=1,
            hypothesis="Test hypothesis",
            reasoning="Test reasoning",
            result="success",
            score_before=0.80,
            score_after=0.85,
        )
        assert 1 in memory.runs
        assert len(memory.runs[1].experiments) == 1

    def test_add_experiment_updates_best_score(self, memory_file):
        """Successful experiments update best score tracking."""
        memory = AgentMemory(memory_file)
        memory.add_experiment(
            run=1, iteration=1, hypothesis="h1", reasoning="r1",
            result="success", score_before=0.80, score_after=0.85,
        )
        assert memory.runs[1].best_score == 0.85

    def test_is_stuck_threshold(self, memory_file):
        """is_stuck() returns True after consecutive failures."""
        memory = AgentMemory(memory_file)
        # Add 3 failures (default threshold)
        for i in range(3):
            memory.add_experiment(
                run=1, iteration=i+1, hypothesis=f"h{i}", reasoning=f"r{i}",
                result="failure", score_before=0.80, score_after=0.75,
            )
        assert memory.is_stuck(run=1) is True

    def test_finalize_run_detects_progress(self, memory_file):
        """Run with improvement gets PROGRESS_CONTINUING conclusion."""
        memory = AgentMemory(memory_file)
        memory.add_experiment(
            run=1, iteration=1, hypothesis="h1", reasoning="r1",
            result="success", score_before=0.80, score_after=0.90,
        )
        conclusion = memory.finalize_run(1)
        assert "PROGRESS" in conclusion

    def test_corrupted_json_handling(self, memory_file):
        """Gracefully handles malformed memory.json."""
        memory_file.write_text("{ invalid json }")
        # Should not raise - should initialize empty
        memory = AgentMemory(memory_file)
        assert memory.runs == {}

    def test_format_for_prompt(self, memory_file):
        """format_for_prompt returns readable string."""
        memory = AgentMemory(memory_file)
        memory.add_experiment(
            run=1, iteration=1, hypothesis="Add XGBoost", reasoning="Better trees",
            result="success", score_before=0.80, score_after=0.85,
        )
        prompt = memory.format_for_prompt(run=1)
        assert "XGBoost" in prompt or "success" in prompt.lower()
```

**Implementation notes:**
- Use `monkeypatch` to set PHARMA_EXPERIMENT for direction tests
- Memory file should be a `tmp_path` fixture
- Test both success and failure paths

---

### Task 1.2: Add tests for `evaluate.py`

**File to create:** `tests/test_evaluate.py`

**Test cases to implement:**

```python
import pytest
from pathlib import Path
from pharma_agents.tools.evaluate import run_training, ExperimentResult

class TestRunTraining:
    """Test run_training function."""

    @pytest.fixture
    def mock_experiment(self, tmp_path, monkeypatch):
        """Set up a minimal experiment directory."""
        exp_dir = tmp_path / "test_exp"
        exp_dir.mkdir()

        # Create baseline.json
        baseline = exp_dir / "baseline.json"
        baseline.write_text('{"metric": "ROC_AUC", "direction": "higher_is_better", "score": 0.85}')

        # Create minimal train.py that prints metric
        train_py = exp_dir / "train.py"
        train_py.write_text('''
print("Training...")
print("ROC_AUC: 0.90")
''')

        monkeypatch.setenv("PHARMA_EXPERIMENTS_DIR", str(exp_dir))
        monkeypatch.setenv("PHARMA_EXPERIMENT", "test_exp")
        return exp_dir

    def test_successful_training(self, mock_experiment):
        """Training that prints valid metric succeeds."""
        result = run_training()
        assert result.success is True
        assert result.rmse == 0.90  # Note: rmse is the metric value regardless of name

    def test_training_timeout(self, mock_experiment, monkeypatch):
        """Training that hangs times out."""
        train_py = mock_experiment / "train.py"
        train_py.write_text('''
import time
time.sleep(120)  # Hang forever
''')
        result = run_training(timeout_seconds=1)
        assert result.success is False
        assert "timeout" in result.error.lower()

    def test_training_syntax_error(self, mock_experiment):
        """Training with syntax error fails gracefully."""
        train_py = mock_experiment / "train.py"
        train_py.write_text('def broken(')
        result = run_training()
        assert result.success is False
        assert result.error is not None

    def test_missing_metric_in_output(self, mock_experiment):
        """Training without metric in output fails."""
        train_py = mock_experiment / "train.py"
        train_py.write_text('print("Hello world")')
        result = run_training()
        assert result.success is False


class TestExperimentResult:
    """Test ExperimentResult dataclass."""

    def test_recommendation_keep(self):
        """Successful result recommends KEEP."""
        result = ExperimentResult(
            success=True, rmse=0.90, metric="ROC_AUC",
            recommendation="KEEP", error=None
        )
        assert result.recommendation == "KEEP"

    def test_recommendation_revert(self):
        """Failed result recommends REVERT."""
        result = ExperimentResult(
            success=False, rmse=None, metric="ROC_AUC",
            recommendation="REVERT", error="Syntax error"
        )
        assert result.recommendation == "REVERT"
```

**Implementation notes:**
- Use `monkeypatch` for environment variables
- Create minimal train.py scripts for each test case
- Test timeout with a very short timeout value (1 second)

---

## Priority 2: Magic Numbers (Minor)

### Task 2.1: Extract constants in `main.py`

**File:** `src/pharma_agents/main.py`

**Changes:**

Add at top of file (after imports):
```python
# Content truncation limits
MAX_HYPOTHESIS_LENGTH = 200
MAX_REASONING_LENGTH = 300
```

Replace in `parse_hypothesis_from_log()` (around line 280):
```python
# BEFORE:
if len(hypothesis) > 200:
    hypothesis = hypothesis[:197] + "..."
if len(reasoning) > 300:
    reasoning = reasoning[:297] + "..."

# AFTER:
if len(hypothesis) > MAX_HYPOTHESIS_LENGTH:
    hypothesis = hypothesis[:MAX_HYPOTHESIS_LENGTH - 3] + "..."
if len(reasoning) > MAX_REASONING_LENGTH:
    reasoning = reasoning[:MAX_REASONING_LENGTH - 3] + "..."
```

### Task 2.2: Extract constants in `literature.py`

**File:** `src/pharma_agents/tools/literature.py`

**Changes:**

Add at top of file (after imports):
```python
# Content limits
MAX_SUMMARY_LENGTH = 1000
MAX_KEY_METHODS = 5
MIN_FULL_CONTENT_LENGTH = 200
```

Replace usages:
- Line ~88: `summary[:1000]` → `summary[:MAX_SUMMARY_LENGTH]`
- Line ~107: `key_methods[:5]` → `key_methods[:MAX_KEY_METHODS]`
- Line ~180: `len(full_content) > 200` → `len(full_content) > MIN_FULL_CONTENT_LENGTH`

---

## Priority 3: CrewAI Improvements (Medium)

### Task 3.1: Add Pydantic output types to hypothesis task

**File:** `src/pharma_agents/crew.py`

**Changes:**

Add after imports:
```python
from pydantic import BaseModel

class HypothesisOutput(BaseModel):
    """Structured output from hypothesis agent."""
    proposal: str
    reasoning: str
    change_description: str
    literature_insight: str | None = None
```

Update hypothesis_task:
```python
@task
def hypothesis_task(self) -> Task:
    """Task: Propose an improvement."""
    return Task(
        config=self.tasks_config["hypothesis_task"],
        output_pydantic=HypothesisOutput,
    )
```

**Note:** This may require updating `parse_hypothesis_from_log()` in main.py to use the structured output instead of regex parsing.

### Task 3.2: Per-agent temperature tuning

**File:** `src/pharma_agents/crew.py`

**Changes:**

Update `get_llm()`:
```python
def get_llm(temperature: float = 0.7) -> LLM:
    """Get configured LLM from environment variables."""
    model = os.getenv("LLM_MODEL", "gemini/gemini-3-flash-preview")
    return LoggingLLM(
        model=model,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=temperature,
    )
```

Update agent methods:
```python
@agent
def hypothesis_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["hypothesis_agent"],
        llm=get_llm(temperature=0.8),  # Higher for creativity
        ...
    )

@agent
def model_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["model_agent"],
        llm=get_llm(temperature=0.3),  # Lower for code generation
        ...
    )
```

---

## Validation Checklist

After implementing each task:

1. [ ] Run `uv run ruff check src/ tests/`
2. [ ] Run `uv run pytest tests/ -v`
3. [ ] All 16+ tests pass
4. [ ] Commit with descriptive message

---

## Summary

| Task | Priority | Complexity | Status |
|------|----------|------------|--------|
| 1.1 Tests for memory.py | Major | Medium | TODO |
| 1.2 Tests for evaluate.py | Major | Medium | TODO |
| 2.1 Constants in main.py | Minor | Easy | TODO |
| 2.2 Constants in literature.py | Minor | Easy | TODO |
| 3.1 Pydantic output types | Medium | Medium | TODO |
| 3.2 Per-agent temperature | Medium | Easy | TODO |

Start with Priority 1 (tests), then Priority 2 (constants), then Priority 3 (CrewAI improvements).
