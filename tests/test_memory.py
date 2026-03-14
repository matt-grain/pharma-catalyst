"""Unit tests for pharma_agents.memory module."""

import pytest
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

    def test_compute_improvement_pct(self, monkeypatch):
        """Improvement percentage calculation."""
        monkeypatch.setenv(
            "PHARMA_EXPERIMENT", "solubility"
        )  # lower_is_better: old=0.80, new=0.88 is worse
        # For lower_is_better: improvement = (old - new) / old * 100
        # old=0.80, new=0.72 → (0.80 - 0.72) / 0.80 * 100 = 10.0%
        assert abs(compute_improvement_pct(0.80, 0.72) - 10.0) < 0.1


class TestAgentMemory:
    """Test AgentMemory class."""

    @pytest.fixture
    def memory_file(self, tmp_path):
        return tmp_path / "memory.json"

    @pytest.fixture(autouse=True)
    def set_experiment(self, monkeypatch):
        """Ensure PHARMA_EXPERIMENT points to a valid experiment for all tests."""
        monkeypatch.setenv("PHARMA_EXPERIMENT", "bbbp")

    def test_load_empty_memory(self, memory_file):
        """Memory initializes correctly from empty state."""
        memory = AgentMemory(memory_file)
        assert memory.runs == {}
        # global_best_score is seeded from baseline (not None) when no memory file exists
        assert memory.global_best_score is not None

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
            insight="Score improved with new features.",
        )
        assert 1 in memory.runs
        assert len(memory.runs[1].experiments) == 1

    def test_add_experiment_updates_best_score(self, memory_file):
        """Successful experiments update best score tracking."""
        memory = AgentMemory(memory_file)
        memory.add_experiment(
            run=1,
            iteration=1,
            hypothesis="h1",
            reasoning="r1",
            result="success",
            score_before=0.80,
            score_after=0.95,
            insight="XGBoost outperformed logistic regression.",
        )
        assert memory.runs[1].best_score == 0.95

    def test_is_stuck_threshold(self, memory_file):
        """is_stuck() returns True after consecutive failures."""
        memory = AgentMemory(memory_file)
        # Add 3 failures (default threshold)
        for i in range(3):
            memory.add_experiment(
                run=1,
                iteration=i + 1,
                hypothesis=f"h{i}",
                reasoning=f"r{i}",
                result="failure",
                score_before=0.80,
                score_after=0.75,
                insight=f"Attempt {i} did not improve.",
            )
        assert memory.is_stuck(run=1) is True

    def test_finalize_run_detects_progress(self, memory_file):
        """Run with improvement gets PROGRESS_CONTINUING conclusion."""
        memory = AgentMemory(memory_file)
        memory.add_experiment(
            run=1,
            iteration=1,
            hypothesis="h1",
            reasoning="r1",
            result="success",
            score_before=0.80,
            score_after=0.90,
            insight="Large improvement from ensemble method.",
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
            run=1,
            iteration=1,
            hypothesis="Add XGBoost",
            reasoning="Better trees",
            result="success",
            score_before=0.80,
            score_after=0.85,
            insight="XGBoost provided measurable uplift.",
        )
        prompt = memory.format_for_prompt(current_run=1)
        assert "XGBoost" in prompt or "success" in prompt.lower()
