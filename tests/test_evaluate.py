"""
Unit tests for pharma_agents.tools.evaluate.

Tests run_training() with lightweight subprocess-spawned train.py scripts
and verify ExperimentResult structure.
"""

import pytest
from pharma_agents.tools.evaluate import run_training, ExperimentResult


class TestRunTraining:
    """Test run_training function."""

    @pytest.fixture
    def mock_experiment(self, tmp_path, monkeypatch):
        """
        Set up a minimal experiment directory with a valid baseline.json
        and a train.py that prints a metric line.

        Patches get_experiments_dir and get_baseline_config so that
        validate_experiment() is bypassed (it resolves paths from the real
        experiments root, not PHARMA_EXPERIMENTS_DIR).
        """
        exp_dir = tmp_path / "test_exp"
        exp_dir.mkdir()

        baseline_cfg = {
            "metric": "ROC_AUC",
            "direction": "higher_is_better",
            "score": 0.85,
        }

        # Write a baseline train.py that prints the metric
        train_py = exp_dir / "train.py"
        train_py.write_text('print("Training...")\nprint("ROC_AUC: 0.90")\n')

        monkeypatch.setattr(
            "pharma_agents.tools.evaluate.get_experiments_dir",
            lambda: exp_dir,
        )
        monkeypatch.setattr(
            "pharma_agents.tools.evaluate.get_baseline_config",
            lambda: baseline_cfg,
        )

        return exp_dir

    def test_successful_training(self, mock_experiment):
        """Training that prints valid metric succeeds."""
        result = run_training()

        assert result.success is True
        assert result.rmse == pytest.approx(0.90)  # rmse is alias for score
        assert result.metric == "ROC_AUC"
        assert result.error is None
        assert result.recommendation == "KEEP"  # 0.90 > 0.85 for higher_is_better

    def test_training_timeout(self, mock_experiment):
        """Training that hangs is killed and returns a timeout error."""
        train_py = mock_experiment / "train.py"
        train_py.write_text("import time\ntime.sleep(120)  # Hang forever\n")

        result = run_training(timeout_seconds=1)

        assert result.success is False
        assert result.error is not None
        assert "timeout" in result.error.lower()

    def test_training_syntax_error(self, mock_experiment):
        """Training with a syntax error fails gracefully."""
        train_py = mock_experiment / "train.py"
        train_py.write_text("def broken(")  # Unclosed parenthesis → SyntaxError

        result = run_training()

        assert result.success is False
        assert result.error is not None

    def test_missing_metric_in_output(self, mock_experiment):
        """Training that runs successfully but doesn't print the metric fails."""
        train_py = mock_experiment / "train.py"
        train_py.write_text('print("Hello world")\n')

        result = run_training()

        assert result.success is False
        assert result.error is not None
        assert "ROC_AUC" in result.error  # error message names the missing metric


class TestExperimentResult:
    """Test ExperimentResult dataclass construction and properties."""

    def test_recommendation_keep(self):
        """A successful result with recommendation=KEEP is stored correctly."""
        result = ExperimentResult(
            timestamp="2026-03-14T00:00:00",
            score=0.90,
            baseline_score=0.85,
            metric="ROC_AUC",
            improvement_pct=5.88,
            success=True,
            error=None,
            duration_seconds=1.5,
            recommendation="KEEP",
        )

        assert result.recommendation == "KEEP"
        assert result.success is True
        assert result.rmse == pytest.approx(0.90)  # alias for score

    def test_recommendation_revert(self):
        """A failed result with recommendation=REVERT is stored correctly."""
        result = ExperimentResult(
            timestamp="2026-03-14T00:00:00",
            score=None,
            baseline_score=0.85,
            metric="ROC_AUC",
            improvement_pct=None,
            success=False,
            error="Syntax error",
            duration_seconds=0.1,
            recommendation="REVERT",
        )

        assert result.recommendation == "REVERT"
        assert result.success is False
        assert result.rmse is None
        assert result.error == "Syntax error"
