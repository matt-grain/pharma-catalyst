"""
Fixed evaluation harness.

THIS FILE IS NEVER MODIFIED BY AGENTS.

Provides consistent evaluation to prevent gaming metrics.
"""

import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

from pharma_agents.memory import get_baseline_config, get_experiments_dir


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    timestamp: str
    score: float | None  # Generic: could be RMSE, MAE, accuracy, etc.
    baseline_score: float
    metric: str  # Name of the metric (RMSE, accuracy, etc.)
    improvement_pct: float | None
    success: bool
    error: str | None
    duration_seconds: float
    recommendation: str  # KEEP or REVERT

    # Aliases for backward compatibility
    @property
    def rmse(self) -> float | None:
        return self.score

    @property
    def baseline_rmse(self) -> float:
        return self.baseline_score

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "score": self.score,
            "baseline_score": self.baseline_score,
            "metric": self.metric,
            "improvement_pct": self.improvement_pct,
            "success": self.success,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "recommendation": self.recommendation,
        }


def run_training(timeout_seconds: int = 180) -> ExperimentResult:
    """
    Execute train.py and capture results.

    Returns ExperimentResult with metrics or error info.
    """
    train_path = get_experiments_dir() / "train.py"
    config = get_baseline_config()
    baseline_score = config["score"]
    metric = config["metric"]
    lower_is_better = config.get("direction", "lower_is_better") == "lower_is_better"

    start_time = time.time()
    timestamp = datetime.now().isoformat()

    def make_result(
        score: float | None,
        success: bool,
        error: str | None,
        duration: float,
    ) -> ExperimentResult:
        improvement_pct = None
        recommendation = "REVERT"
        if score is not None and baseline_score != 0:
            improvement_pct = ((baseline_score - score) / baseline_score) * 100
            if not lower_is_better:
                improvement_pct = -improvement_pct  # Flip for higher_is_better
            is_better = (
                score < baseline_score if lower_is_better else score > baseline_score
            )
            recommendation = "KEEP" if is_better else "REVERT"
        elif score is not None:
            improvement_pct = 0.0
        return ExperimentResult(
            timestamp=timestamp,
            score=score,
            baseline_score=baseline_score,
            metric=metric,
            improvement_pct=improvement_pct,
            success=success,
            error=error,
            duration_seconds=duration,
            recommendation=recommendation,
        )

    try:
        result = subprocess.run(
            [sys.executable, str(train_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=train_path.parent,
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            return make_result(
                None, False, result.stderr or "Non-zero exit code", duration
            )

        # Parse metric from output — prefer ###RESULT### marker over raw lines
        RESULT_MARKER = "###RESULT###"
        output = result.stdout.strip()
        metric_line = [
            line for line in output.split("\n")
            if RESULT_MARKER in line and metric in line
        ]
        if not metric_line:
            # Fallback: original behavior for backward compat
            metric_line = [line for line in output.split("\n") if metric in line]

        if not metric_line:
            return make_result(
                None, False, f"Could not parse {metric} from output", duration
            )

        score = float(metric_line[-1].split(":")[-1].strip())
        return make_result(score, True, None, duration)

    except subprocess.TimeoutExpired:
        return make_result(
            None, False, f"Timeout after {timeout_seconds}s", timeout_seconds
        )
    except Exception as e:
        return make_result(None, False, str(e), time.time() - start_time)


def log_experiment(result: ExperimentResult, log_dir: Path) -> None:
    """Append experiment result to JSON log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiments.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


if __name__ == "__main__":
    result = run_training()
    print(f"Success: {result.success}")
    print(f"{result.metric}: {result.score}")
    print(f"Recommendation: {result.recommendation}")
