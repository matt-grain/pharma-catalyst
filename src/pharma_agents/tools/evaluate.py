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


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    timestamp: str
    rmse: float | None
    baseline_rmse: float
    improvement_pct: float | None
    success: bool
    error: str | None
    duration_seconds: float
    recommendation: str  # KEEP or REVERT

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "rmse": self.rmse,
            "baseline_rmse": self.baseline_rmse,
            "improvement_pct": self.improvement_pct,
            "success": self.success,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "recommendation": self.recommendation,
        }


def run_training(timeout_seconds: int = 60) -> ExperimentResult:
    """
    Execute train.py and capture results.

    Returns ExperimentResult with metrics or error info.
    """
    train_path = Path(__file__).parent / "train.py"
    baseline_rmse = 1.32  # Measured baseline for ESOL with Morgan FP + RF

    start_time = time.time()
    timestamp = datetime.now().isoformat()

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
            return ExperimentResult(
                timestamp=timestamp,
                rmse=None,
                baseline_rmse=baseline_rmse,
                improvement_pct=None,
                success=False,
                error=result.stderr or "Non-zero exit code",
                duration_seconds=duration,
                recommendation="REVERT",
            )

        # Parse RMSE from output
        output = result.stdout.strip()
        rmse_line = [line for line in output.split("\n") if "RMSE" in line]

        if not rmse_line:
            return ExperimentResult(
                timestamp=timestamp,
                rmse=None,
                baseline_rmse=baseline_rmse,
                improvement_pct=None,
                success=False,
                error="Could not parse RMSE from output",
                duration_seconds=duration,
                recommendation="REVERT",
            )

        rmse = float(rmse_line[-1].split(":")[-1].strip())
        improvement_pct = ((baseline_rmse - rmse) / baseline_rmse) * 100

        return ExperimentResult(
            timestamp=timestamp,
            rmse=rmse,
            baseline_rmse=baseline_rmse,
            improvement_pct=improvement_pct,
            success=True,
            error=None,
            duration_seconds=duration,
            recommendation="KEEP" if rmse < baseline_rmse else "REVERT",
        )

    except subprocess.TimeoutExpired:
        return ExperimentResult(
            timestamp=timestamp,
            rmse=None,
            baseline_rmse=baseline_rmse,
            improvement_pct=None,
            success=False,
            error=f"Timeout after {timeout_seconds}s",
            duration_seconds=timeout_seconds,
            recommendation="REVERT",
        )
    except Exception as e:
        return ExperimentResult(
            timestamp=timestamp,
            rmse=None,
            baseline_rmse=baseline_rmse,
            improvement_pct=None,
            success=False,
            error=str(e),
            duration_seconds=time.time() - start_time,
            recommendation="REVERT",
        )


def log_experiment(result: ExperimentResult, log_dir: Path) -> None:
    """Append experiment result to JSON log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiments.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


if __name__ == "__main__":
    result = run_training()
    print(f"Success: {result.success}")
    print(f"RMSE: {result.rmse}")
    print(f"Recommendation: {result.recommendation}")
