"""Reset train.py to baseline state."""

import shutil
from pathlib import Path


def reset_to_baseline() -> None:
    """Copy baseline_train.py to train.py, resetting all agent modifications."""
    tools_dir = Path(__file__).parent
    baseline = tools_dir / "baseline_train.py"
    train = tools_dir / "train.py"

    shutil.copy(baseline, train)
    print("Reset train.py to baseline state")
    print("Baseline RMSE: 1.3175")


if __name__ == "__main__":
    reset_to_baseline()
