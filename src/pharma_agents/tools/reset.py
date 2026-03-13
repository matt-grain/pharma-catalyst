"""Reset train.py to baseline state."""

import shutil

from pharma_agents.memory import get_baseline_config, get_experiments_dir


def reset_to_baseline() -> None:
    """Copy baseline_train.py to train.py, resetting all agent modifications."""
    experiments_dir = get_experiments_dir()
    baseline = experiments_dir / "baseline_train.py"
    train = experiments_dir / "train.py"

    config = get_baseline_config()
    shutil.copy(baseline, train)
    print("Reset train.py to baseline state")
    print(f"Baseline {config['metric']}: {config['score']:.4f}")


if __name__ == "__main__":
    reset_to_baseline()
