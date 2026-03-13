"""Discard a cancelled or failed run."""

import os
import shutil
import subprocess
from pathlib import Path

from loguru import logger

from .memory import AgentMemory, get_experiment_name


def discard(
    run_number: int, experiment: str | None = None, keep_logs: bool = False
) -> None:
    """
    Discard a run - remove worktree, memory entry, branch, and optionally logs.

    Use this for runs that were cancelled, got stuck, or produced bad data.
    """
    experiment = experiment or get_experiment_name()
    project_root = Path(__file__).parent.parent.parent
    experiments_dir = project_root / "experiments" / experiment
    branch_name = f"run/{experiment}/{run_number:03d}"
    worktree_path = project_root / ".worktrees" / experiment / f"run_{run_number:03d}"

    logger.info(f"Discarding {experiment} run {run_number}...")

    # 1. Remove worktree (if exists)
    if worktree_path.exists():
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"Removed worktree {worktree_path}")
        else:
            logger.warning(f"Could not remove worktree: {result.stderr}")
            # Try manual removal
            shutil.rmtree(worktree_path, ignore_errors=True)
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=project_root,
                capture_output=True,
            )
            logger.info("Force-removed worktree directory")

    # 2. Remove from memory
    memory_file = experiments_dir / "memory.json"
    if memory_file.exists():
        memory = AgentMemory(memory_file)
        if run_number in memory.runs:
            del memory.runs[run_number]
            memory.save()
            logger.info(f"Removed run {run_number} from memory.json")
        else:
            logger.info(f"Run {run_number} not found in memory.json")

    # 3. Delete git branch
    result = subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info(f"Deleted branch {branch_name}")
    else:
        logger.info(f"Branch {branch_name} not found or already deleted")

    # 4. Optionally delete logs
    log_dir = experiments_dir / f"run_{run_number:03d}"
    if log_dir.exists():
        if keep_logs:
            logger.info(f"Keeping logs in {log_dir}")
        else:
            shutil.rmtree(log_dir)
            logger.info(f"Deleted log directory {log_dir}")

    logger.success(f"{experiment} run {run_number} discarded.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discard a run")
    parser.add_argument("run_number", type=int, help="Run number to discard")
    parser.add_argument(
        "--experiment",
        "-e",
        default=os.environ.get("PHARMA_EXPERIMENT", "bbbp"),
        help="Experiment name (default: bbbp)",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="Keep log files",
    )
    args = parser.parse_args()

    discard(args.run_number, experiment=args.experiment, keep_logs=args.keep_logs)
