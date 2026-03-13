"""Discard a cancelled or failed run."""

import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

from .memory import AgentMemory


def discard(run_number: int, keep_logs: bool = False) -> None:
    """
    Discard a run - remove from memory, delete branch, optionally clean logs.

    Use this for runs that were cancelled, got stuck, or produced bad data.
    """
    project_root = Path(__file__).parent.parent.parent
    experiments_dir = project_root / "experiments"
    branch_name = f"run/{run_number:03d}"

    logger.info(f"Discarding run {run_number}...")

    # 1. Remove from memory
    memory = AgentMemory(experiments_dir / "memory.json")
    if run_number in memory.runs:
        del memory.runs[run_number]
        memory.save()
        logger.info(f"Removed run {run_number} from memory.json")
    else:
        logger.info(f"Run {run_number} not found in memory.json")

    # 2. Delete git branch (if exists and not current)
    # First check current branch
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    current_branch = result.stdout.strip()

    if current_branch == branch_name:
        # Stash any uncommitted changes, switch to main, then restore
        logger.info("Stashing changes and switching to main...")
        subprocess.run(["git", "stash"], cwd=project_root, capture_output=True)
        result = subprocess.run(
            ["git", "checkout", "main"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"Failed to checkout main: {result.stderr}")
            subprocess.run(
                ["git", "stash", "pop"], cwd=project_root, capture_output=True
            )
            return

    # Delete the branch
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

    # Restore stashed changes if we stashed
    if current_branch == branch_name:
        subprocess.run(["git", "stash", "pop"], cwd=project_root, capture_output=True)

    # 3. Optionally delete logs
    log_dir = experiments_dir / f"run_{run_number:03d}"
    if log_dir.exists():
        if keep_logs:
            logger.info(f"Keeping logs in {log_dir}")
        else:
            shutil.rmtree(log_dir)
            logger.info(f"Deleted log directory {log_dir}")

    logger.success(f"Run {run_number} discarded.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pharma_agents.discard <run_number> [--keep-logs]")
        sys.exit(1)

    run_num = int(sys.argv[1])
    keep = "--keep-logs" in sys.argv
    discard(run_num, keep_logs=keep)
