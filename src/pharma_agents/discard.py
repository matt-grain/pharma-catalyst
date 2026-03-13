"""Discard a cancelled or failed run."""

import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

from .memory import AgentMemory


def discard(run_number: int, keep_logs: bool = False) -> None:
    """
    Discard a run - remove worktree, memory entry, branch, and optionally logs.

    Use this for runs that were cancelled, got stuck, or produced bad data.
    """
    project_root = Path(__file__).parent.parent.parent
    experiments_dir = project_root / "experiments"
    branch_name = f"run/{run_number:03d}"
    worktree_path = project_root / ".worktrees" / f"run_{run_number:03d}"

    logger.info(f"Discarding run {run_number}...")

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
    memory = AgentMemory(experiments_dir / "memory.json")
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
