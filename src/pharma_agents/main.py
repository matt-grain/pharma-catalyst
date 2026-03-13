"""
Main entry point for pharma-agents.

Runs the agent crew for N iterations, tracking improvements.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime

from loguru import logger

from .crew import PharmaAgentsCrew
from .tools.evaluate import run_training, log_experiment


# Configure loguru
def setup_logging(log_dir: Path) -> None:
    """Configure loguru for file and console output."""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB",
    )
    logger.info(f"Logging to {log_file}")


# Git integration
def git_init_if_needed(repo_path: Path) -> None:
    """Initialize git repo if not already initialized."""
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        logger.info("Initializing git repository...")
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit: baseline train.py"],
            cwd=repo_path,
            capture_output=True,
        )
        logger.info("Git repository initialized with baseline")


def git_get_next_run_number(repo_path: Path) -> int:
    """Get next run number by checking existing run branches."""
    result = subprocess.run(
        ["git", "branch", "-a"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    branches = result.stdout.strip().split("\n")
    run_numbers = []
    for branch in branches:
        branch = branch.strip().replace("* ", "")
        if branch.startswith("run/"):
            try:
                num = int(branch.split("/")[1])
                run_numbers.append(num)
            except (IndexError, ValueError):
                pass
    return max(run_numbers, default=0) + 1


def git_create_run_branch(repo_path: Path, run_number: int) -> str:
    """Create a new branch for this run, starting from main."""
    branch_name = f"run/{run_number:03d}"

    # Ensure we're on main and it's clean
    subprocess.run(["git", "checkout", "main"], cwd=repo_path, capture_output=True)

    # Create and checkout new branch
    subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=repo_path,
        capture_output=True,
    )
    logger.info(f"Created branch: {branch_name}")
    return branch_name


def git_reset_train_to_baseline(repo_path: Path) -> None:
    """Reset train.py to baseline state."""
    import shutil

    tools_dir = repo_path / "src" / "pharma_agents" / "tools"
    baseline = tools_dir / "baseline_train.py"
    train = tools_dir / "train.py"
    shutil.copy(baseline, train)
    logger.info("Reset train.py to baseline")


def git_commit_change(repo_path: Path, message: str) -> bool:
    """Commit current changes to train.py."""
    try:
        train_py = repo_path / "src" / "pharma_agents" / "tools" / "train.py"
        subprocess.run(
            ["git", "add", str(train_py)],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        logger.info(f"Committed: {message}")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git commit failed: {e}")
        return False


def git_revert_changes(repo_path: Path) -> bool:
    """Revert uncommitted changes to train.py."""
    try:
        train_py = repo_path / "src" / "pharma_agents" / "tools" / "train.py"
        subprocess.run(
            ["git", "checkout", str(train_py)],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        logger.info("Reverted train.py to last committed version")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git revert failed: {e}")
        return False


def run(iterations: int = 10) -> None:
    """
    Run the pharma-agents crew for multiple iterations.

    Each run:
    1. Creates a new branch (run/001, run/002, ...)
    2. Resets train.py to baseline
    3. Agents iterate and improve
    4. Commits improvements to the run branch
    5. Main branch stays clean at baseline
    """
    project_root = Path(__file__).parent.parent.parent
    experiments_dir = project_root / "experiments"
    experiments_dir.mkdir(exist_ok=True)

    # Initialize git if needed
    git_init_if_needed(project_root)

    # Create new run branch
    run_number = git_get_next_run_number(project_root)
    branch_name = git_create_run_branch(project_root, run_number)

    # Setup logging for this run
    run_log_dir = experiments_dir / f"run_{run_number:03d}"
    run_log_dir.mkdir(exist_ok=True)
    setup_logging(run_log_dir)

    logger.info("=" * 60)
    logger.info("PHARMA-AGENTS: Autonomous Molecular ML Optimization")
    logger.info(f"Run #{run_number} on branch: {branch_name}")
    logger.info("=" * 60)

    # Reset train.py to baseline
    git_reset_train_to_baseline(project_root)

    # Get baseline
    logger.info("Establishing baseline...")
    baseline_result = run_training()
    if not baseline_result.success:
        logger.error(f"Baseline training failed: {baseline_result.error}")
        return

    baseline_rmse = baseline_result.rmse
    best_rmse = baseline_rmse
    logger.info(f"Baseline RMSE: {baseline_rmse:.4f}")

    # Commit baseline state on this branch
    git_commit_change(
        project_root, f"[Run {run_number}] Baseline: RMSE {baseline_rmse:.4f}"
    )

    # Initialize crew
    crew = PharmaAgentsCrew()

    # Track experiment history for context
    experiment_history: list[dict] = []

    # Run iterations
    for i in range(iterations):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"ITERATION {i + 1}/{iterations}")
        logger.info("=" * 60)
        logger.info(f"Current best RMSE: {best_rmse:.4f}")

        # Prepare inputs for the crew
        inputs = {
            "property": "solubility (logS)",
            "baseline_rmse": f"{best_rmse:.4f}",
            "experiment_history": json.dumps(experiment_history[-5:], indent=2)
            if experiment_history
            else "No previous experiments",
        }

        # Run the crew
        try:
            logger.info("Running agent crew...")
            result = crew.crew().kickoff(inputs=inputs)
            logger.info("Crew completed")
            logger.debug(f"Crew output: {result}")
        except Exception as e:
            logger.error(f"Crew error: {e}")
            git_revert_changes(project_root)
            continue

        # Evaluate the result
        logger.info("Evaluating changes...")
        eval_result = run_training()
        log_experiment(eval_result, experiments_dir)

        # Record in history
        experiment_entry = {
            "iteration": i + 1,
            "timestamp": datetime.now().isoformat(),
            "rmse": eval_result.rmse,
            "success": eval_result.success,
            "recommendation": eval_result.recommendation,
        }
        experiment_history.append(experiment_entry)

        if eval_result.success and eval_result.rmse is not None:
            if eval_result.rmse < best_rmse:
                improvement = ((best_rmse - eval_result.rmse) / best_rmse) * 100
                logger.success(
                    f"IMPROVEMENT: {eval_result.rmse:.4f} ({improvement:.1f}% better)"
                )
                best_rmse = eval_result.rmse
                # Commit the improvement
                git_commit_change(
                    project_root,
                    f"[Run {run_number}] Iter {i + 1}: RMSE {eval_result.rmse:.4f} ({improvement:.1f}% better)",
                )
            else:
                logger.warning(f"NO IMPROVEMENT: {eval_result.rmse:.4f} (reverting)")
                git_revert_changes(project_root)
        else:
            logger.error(f"FAILED: {eval_result.error}")
            git_revert_changes(project_root)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Initial RMSE:  {baseline_rmse:.4f}")
    logger.info(f"Final RMSE:    {best_rmse:.4f}")
    total_improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
    logger.success(f"Total Improvement: {total_improvement:.1f}%")
    logger.info(f"Experiments:   {len(experiment_history)}")
    logger.info(f"Log dir:       {experiments_dir}")

    # Show git log
    logger.info("")
    logger.info(f"Git history ({branch_name}):")
    result = subprocess.run(
        ["git", "log", "--oneline", "-10"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().split("\n"):
        logger.info(f"  {line}")

    logger.info("")
    logger.info("To promote this run as new baseline:")
    logger.info(f"  uv run python -m pharma_agents.promote {run_number}")


def promote(run_number: int) -> None:
    """
    Promote a run branch to main, making it the new baseline.

    This merges run/XXX into main and updates baseline_train.py.
    """
    project_root = Path(__file__).parent.parent.parent
    branch_name = f"run/{run_number:03d}"

    logger.info(f"Promoting {branch_name} to main...")

    # Checkout main
    result = subprocess.run(
        ["git", "checkout", "main"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Failed to checkout main: {result.stderr}")
        return

    # Merge run branch
    result = subprocess.run(
        ["git", "merge", branch_name, "-m", f"Promote {branch_name} as new baseline"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"Failed to merge: {result.stderr}")
        return

    # Update baseline_train.py with current train.py
    import shutil

    tools_dir = project_root / "src" / "pharma_agents" / "tools"
    train = tools_dir / "train.py"
    baseline = tools_dir / "baseline_train.py"
    shutil.copy(train, baseline)

    # Commit the new baseline
    subprocess.run(["git", "add", str(baseline)], cwd=project_root, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"Update baseline from {branch_name}"],
        cwd=project_root,
        capture_output=True,
    )

    # Get new baseline RMSE
    from .tools.evaluate import run_training

    result = run_training()
    logger.success(f"Promoted {branch_name} to main!")
    logger.info(f"New baseline RMSE: {result.rmse:.4f}")


if __name__ == "__main__":
    run(iterations=5)
