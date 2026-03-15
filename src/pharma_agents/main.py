"""
Main entry point for pharma-agents.

Runs the agent crew for N iterations, tracking improvements.
"""

import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

from loguru import logger

from .crew import PharmaAgentsCrew
from .tools.evaluate import run_training, log_experiment
from .memory import AgentMemory, get_metric_name, is_better, compute_improvement_pct

# Content truncation limits
MAX_HYPOTHESIS_LENGTH = 200
MAX_REASONING_LENGTH = 300


class TeeStream:
    """Stream that writes to both stdout and a log file."""

    # Regex to strip ANSI escape codes
    ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

    def __init__(self, original_stream, log_file: Path):
        self.original = original_stream
        self.log_file = open(log_file, "a", encoding="utf-8")
        self._closed = False

    def write(self, data):
        self.original.write(data)
        # Write to log file if still open (CrewAI async events may fire late)
        if not self._closed:
            try:
                clean_data = self.ANSI_ESCAPE.sub("", data)
                self.log_file.write(clean_data)
                self.log_file.flush()
            except (ValueError, OSError):
                pass  # File closed or invalid, ignore late writes

    def flush(self):
        self.original.flush()
        if not self._closed:
            try:
                self.log_file.flush()
            except (ValueError, OSError):
                pass

    def close(self):
        self._closed = True
        self.log_file.close()

    def __del__(self):
        """Cleanup file handle on garbage collection."""
        if not self._closed:
            self.close()


@contextmanager
def capture_stdout_to_log(log_file: Path):
    """Context manager to tee stdout to log file."""
    tee = TeeStream(sys.stdout, log_file)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = old_stdout
        tee.close()


# Configure loguru
def setup_logging(log_dir: Path) -> Path:
    """Configure loguru for file output only (no stdout)."""
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Remove default stderr handler
    logger.remove()

    # Add file handler only
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB",
    )
    logger.info(f"Logging to {log_file}")
    return log_file


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


def git_get_next_run_number(repo_path: Path, experiment: str) -> int:
    """Get next run number for this experiment by checking existing branches."""
    result = subprocess.run(
        ["git", "branch", "-a"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    branches = result.stdout.strip().split("\n")
    run_numbers = []
    prefix = f"run/{experiment}/"
    for branch in branches:
        branch = branch.strip().replace("* ", "").replace("+ ", "")
        if branch.startswith(prefix):
            try:
                num = int(branch.split("/")[2])
                run_numbers.append(num)
            except (IndexError, ValueError):
                pass
    return max(run_numbers, default=0) + 1


def git_create_worktree(
    repo_path: Path, experiment: str, run_number: int
) -> tuple[str, Path]:
    """Create a worktree for this run, isolated from main.

    Returns (branch_name, worktree_path).
    """
    branch_name = f"run/{experiment}/{run_number:03d}"
    worktrees_dir = repo_path / ".worktrees" / experiment
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    worktree_path = worktrees_dir / f"run_{run_number:03d}"

    # Remove stale worktree if exists
    if worktree_path.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            cwd=repo_path,
            capture_output=True,
        )

    # Create worktree with new branch from main
    result = subprocess.run(
        ["git", "worktree", "add", str(worktree_path), "-b", branch_name, "main"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Branch might exist, try without -b
        subprocess.run(
            ["git", "worktree", "add", str(worktree_path), branch_name],
            cwd=repo_path,
            capture_output=True,
        )

    logger.info(f"Created worktree: {worktree_path} on branch {branch_name}")
    return branch_name, worktree_path


def git_reset_train_to_baseline(repo_path: Path) -> None:
    """Reset train.py to baseline state."""
    import shutil
    from .memory import get_experiment_name

    experiment_name = get_experiment_name()
    experiments_dir = repo_path / "experiments" / experiment_name
    baseline = experiments_dir / "baseline_train.py"
    train = experiments_dir / "train.py"

    if not baseline.exists():
        raise FileNotFoundError(
            f"Baseline training script not found: {baseline}\n"
            f"Each experiment needs a baseline_train.py file."
        )
    shutil.copy(baseline, train)
    logger.info(f"Reset train.py to baseline ({experiment_name})")


def git_commit_change(repo_path: Path, message: str) -> bool:
    """Commit current changes to train.py."""
    from .memory import get_experiment_name

    experiment_name = get_experiment_name()
    try:
        train_py = repo_path / "experiments" / experiment_name / "train.py"
        subprocess.run(
            ["git", "add", str(train_py)],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        # Check if there are staged changes
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_path,
            capture_output=True,
        )
        if result.returncode == 0:
            logger.info("No changes to commit (already at this state)")
            return True
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Git commit failed: {result.stderr or result.stdout}")
            return False
        logger.info(f"Committed: {message}")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git commit error: {e}")
        return False


def git_revert_changes(repo_path: Path) -> bool:
    """Revert uncommitted changes to train.py."""
    from .memory import get_experiment_name

    experiment_name = get_experiment_name()
    try:
        train_py = repo_path / "experiments" / experiment_name / "train.py"
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


def parse_hypothesis_from_log(log_file: Path) -> tuple[str, str]:
    """Extract hypothesis and reasoning from log file.

    Looks for PROPOSAL: and REASONING: lines from hypothesis_agent output.
    Returns (hypothesis, reasoning) tuple with defaults if not found.
    """
    hypothesis = "See log for details"
    reasoning = "See log for details"

    def clean_terminal_chars(text: str) -> str:
        """Remove terminal UI box-drawing characters and clean up text."""
        import re

        # Remove box-drawing characters (│ ╭ ╮ ╯ ╰ ─ etc.)
        text = re.sub(r"[│╭╮╯╰─╱╲┌┐└┘├┤┬┴┼]", "", text)
        # Remove ANSI escape sequences
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        # Clean up multiple spaces
        text = re.sub(r"  +", " ", text)
        return text.strip()

    try:
        content = log_file.read_text(encoding="utf-8")
        # Look for the last PROPOSAL/REASONING pair (most recent iteration)
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if "PROPOSAL:" in line:
                # Extract everything after PROPOSAL:
                proposal_start = line.find("PROPOSAL:") + len("PROPOSAL:")
                hypothesis = clean_terminal_chars(line[proposal_start:])
                # If hypothesis continues on next lines, grab them
                if not hypothesis and i + 1 < len(lines):
                    hypothesis = clean_terminal_chars(lines[i + 1])
            elif "REASONING:" in line:
                reasoning_start = line.find("REASONING:") + len("REASONING:")
                reasoning = clean_terminal_chars(line[reasoning_start:])
                if not reasoning and i + 1 < len(lines):
                    reasoning = clean_terminal_chars(lines[i + 1])

        # Truncate if too long
        if len(hypothesis) > MAX_HYPOTHESIS_LENGTH:
            hypothesis = hypothesis[: MAX_HYPOTHESIS_LENGTH - 3] + "..."
        if len(reasoning) > MAX_REASONING_LENGTH:
            reasoning = reasoning[: MAX_REASONING_LENGTH - 3] + "..."

    except Exception as e:
        logger.debug(f"Could not parse hypothesis from log: {e}")

    return hypothesis, reasoning


def run(iterations: int = 10) -> None:
    """
    Run the pharma-agents crew for multiple iterations.

    Each run:
    1. Creates an isolated worktree (.worktrees/run_XXX/)
    2. Works in that worktree (no stash/conflict issues)
    3. Agents iterate and improve train.py
    4. Commits improvements to the run branch
    5. Main branch stays completely untouched
    """
    from .memory import get_experiment_name

    project_root = Path(__file__).parent.parent.parent.resolve()
    experiment_name = get_experiment_name()
    main_experiments_root = project_root / "experiments"
    main_experiments_dir = main_experiments_root / experiment_name
    main_experiments_dir.mkdir(parents=True, exist_ok=True)

    # Load persistent memory (per-experiment, stays in main repo)
    memory = AgentMemory(main_experiments_dir / "memory.json")
    total_experiments = sum(len(r.experiments) for r in memory.runs.values())
    logger.info(
        f"Loaded memory: {total_experiments} past experiments across {len(memory.runs)} runs"
    )

    # Get metric name from config (avoid hardcoding "RMSE")
    metric = get_metric_name()

    # Initialize git if needed
    git_init_if_needed(project_root)

    # Create isolated worktree for this run (per-experiment numbering)
    run_number = git_get_next_run_number(project_root, experiment_name)
    branch_name, worktree_path = git_create_worktree(
        project_root, experiment_name, run_number
    )

    # Paths for this run (in worktree, experiment-specific)
    experiments_dir = worktree_path / "experiments" / experiment_name
    experiments_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging for this run (in main repo, not worktree)
    run_log_dir = main_experiments_dir / f"run_{run_number:03d}"
    run_log_dir.mkdir(exist_ok=True)
    log_file = setup_logging(run_log_dir)

    logger.info("=" * 60)
    logger.info("PHARMA-AGENTS: Autonomous Molecular ML Optimization")
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Run #{run_number} on branch: {branch_name}")
    logger.info("=" * 60)

    # Print header to stdout (will also be captured to log)
    print(f"\n{'=' * 60}")
    print(f"PHARMA-AGENTS: {experiment_name}")
    print(f"Run #{run_number} on branch: {branch_name}")
    print(f"Log file: {log_file}")
    print(f"Worktree: {worktree_path}")
    print(f"{'=' * 60}\n")

    # Set experiments dir for tools/crew (they use get_experiments_dir())
    # Resolve to absolute path to avoid FileReadTool issues
    os.environ["PHARMA_EXPERIMENTS_DIR"] = str(experiments_dir.resolve())

    # Reset train.py to baseline (in worktree)
    git_reset_train_to_baseline(worktree_path)

    # Get baseline
    logger.info("Establishing baseline...")
    baseline_result = run_training()
    if not baseline_result.success:
        logger.error(f"Baseline training failed: {baseline_result.error}")
        return

    # After success check, rmse is guaranteed to be set
    assert baseline_result.rmse is not None
    baseline_score: float = baseline_result.rmse
    best_score: float = baseline_score
    logger.info(f"Baseline {metric}: {baseline_score:.4f}")

    # Commit baseline state on this branch
    git_commit_change(
        worktree_path, f"[Run {run_number}] Baseline: {metric} {baseline_score:.4f}"
    )

    # Initialize crew
    crew = PharmaAgentsCrew()

    # Check if we need to run the Archivist (literature research)
    # Runs on: (1) first time (no literature exists), or (2) exploration mode (stuck)
    # NOTE: Literature is stored in main repo, not worktree
    literature_dir = main_experiments_dir / "literature"
    literature_index = literature_dir / "index.json"
    no_literature = not literature_index.exists()
    exploration_mode = memory.is_globally_stagnant() or memory.is_stuck(run_number)

    run_archivist = no_literature or exploration_mode
    archivist_reason = (
        "No literature database"
        if no_literature
        else "EXPLORATION MODE - gathering fresh papers"
        if exploration_mode
        else None
    )

    # Decide which crew to use for first iteration
    # Archivist runs async in parallel on first iteration when needed
    use_archivist_first_iter = run_archivist and archivist_reason is not None

    if use_archivist_first_iter:
        logger.info("=" * 60)
        logger.info(f"ARCHIVIST (async): {archivist_reason}")
        logger.info("=" * 60)
        print(f"[ARCHIVIST] Will run async in first iteration ({archivist_reason})...")
    else:
        logger.info(
            "Literature database exists and not in exploration mode - skipping Archivist"
        )

    # Track experiment history for context
    experiment_history: list[dict] = []

    def _reset_tool_counters() -> None:
        """Reset per-iteration rate limits on all tools."""
        from .tools.arxiv import AlphaxivTool, ArxivSearchTool
        from .tools.literature import FetchMorePapersTool
        from .tools.training import InstallPackageTool
        from .tools.skills import SkillLoaderTool

        AlphaxivTool.reset_counters()
        ArxivSearchTool.reset_counters()
        FetchMorePapersTool.reset_counters()
        InstallPackageTool.reset_counters()
        SkillLoaderTool.reset_counters()

    # Run iterations
    for i in range(iterations):
        _reset_tool_counters()
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"ITERATION {i + 1}/{iterations}")
        logger.info("=" * 60)
        logger.info(f"Current best {metric}: {best_score:.4f}")

        # Prepare inputs for the crew
        from .memory import get_metric_direction, get_baseline_config

        config = get_baseline_config()
        inputs = {
            "property": config.get("property", "molecular property"),
            "metric": metric,
            "direction": get_metric_direction(),
            "baseline_score": f"{best_score:.4f}",
            "experiment_history": json.dumps(experiment_history[-5:], indent=2)
            if experiment_history
            else "No previous experiments in this run",
            "agent_memory": memory.format_for_prompt(run_number),
        }

        # Run the crew (capture stdout to log file)
        # First iteration: use crew_with_archivist if needed (archivist runs async)
        # Subsequent iterations: use regular crew
        try:
            if i == 0 and use_archivist_first_iter:
                logger.info("Running crew with async archivist...")
                with capture_stdout_to_log(log_file):
                    result = crew.crew_with_archivist().kickoff(inputs=inputs)
            else:
                logger.info("Running agent crew...")
                with capture_stdout_to_log(log_file):
                    result = crew.crew().kickoff(inputs=inputs)
            logger.info("Crew completed")
            logger.debug(f"Crew output: {result}")
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Crew error: {e}")
            git_revert_changes(worktree_path)

            # Permanent errors — abort all iterations
            permanent_markers = [
                "api key",
                "authentication",
                "unauthorized",
                "403",
                "invalid model",
                "model not found",
                "permission denied",
            ]
            if any(marker in error_msg for marker in permanent_markers):
                logger.error("PERMANENT ERROR detected — aborting run")
                break

            # Rate limit — wait and retry
            rate_limit_markers = ["rate limit", "429", "quota", "resource exhausted"]
            if any(marker in error_msg for marker in rate_limit_markers):
                logger.warning("Rate limit hit — waiting 60s before next iteration")
                time.sleep(60)

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

        # Parse hypothesis from log file (hypothesis_agent outputs PROPOSAL/REASONING)
        hypothesis, reasoning = parse_hypothesis_from_log(log_file)

        # Log prominent iteration result
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"ITERATION {i + 1} RESULT")
        logger.info("-" * 60)

        if eval_result.success and eval_result.rmse is not None:
            if is_better(eval_result.rmse, best_score):
                improvement = compute_improvement_pct(best_score, eval_result.rmse)
                logger.success(f"{metric}: {best_score:.4f} -> {eval_result.rmse:.4f}")
                logger.success(f"IMPROVEMENT: {improvement:.1f}% better - KEEPING")

                # Save to memory
                memory.add_experiment(
                    run=run_number,
                    iteration=i + 1,
                    hypothesis=hypothesis,
                    reasoning=reasoning,
                    result="success",
                    score_before=best_score,
                    score_after=eval_result.rmse,
                    insight=f"Achieved {improvement:.1f}% improvement",
                )

                best_score = eval_result.rmse
                # Commit the improvement
                git_commit_change(
                    worktree_path,
                    f"[Run {run_number}] Iter {i + 1}: {metric} {eval_result.rmse:.4f} ({improvement:.1f}% better)",
                )
            else:
                logger.warning(f"{metric}: {best_score:.4f} -> {eval_result.rmse:.4f}")
                logger.warning("NO IMPROVEMENT - REVERTING")

                # Save failure to memory
                memory.add_experiment(
                    run=run_number,
                    iteration=i + 1,
                    hypothesis=hypothesis,
                    reasoning=reasoning,
                    result="failure",
                    score_before=best_score,
                    score_after=eval_result.rmse,
                    insight=f"No improvement ({metric} {eval_result.rmse:.4f} vs baseline {best_score:.4f})",
                )

                git_revert_changes(worktree_path)
        else:
            logger.error(f"TRAINING FAILED: {eval_result.error}")
            logger.error("REVERTING")

            # Save error to memory
            memory.add_experiment(
                run=run_number,
                iteration=i + 1,
                hypothesis=hypothesis,
                reasoning=reasoning,
                result="failure",
                score_before=best_score,
                score_after=None,
                insight=f"Training failed: {eval_result.error}",
            )

            git_revert_changes(worktree_path)

    # Finalize run with conclusion for future researchers
    conclusion = memory.finalize_run(run_number)
    logger.info(f"Run conclusion: {conclusion}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Initial {metric}:  {baseline_score:.4f}")
    logger.info(f"Final {metric}:    {best_score:.4f}")
    total_improvement = compute_improvement_pct(baseline_score, best_score)
    logger.success(f"Total Improvement: {total_improvement:.1f}%")
    logger.info(f"Experiments:   {len(experiment_history)}")
    logger.info(f"Log dir:       {run_log_dir}")

    # Generate HTML report with charts
    from .report import generate_run_report
    from .memory import get_metric_direction

    report_path = generate_run_report(
        run_id=run_number,
        experiment_name=experiment_name,
        metric=metric,
        direction=get_metric_direction(),
        baseline_score=baseline_score,
        experiments=experiment_history,
        output_dir=run_log_dir,
    )
    logger.info(f"Report:        {report_path}")

    # Print summary to stdout
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Initial {metric}:      {baseline_score:.4f}")
    print(f"Final {metric}:        {best_score:.4f}")
    print(f"Total Improvement: {total_improvement:.1f}%")
    print(f"Experiments:       {len(experiment_history)}")
    print(f"Log file:          {log_file}")
    print(f"Report:            {report_path}")

    # Show git log (from worktree to see run-specific commits)
    logger.info("")
    logger.info(f"Git history ({branch_name}):")
    result = subprocess.run(
        ["git", "log", "--oneline", "-10"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.strip().split("\n"):
        logger.info(f"  {line}")

    logger.info("")
    logger.info("To promote this run as new baseline:")
    logger.info(f"  uv run python -m pharma_agents.promote {run_number}")


def promote(run_number: int, experiment: str | None = None) -> None:
    """
    Promote a run branch to main, making it the new baseline.

    This merges run/EXPERIMENT/XXX into main and updates baseline_train.py + baseline.json.
    """
    from .memory import get_experiment_name

    project_root = Path(__file__).parent.parent.parent.resolve()
    experiment = experiment or get_experiment_name()
    branch_name = f"run/{experiment}/{run_number:03d}"

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

    experiments_dir = project_root / "experiments" / experiment
    train = experiments_dir / "train.py"
    baseline = experiments_dir / "baseline_train.py"
    baseline_json = experiments_dir / "baseline.json"
    shutil.copy(train, baseline)

    # Get new baseline score
    from .tools.evaluate import run_training

    eval_result = run_training()

    # Update baseline.json - preserve metric config, update score
    existing_config = json.loads(baseline_json.read_text())

    baseline_data = {
        "score": eval_result.score,
        "metric": existing_config.get("metric", "RMSE"),
        "direction": existing_config.get("direction", "lower_is_better"),
        "description": f"Promoted from {branch_name}",
        "updated": datetime.now().strftime("%Y-%m-%d"),
    }
    baseline_json.write_text(json.dumps(baseline_data, indent=2))

    # Commit the new baseline
    subprocess.run(
        ["git", "add", str(baseline), str(baseline_json)],
        cwd=project_root,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", f"Update baseline from {branch_name}"],
        cwd=project_root,
        capture_output=True,
    )

    logger.success(f"Promoted {branch_name} to main!")
    logger.info(f"New baseline {eval_result.metric}: {eval_result.rmse:.4f}")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Run pharma-agents experiment")
    parser.add_argument(
        "--experiment",
        "-e",
        default=os.environ.get("PHARMA_EXPERIMENT", "bbbp"),
        help="Experiment to run: 'bbbp' or 'solubility' (default: bbbp)",
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=int(os.environ.get("MAX_ITERATIONS", "5")),
        help="Number of iterations (default: 5 or MAX_ITERATIONS env var)",
    )
    args = parser.parse_args()

    # Set experiment before importing anything that uses it
    os.environ["PHARMA_EXPERIMENT"] = args.experiment
    print(f"Running experiment: {args.experiment}")

    run(iterations=args.iterations)
