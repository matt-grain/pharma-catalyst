"""
Persistent memory for pharma-agents.

Stores experiment history across runs so agents can learn from past attempts.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional


def get_experiments_dir() -> Path:
    """Get the experiments directory path.

    Can be overridden via PHARMA_EXPERIMENTS_DIR env var (used for worktrees).
    """
    override = os.environ.get("PHARMA_EXPERIMENTS_DIR")
    if override:
        return Path(override)
    return Path(__file__).parent.parent.parent / "experiments"


def get_baseline_config() -> dict:
    """Load baseline config from baseline.json."""
    baseline_path = get_experiments_dir() / "baseline.json"
    return json.loads(baseline_path.read_text())


def get_baseline_score() -> float:
    """Load baseline score from baseline.json."""
    return float(get_baseline_config().get("score", 1.3175))


def get_metric_name() -> str:
    """Get the metric name (e.g., RMSE, MAE, accuracy)."""
    return get_baseline_config().get("metric", "RMSE")


@dataclass
class Experiment:
    """A single experiment record with full context."""

    iteration: int
    timestamp: str

    # The hypothesis (parsed from crew output)
    hypothesis: str  # What change was proposed
    reasoning: str  # WHY this direction was taken

    # The result
    result: str  # "success" or "failure"
    rmse_before: float
    rmse_after: Optional[float]
    improvement_pct: Optional[float]

    # The learning
    insight: str  # What we learned (especially WHY it failed/succeeded)


@dataclass
class RunMemory:
    """Memory for a single run."""

    run_id: int
    start_time: str
    experiments: list[Experiment] = field(default_factory=list)
    best_rmse: float = field(default_factory=get_baseline_score)
    consecutive_failures: int = 0
    # Run conclusion - set when run ends
    # LOCAL_OPTIMUM: diminishing returns, need radical change
    # PROGRESS_CONTINUING: still improving when iterations ran out
    # STUCK: couldn't make meaningful progress
    conclusion: str = ""
    conclusion_detail: str = ""


class AgentMemory:
    """Persistent memory store for agent learnings."""

    def __init__(self, memory_path: Path):
        self.memory_path = memory_path
        self.runs: dict[int, RunMemory] = {}
        self.global_best_rmse: float = get_baseline_score()
        self.key_learnings: list[str] = []
        self._load()

    def _load(self) -> None:
        """Load memory from disk."""
        if self.memory_path.exists():
            data = json.loads(self.memory_path.read_text())
            baseline = get_baseline_score()

            # Load runs
            for run_id_str, run_data in data.get("runs", {}).items():
                run_id = int(run_id_str)
                experiments = [
                    Experiment(**exp) for exp in run_data.get("experiments", [])
                ]
                self.runs[run_id] = RunMemory(
                    run_id=run_id,
                    start_time=run_data.get("start_time", ""),
                    experiments=experiments,
                    best_rmse=run_data.get("best_rmse", baseline),
                    consecutive_failures=run_data.get("consecutive_failures", 0),
                    conclusion=run_data.get("conclusion", ""),
                    conclusion_detail=run_data.get("conclusion_detail", ""),
                )

            self.global_best_rmse = data.get("global_best_rmse", baseline)
            self.key_learnings = data.get("key_learnings", [])

    def save(self) -> None:
        """Save memory to disk."""
        self.memory_path.parent.mkdir(exist_ok=True)
        data = {
            "runs": {
                str(run_id): {
                    "run_id": run.run_id,
                    "start_time": run.start_time,
                    "experiments": [asdict(exp) for exp in run.experiments],
                    "best_rmse": run.best_rmse,
                    "consecutive_failures": run.consecutive_failures,
                    "conclusion": run.conclusion,
                    "conclusion_detail": run.conclusion_detail,
                }
                for run_id, run in self.runs.items()
            },
            "global_best_rmse": self.global_best_rmse,
            "key_learnings": self.key_learnings,
            "last_updated": datetime.now().isoformat(),
        }
        self.memory_path.write_text(json.dumps(data, indent=2))

    def start_run(self, run_id: int) -> None:
        """Initialize a new run."""
        if run_id not in self.runs:
            self.runs[run_id] = RunMemory(
                run_id=run_id,
                start_time=datetime.now().isoformat(),
            )
            self.save()

    def add_experiment(
        self,
        run: int,
        iteration: int,
        hypothesis: str,
        reasoning: str,
        result: str,
        rmse_before: float,
        rmse_after: Optional[float],
        insight: str,
    ) -> None:
        """Record an experiment."""
        # Ensure run exists
        if run not in self.runs:
            self.start_run(run)

        run_memory = self.runs[run]

        # Calculate improvement
        improvement = None
        if rmse_after and rmse_before:
            improvement = ((rmse_before - rmse_after) / rmse_before) * 100

        exp = Experiment(
            iteration=iteration,
            timestamp=datetime.now().isoformat(),
            hypothesis=hypothesis,
            reasoning=reasoning,
            result=result,
            rmse_before=rmse_before,
            rmse_after=rmse_after,
            improvement_pct=improvement,
            insight=insight,
        )
        run_memory.experiments.append(exp)

        # Track consecutive failures
        if result == "success":
            run_memory.consecutive_failures = 0
            if rmse_after and rmse_after < run_memory.best_rmse:
                run_memory.best_rmse = rmse_after
            if rmse_after and rmse_after < self.global_best_rmse:
                self.global_best_rmse = rmse_after
        else:
            run_memory.consecutive_failures += 1

        self.save()

    def add_learning(self, learning: str) -> None:
        """Add a key learning."""
        if learning not in self.key_learnings:
            self.key_learnings.append(learning)
            self.save()

    def finalize_run(self, run_id: int) -> str:
        """
        Analyze run trajectory and set conclusion.

        Returns the conclusion for logging.
        """
        if run_id not in self.runs:
            return ""

        run = self.runs[run_id]
        if not run.experiments:
            run.conclusion = "NO_EXPERIMENTS"
            run.conclusion_detail = "Run ended with no experiments completed."
            self.save()
            return run.conclusion

        # Analyze the trajectory
        successes = [e for e in run.experiments if e.result == "success"]
        failures = [e for e in run.experiments if e.result == "failure"]
        total = len(run.experiments)

        # Check last 3 experiments for trend
        recent = run.experiments[-3:] if len(run.experiments) >= 3 else run.experiments
        recent_successes = [e for e in recent if e.result == "success"]
        recent_improvements = [
            e.improvement_pct for e in recent_successes if e.improvement_pct is not None
        ]

        # Determine conclusion
        metric = get_metric_name()
        if len(failures) == total:
            run.conclusion = "STUCK"
            run.conclusion_detail = (
                "No successful experiments. This direction may be fundamentally flawed. "
                "Future runs should try completely different approaches."
            )
        elif run.consecutive_failures >= 2:
            run.conclusion = "LOCAL_OPTIMUM"
            run.conclusion_detail = (
                f"Ended with {run.consecutive_failures} consecutive failures after "
                f"reaching {metric} {run.best_rmse:.4f}. Likely hit diminishing returns. "
                "Future runs should explore radically different directions."
            )
        elif recent_improvements and max(recent_improvements) < 1.0:
            run.conclusion = "LOCAL_OPTIMUM"
            run.conclusion_detail = (
                f"Recent improvements < 1%. Reached {metric} {run.best_rmse:.4f}. "
                "Incremental changes exhausted. Try different model families or features."
            )
        elif len(successes) > 0:
            avg_improvement = sum(
                e.improvement_pct for e in successes if e.improvement_pct
            ) / len(successes)
            run.conclusion = "PROGRESS_CONTINUING"
            run.conclusion_detail = (
                f"Still making progress (avg {avg_improvement:.1f}% per success). "
                f"Reached {metric} {run.best_rmse:.4f}. More iterations could yield improvements."
            )
        else:
            run.conclusion = "INCONCLUSIVE"
            run.conclusion_detail = "Mixed results. Review experiments for insights."

        self.save()
        return run.conclusion

    def get_all_experiments(self) -> list[Experiment]:
        """Get all experiments across all runs."""
        all_exps = []
        for run in self.runs.values():
            all_exps.extend(run.experiments)
        return all_exps

    def get_successful_experiments(self) -> list[Experiment]:
        """Get all successful experiments."""
        return [e for e in self.get_all_experiments() if e.result == "success"]

    def get_failed_experiments(self) -> list[Experiment]:
        """Get all failed experiments."""
        return [e for e in self.get_all_experiments() if e.result == "failure"]

    def is_stuck(self, run: int, threshold: int = 3) -> bool:
        """Check if current run is stuck (consecutive failures)."""
        if run not in self.runs:
            return False
        return self.runs[run].consecutive_failures >= threshold

    def is_globally_stagnant(self, threshold: int = 5) -> bool:
        """Check if global best hasn't improved in last N experiments."""
        all_exps = self.get_all_experiments()
        if len(all_exps) < threshold:
            return False

        # Get last N experiments
        recent = all_exps[-threshold:]

        # Check if any improved the global best
        for exp in recent:
            if exp.result == "success" and exp.rmse_after:
                if exp.rmse_after <= self.global_best_rmse:
                    return False  # Recent improvement exists

        return True  # No improvements in last N experiments

    def format_for_prompt(self, current_run: int) -> str:
        """Format memory as context for the hypothesis agent."""
        all_experiments = self.get_all_experiments()
        if not all_experiments:
            return "No previous experiments. This is a fresh start."

        metric = get_metric_name()
        lines = [
            f"## Agent Memory (Global Best {metric}: {self.global_best_rmse:.4f})",
            "",
        ]

        # Check if stuck - suggest exploration
        is_run_stuck = self.is_stuck(current_run)
        is_globally_stuck = self.is_globally_stagnant()

        if is_run_stuck or is_globally_stuck:
            lines.append("### EXPLORATION MODE REQUIRED")
            if is_globally_stuck:
                lines.append(
                    "**GLOBAL STAGNATION: No improvement in recent experiments. "
                    "Current approaches have been exhausted.**"
                )
            else:
                lines.append(
                    "**RUN STUCK: Multiple consecutive failures in this run.**"
                )
            lines.append("")
            lines.append(
                "You MUST propose something RADICALLY DIFFERENT from what's been tried."
            )
            lines.append(
                "Look at 'What Failed' and 'What Worked' - then think outside that box."
            )
            lines.append(
                "Consider: different model families, different feature representations,"
            )
            lines.append(
                "simplification instead of complexity, or entirely new scientific angles."
            )
            lines.append("")

        # Notes from previous runs (conclusions)
        past_runs = [
            r for r in self.runs.values() if r.run_id != current_run and r.conclusion
        ]
        if past_runs:
            lines.append("### Notes from Previous Researchers")
            for run in sorted(past_runs, key=lambda r: r.run_id)[-3:]:  # Last 3 runs
                icon = {
                    "LOCAL_OPTIMUM": "⚠️",
                    "STUCK": "🚫",
                    "PROGRESS_CONTINUING": "📈",
                }.get(run.conclusion, "📝")
                lines.append(
                    f"- **Run {run.run_id}** [{icon} {run.conclusion}]: {run.conclusion_detail}"
                )
            lines.append("")

        # Key learnings
        if self.key_learnings:
            lines.append("### Key Learnings")
            for learning in self.key_learnings[-5:]:
                lines.append(f"- {learning}")
            lines.append("")

        # Successful experiments (deduplicated by hypothesis)
        successes = self.get_successful_experiments()
        seen_hypotheses = set()
        if successes:
            lines.append("### What Worked")
            for exp in successes[-10:]:
                if exp.hypothesis in seen_hypotheses:
                    continue
                seen_hypotheses.add(exp.hypothesis)
                pct = exp.improvement_pct or 0
                lines.append(f"- **{exp.hypothesis}** (+{pct:.1f}%)")
                if exp.reasoning and exp.reasoning != "See crew output for details":
                    lines.append(f"  Reasoning: {exp.reasoning}")
                lines.append(f"  Insight: {exp.insight}")
            lines.append("")

        # Failed experiments (deduplicated, important to avoid repeating!)
        failures = self.get_failed_experiments()
        seen_hypotheses = set()
        if failures:
            lines.append("### What Failed (DO NOT REPEAT)")
            for exp in failures[-10:]:
                if exp.hypothesis in seen_hypotheses:
                    continue
                seen_hypotheses.add(exp.hypothesis)
                lines.append(f"- **{exp.hypothesis}**")
                if exp.reasoning and exp.reasoning != "See crew output for details":
                    lines.append(f"  Reasoning: {exp.reasoning}")
                lines.append(f"  Why it failed: {exp.insight}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def parse_crew_output(output: str) -> tuple[str, str]:
        """Extract hypothesis and reasoning from crew output."""
        hypothesis = "Unknown change"
        reasoning = "No reasoning provided"

        # Try to parse PROPOSAL: line
        proposal_match = re.search(
            r"PROPOSAL:\s*(.+?)(?:\n|REASONING:|CHANGE:|$)",
            output,
            re.IGNORECASE | re.DOTALL,
        )
        if proposal_match:
            hypothesis = proposal_match.group(1).strip()

        # Try to parse REASONING: line
        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?:\n\n|CHANGE:|PROPOSAL:|$)",
            output,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return hypothesis, reasoning
