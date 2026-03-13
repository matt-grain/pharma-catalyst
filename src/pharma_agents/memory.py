"""
Persistent memory for pharma-agents.

Stores experiment history across runs so agents can learn from past attempts.
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ExperimentMemory:
    """A single experiment record with full context."""

    run: int
    iteration: int
    timestamp: str

    # The hypothesis
    hypothesis: str  # What change was proposed
    reasoning: str  # WHY this direction was taken

    # The result
    result: str  # "success" or "failure"
    rmse_before: float
    rmse_after: Optional[float]
    improvement_pct: Optional[float]

    # The learning
    insight: str  # What we learned (especially WHY it failed/succeeded)


class AgentMemory:
    """Persistent memory store for agent learnings."""

    def __init__(self, memory_path: Path):
        self.memory_path = memory_path
        self.experiments: list[ExperimentMemory] = []
        self.best_rmse: float = 1.3175  # baseline
        self.key_learnings: list[str] = []
        self._load()

    def _load(self) -> None:
        """Load memory from disk."""
        if self.memory_path.exists():
            data = json.loads(self.memory_path.read_text())
            self.experiments = [
                ExperimentMemory(**exp) for exp in data.get("experiments", [])
            ]
            self.best_rmse = data.get("best_rmse", 1.3175)
            self.key_learnings = data.get("key_learnings", [])

    def save(self) -> None:
        """Save memory to disk."""
        self.memory_path.parent.mkdir(exist_ok=True)
        data = {
            "experiments": [asdict(exp) for exp in self.experiments],
            "best_rmse": self.best_rmse,
            "key_learnings": self.key_learnings,
            "last_updated": datetime.now().isoformat(),
        }
        self.memory_path.write_text(json.dumps(data, indent=2))

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
        improvement = None
        if rmse_after and rmse_before:
            improvement = ((rmse_before - rmse_after) / rmse_before) * 100

        exp = ExperimentMemory(
            run=run,
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
        self.experiments.append(exp)

        if result == "success" and rmse_after and rmse_after < self.best_rmse:
            self.best_rmse = rmse_after

        self.save()

    def add_learning(self, learning: str) -> None:
        """Add a key learning."""
        if learning not in self.key_learnings:
            self.key_learnings.append(learning)
            self.save()

    def get_successful_experiments(self) -> list[ExperimentMemory]:
        """Get all successful experiments."""
        return [e for e in self.experiments if e.result == "success"]

    def get_failed_experiments(self) -> list[ExperimentMemory]:
        """Get all failed experiments."""
        return [e for e in self.experiments if e.result == "failure"]

    def format_for_prompt(self, max_entries: int = 10) -> str:
        """Format memory as context for the hypothesis agent."""
        if not self.experiments:
            return "No previous experiments. This is a fresh start."

        lines = [
            f"## Agent Memory (Best RMSE achieved: {self.best_rmse:.4f})",
            "",
        ]

        # Key learnings first
        if self.key_learnings:
            lines.append("### Key Learnings")
            for learning in self.key_learnings[-5:]:
                lines.append(f"- {learning}")
            lines.append("")

        # Successful experiments
        successes = self.get_successful_experiments()
        if successes:
            lines.append("### What Worked")
            for exp in successes[-5:]:
                lines.append(f"- **{exp.hypothesis}** (+{exp.improvement_pct:.1f}%)")
                lines.append(f"  Reasoning: {exp.reasoning}")
                lines.append(f"  Insight: {exp.insight}")
            lines.append("")

        # Failed experiments (important to avoid repeating!)
        failures = self.get_failed_experiments()
        if failures:
            lines.append("### What Failed (DO NOT REPEAT)")
            for exp in failures[-5:]:
                lines.append(f"- **{exp.hypothesis}**")
                lines.append(f"  Reasoning: {exp.reasoning}")
                lines.append(f"  Why it failed: {exp.insight}")
            lines.append("")

        return "\n".join(lines)
