"""Rebuild the knowledge base index for an experiment.

Usage:
    uv run python -m pharma_agents.ingest_kb -e bbbp
"""

from __future__ import annotations

import argparse
import os
import sys

from .memory import get_experiment_name, validate_experiment
from .tools.knowledge_base import get_kb_dir, rebuild_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild knowledge base index from source documents."
    )
    parser.add_argument(
        "-e",
        "--experiment",
        default=None,
        help="Experiment name (default: $PHARMA_EXPERIMENT)",
    )
    args = parser.parse_args()

    experiment = args.experiment or get_experiment_name()
    os.environ["PHARMA_EXPERIMENT"] = experiment
    validate_experiment(experiment)

    kb_dir = get_kb_dir()
    if not kb_dir.exists():
        print(f"No knowledge_base directory found at: {kb_dir}")
        sys.exit(1)

    md_count = len(list(kb_dir.glob("**/*.md")))
    csv_count = len(list(kb_dir.glob("**/*.csv")))
    print(f"Rebuilding knowledge base for '{experiment}'...")
    print(f"  Source: {kb_dir}")
    print(f"  Files: {md_count} markdown, {csv_count} CSV")
    print()

    result = rebuild_index(kb_dir)
    print(result)


if __name__ == "__main__":
    main()
