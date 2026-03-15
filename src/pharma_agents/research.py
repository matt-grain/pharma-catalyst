"""Run archivist-only literature research for an experiment."""

import json
import os
from pathlib import Path

from loguru import logger

from .crew import PharmaAgentsCrew
from .main import setup_logging, capture_stdout_to_log
from .memory import get_experiment_name, validate_experiment, get_baseline_config


def research(experiment: str | None = None) -> None:
    """
    Run the Archivist agent to build/refresh the literature database.

    This is a standalone command — no worktree, no training, no model changes.
    Use it to prepare the literature before a run, or to let a human DS
    review what the archivist found.
    """
    experiment = experiment or get_experiment_name()
    os.environ["PHARMA_EXPERIMENT"] = experiment
    validate_experiment(experiment)

    project_root = Path(__file__).parent.parent.parent.resolve()
    experiments_dir = project_root / "experiments" / experiment
    literature_dir = experiments_dir / "literature"

    # Setup logging
    research_log_dir = experiments_dir / "research"
    research_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(research_log_dir)

    # Set experiments dir for tools (archivist uses get_literature_dir()
    # which reads from main experiments root, not PHARMA_EXPERIMENTS_DIR)
    os.environ["PHARMA_EXPERIMENTS_DIR"] = str(experiments_dir.resolve())

    config = get_baseline_config()
    prop = config.get("property", "molecular property")

    # Check existing literature
    index_path = literature_dir / "index.json"
    existing_papers = 0
    existing_titles: list[str] = []
    if index_path.exists():
        index = json.loads(index_path.read_text())
        papers = index.get("papers", {})
        existing_papers = len(papers)
        existing_titles = [
            f"- {pid}: {p.get('title', 'untitled')}" for pid, p in papers.items()
        ]

    print(f"\n{'=' * 60}")
    print(f"LITERATURE RESEARCH: {experiment}")
    print(f"Property: {prop}")
    print(f"Existing papers: {existing_papers}")
    print(f"Log: {log_file}")
    print(f"{'=' * 60}\n")

    logger.info(f"Starting literature research for {experiment}")
    logger.info(f"Property: {prop}")
    logger.info(f"Existing papers in database: {existing_papers}")

    # Run archivist-only crew
    crew = PharmaAgentsCrew()
    inputs = {
        "property": prop,
        "metric": config.get("metric", "RMSE"),
        "direction": config.get("direction", "lower_is_better"),
        "baseline_score": str(config.get("score", 0)),
        "experiment_history": "N/A (research mode)",
        "agent_memory": "N/A (research mode)",
        "existing_papers": "\n".join(existing_titles) if existing_titles else "None",
    }

    try:
        with capture_stdout_to_log(log_file):
            result = crew.archivist_crew().kickoff(inputs=inputs)
        logger.info("Archivist completed")
    except Exception as e:
        logger.error(f"Archivist failed: {e}")
        print(f"\nERROR: {e}")
        return

    # Report results
    new_papers = 0
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
            new_papers = len(index.get("papers", {}))
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Literature index corrupted by concurrent writes — rebuilding"
            )
            new_papers = 0

    added = new_papers - existing_papers
    print(f"\n{'=' * 60}")
    print("RESEARCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"Papers before: {existing_papers}")
    print(f"Papers after:  {new_papers} (+{added})")
    print(f"Literature DB: {literature_dir}")
    print(f"\nBrowse papers: {literature_dir / 'papers'}")
    print(f"Search index:  {index_path}")

    logger.success(f"Research complete: {new_papers} papers ({added} new)")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run archivist to build literature database"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        default=os.environ.get("PHARMA_EXPERIMENT", "bbbp"),
        help="Experiment name (default: bbbp)",
    )
    args = parser.parse_args()

    research(experiment=args.experiment)
