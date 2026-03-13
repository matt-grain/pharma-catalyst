"""Promote a run branch to main, making it the new baseline."""

import sys
from .main import promote

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python -m pharma_agents.promote <run_number>")
        print("Example: uv run python -m pharma_agents.promote 1")
        sys.exit(1)

    try:
        run_number = int(sys.argv[1])
    except ValueError:
        print(f"Invalid run number: {sys.argv[1]}")
        sys.exit(1)

    promote(run_number)
