"""Custom CrewAI tools for pharma-agents."""

from crewai.tools import BaseTool
from pathlib import Path


class WriteTrainPyTool(BaseTool):
    """Tool to write the train.py file."""

    name: str = "write_train_py"
    description: str = (
        "Writes the complete content to train.py. "
        "Input must be the FULL Python code for the train.py file. "
        "The file will be completely overwritten."
    )

    def _run(self, content: str) -> str:
        """Write content to train.py."""
        train_path = Path(__file__).parent / "train.py"
        try:
            train_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to train.py"
        except Exception as e:
            return f"Error writing to train.py: {e}"


class RunTrainPyTool(BaseTool):
    """Tool to run train.py and get RMSE."""

    name: str = "run_train_py"
    description: str = (
        "Runs train.py and returns the validation RMSE. "
        "No input required. Returns RMSE value or error message."
    )

    def _run(self, _: str = "") -> str:
        """Run train.py and return RMSE."""
        import subprocess
        import sys

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from pharma_agents.tools.train import train; print(f'RMSE:{train(verbose=False):.4f}')",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path(__file__).parent.parent.parent.parent,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("RMSE:"):
                        return line
                return f"Output: {result.stdout}"
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Training timed out (>60s)"
        except Exception as e:
            return f"Error: {e}"
