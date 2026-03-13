"""Custom CrewAI tools for pharma-agents."""

from crewai.tools import BaseTool

from pharma_agents.memory import get_experiments_dir, get_metric_name


class ReadTrainPyTool(BaseTool):
    """Tool to read the train.py file."""

    name: str = "read_train_py"
    description: str = (
        "Reads and returns the current content of train.py. "
        "No input required. Returns the full Python code."
    )

    def _run(self, _: str = "") -> str:
        """Read train.py content."""
        train_path = get_experiments_dir() / "train.py"
        try:
            return train_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: train.py not found at {train_path}"
        except Exception as e:
            return f"Error reading train.py: {e}"


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
        train_path = get_experiments_dir() / "train.py"
        try:
            train_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to train.py"
        except Exception as e:
            return f"Error writing to train.py: {e}"


class CodeCheckTool(BaseTool):
    """Tool to check train.py for syntax and linting errors."""

    name: str = "code_check"
    description: str = (
        "Runs ruff (linter) on train.py to check for syntax errors and issues. "
        "Returns 'OK' if no issues, or lists errors to fix. "
        "ALWAYS run this AFTER writing code to ensure it will run correctly."
    )
    cache_function: None = None  # Disable caching - always check fresh file state

    def _run(self, _: str = "") -> str:
        """Run ruff on train.py."""
        import subprocess

        train_path = get_experiments_dir() / "train.py"

        # Run ruff through uv (ruff is a dev dependency, not on system PATH)
        try:
            result = subprocess.run(
                ["uv", "run", "ruff", "check", str(train_path), "--output-format=full"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                # Ruff outputs errors to stdout
                output = result.stdout.strip() or result.stderr.strip()
                if output:
                    return f"ERRORS FOUND - Fix these before finishing:\n{output}"
                return "ERRORS FOUND but no details (exit code non-zero)"
        except FileNotFoundError:
            return "ERROR: uv not found"
        except Exception as e:
            return f"ERROR running ruff: {e}"

        return "OK - No linting errors. Code is ready."


class RunTrainPyTool(BaseTool):
    """Tool to run train.py and get the score."""

    name: str = "run_train_py"
    description: str = (
        "Runs train.py and returns the validation score. "
        "No input required. Returns score value or error message."
    )

    def _run(self, _: str = "") -> str:
        """Run train.py and return score."""
        import subprocess
        import sys

        metric = get_metric_name()
        experiments_dir = get_experiments_dir()

        try:
            result = subprocess.run(
                [sys.executable, str(experiments_dir / "train.py")],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=experiments_dir,
            )
            if result.returncode == 0:
                # Find line with metric name
                for line in result.stdout.strip().split("\n"):
                    if metric in line and ":" in line:
                        return line.strip()
                return f"Output: {result.stdout}"
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Training timed out (>60s)"
        except Exception as e:
            return f"Error: {e}"
