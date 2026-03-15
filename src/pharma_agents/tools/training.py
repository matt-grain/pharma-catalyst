"""Training script manipulation tools."""

from typing import ClassVar

from crewai.tools import BaseTool

from pharma_agents.memory import get_experiments_dir, get_metric_name

# Allowed packages for InstallPackageTool (ML/data science packages only)
ALLOWED_PACKAGES = {
    "lightgbm",
    "xgboost",
    "catboost",
    "scikit-learn",
    "sklearn",
    "pandas",
    "numpy",
    "scipy",
    "rdkit",
    "deepchem",
    "torch",
    "pytorch",
    "tensorflow",
    "keras",
    "molfeat",
    "descriptastorus",
    "mordred",
    "pubchempy",
    "chembl-webresource-client",
}


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
        """Write content to train.py with validation."""
        train_path = get_experiments_dir() / "train.py"

        content = content.strip()

        # Strip markdown code fences (common LLM output artifact)
        if content.startswith("```python"):
            content = content[len("```python") :].strip()
        if content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        # Reject empty or too-short content
        if len(content) < 50:
            return "Error: Content too short — provide the COMPLETE train.py file."

        # Must contain core function
        if "def train" not in content:
            return "Error: train.py must contain a 'def train' function."

        # Must have imports
        if "import " not in content:
            return "Error: train.py must have import statements."

        # Block dangerous patterns
        dangerous = [
            "os.system(",
            "subprocess.run(",
            "subprocess.call(",
            "subprocess.Popen(",
            "shutil.rmtree(",
            "__import__(",
            "eval(",
            "exec(",
            "os.remove(",
            "os.rmdir(",
        ]
        for pattern in dangerous:
            if pattern in content:
                return f"Error: Dangerous pattern '{pattern}' not allowed in train.py."

        # Warn if features are computed but inf/NaN not handled
        warns = ""
        has_feature_computation = any(
            p in content for p in ["feature", "descriptor", "fingerprint", "log(", "log2(", "/ "]
        )
        has_nan_handling = any(
            p in content for p in ["nan_to_num", "fillna", "replace([np.inf", "np.isinf", "dropna"]
        )
        if has_feature_computation and not has_nan_handling:
            warns = (
                " WARNING: No inf/NaN handling detected. Add "
                "df.replace([np.inf, -np.inf], np.nan).fillna(0) after feature computation "
                "to prevent XGBoost/LightGBM crashes."
            )

        try:
            train_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to train.py.{warns}"
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
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

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
                timeout=180,
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
            return "Error: Training timed out (>180s). Simplify your model: reduce n_estimators, use fewer features, or avoid grid search."
        except Exception as e:
            return f"Error: {e}"


class InstallPackageTool(BaseTool):
    """Tool to install Python packages via uv."""

    name: str = "install_package"
    description: str = (
        "Installs a Python package using uv. "
        "Input: package name (e.g., 'lightgbm', 'xgboost'). "
        "Only ML/data science packages are allowed. "
        "Use this when you need a package that's not installed."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    _packages_installed: ClassVar[list[str]] = []
    max_installs_per_run: int = 3

    @classmethod
    def reset_counters(cls) -> None:
        cls._packages_installed = []

    def _run(self, package: str) -> str:
        """Install a package via uv add."""
        import subprocess

        package = package.strip().lower()

        # Safety check - only allow whitelisted packages
        if package not in ALLOWED_PACKAGES:
            return (
                f"Error: '{package}' is not in the allowed list. "
                f"Allowed packages: {', '.join(sorted(ALLOWED_PACKAGES))}"
            )

        # Limit installs per run
        if len(InstallPackageTool._packages_installed) >= self.max_installs_per_run:
            return (
                f"Error: Max installs ({self.max_installs_per_run}) reached this run."
            )

        # Check if already installed this run
        if package in InstallPackageTool._packages_installed:
            return f"Package '{package}' was already installed this run."

        try:
            result = subprocess.run(
                ["uv", "add", package],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                InstallPackageTool._packages_installed.append(package)
                return f"Successfully installed '{package}'. You can now import it."
            else:
                error = result.stderr.strip() or result.stdout.strip()
                return f"Error installing '{package}': {error}"
        except subprocess.TimeoutExpired:
            return f"Error: Installation of '{package}' timed out (>120s)"
        except Exception as e:
            return f"Error: {e}"
