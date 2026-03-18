"""Training script manipulation tools."""

import re
import subprocess
import sys
from typing import Callable, ClassVar

from crewai.tools import BaseTool

from pharma_agents.memory import get_experiments_dir, get_metric_name
from pharma_agents.tool_config import get_allowed_packages, get_dangerous_patterns


class ReadTrainPyTool(BaseTool):
    """Tool to read the train.py file (full, outline, or section)."""

    name: str = "read_train_py"
    description: str = (
        "Reads train.py content. Three modes:\n"
        "- 'read' or 'full': returns the full file with line numbers\n"
        "- 'outline': returns function/class signatures with line numbers "
        "(use this first on large files to find what to edit)\n"
        "- 'lines 20-50': returns only lines 20 through 50 "
        "(use after outline to read a specific section)"
    )

    def _run(self, argument: str = "read") -> str:
        """Read train.py content with optional mode."""
        train_path = get_experiments_dir() / "train.py"
        try:
            content = train_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: train.py not found at {train_path}"
        except Exception as e:
            return f"Error reading train.py: {e}"

        lines = content.split("\n")
        arg = argument.strip().lower()

        # Outline mode: show structure with line numbers
        if arg == "outline":
            return self._outline(lines)

        # Line range mode: "lines 20-50" or "20-50"
        range_match = re.match(r"(?:lines?\s*)?(\d+)\s*[-–]\s*(\d+)", arg)
        if range_match:
            start = max(1, int(range_match.group(1)))
            end = min(len(lines), int(range_match.group(2)))
            selected = lines[start - 1 : end]
            numbered = [f"{i:4d} | {line}" for i, line in enumerate(selected, start)]
            return (
                f"train.py lines {start}-{end} (of {len(lines)} total):\n"
                + "\n".join(numbered)
            )

        # Full mode (default): return with line numbers
        numbered = [f"{i:4d} | {line}" for i, line in enumerate(lines, 1)]
        return f"train.py ({len(lines)} lines):\n" + "\n".join(numbered)

    @staticmethod
    def _outline(lines: list[str]) -> str:
        """Extract function/class signatures and key structure."""
        outline_lines: list[str] = []
        total = len(lines)

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            is_top_level = not line.startswith(" ") and not line.startswith("\t")
            # Top-level imports block
            if is_top_level and (
                stripped.startswith("import ") or stripped.startswith("from ")
            ):
                if not outline_lines or not outline_lines[-1].startswith("  imports"):
                    outline_lines.append(f"  imports: L{i}")
                else:
                    prev = outline_lines[-1]
                    start_match = re.search(r"L(\d+)", prev)
                    if start_match:
                        outline_lines[-1] = f"  imports: L{start_match.group(1)}-L{i}"
            # Function/class definitions (any level)
            elif stripped.startswith("def "):
                indent = "    " if not is_top_level else ""
                sig = stripped.rstrip(":")
                outline_lines.append(f"  L{i:4d}: {indent}{sig}")
            elif stripped.startswith("class "):
                sig = stripped.rstrip(":")
                outline_lines.append(f"  L{i:4d}: {sig}")
            # Top-level assignments (constants, config)
            elif is_top_level and re.match(r"^[A-Z_]+ = ", stripped):
                outline_lines.append(f"  L{i:4d}: {stripped[:60]}")

        return (
            f"train.py outline ({total} lines total):\n"
            + "\n".join(outline_lines)
            + "\n\nUse 'lines N-M' to read a specific section."
        )


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

        # Fix double-encoded escape sequences (LLM writes \\n instead of \n
        # when the code passes through JSON serialization twice)
        if "\\n" in content and "\n" not in content:
            content = content.replace("\\n", "\n")
        if "\\t" in content and "\t" not in content:
            content = content.replace("\\t", "\t")

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
        for pattern in get_dangerous_patterns():
            if pattern in content:
                return f"Error: Dangerous pattern '{pattern}' not allowed in train.py."

        # Warn if features are computed but inf/NaN not handled
        warns = ""
        has_feature_computation = any(
            p in content
            for p in ["feature", "descriptor", "fingerprint", "log(", "log2(", "/ "]
        )
        has_nan_handling = any(
            p in content
            for p in ["nan_to_num", "fillna", "replace([np.inf", "np.isinf", "dropna"]
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


class EditTrainPyTool(BaseTool):
    """Tool to make targeted edits to train.py without rewriting the whole file."""

    name: str = "edit_train_py"
    description: str = (
        "Makes a targeted edit to train.py by replacing a specific text snippet. "
        "Provide 'old_text' (exact text to find) and 'new_text' (replacement). "
        "Only the first match is replaced. Use this instead of write_train_py "
        "when you only need to change a small part of the file."
    )

    def _run(self, old_text: str, new_text: str) -> str:
        """Replace old_text with new_text in train.py."""
        train_path = get_experiments_dir() / "train.py"
        try:
            content = train_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: train.py not found at {train_path}"

        if old_text not in content:
            # Show nearby lines to help the LLM find the right text
            lines = content.split("\n")
            # Find closest match: check if the key identifier (before =, (, etc.) appears
            first_line = old_text.strip().split("\n")[0].strip()
            # Extract the variable/function name (left side of = or first word)
            key = first_line.split("=")[0].strip().split("(")[0].strip()
            matches = [
                f"  L{i + 1}: {line.rstrip()}"
                for i, line in enumerate(lines)
                if key and key in line
            ]
            hint = "\n".join(matches[:5]) if matches else ""
            return (
                f"Error: old_text not found in train.py. "
                f"Make sure it matches exactly (including whitespace)."
                f"{chr(10) + 'Similar lines:' + chr(10) + hint if hint else ''}"
            )

        count = content.count(old_text)
        if count > 1:
            return (
                f"Error: old_text appears {count} times in train.py. "
                f"Provide more context to make the match unique."
            )

        new_content = content.replace(old_text, new_text, 1)

        # Block dangerous patterns in new_text
        for pattern in get_dangerous_patterns():
            if pattern in new_text:
                return f"Error: Dangerous pattern '{pattern}' not allowed."

        try:
            train_path.write_text(new_content, encoding="utf-8")
            old_lines = len(old_text.strip().split("\n"))
            new_lines = len(new_text.strip().split("\n"))
            return (
                f"Successfully edited train.py: replaced {old_lines} lines "
                f"with {new_lines} lines ({len(new_content)} chars total)."
            )
        except Exception as e:
            return f"Error writing to train.py: {e}"


class SearchTrainPyTool(BaseTool):
    """Tool to search for patterns in train.py."""

    name: str = "search_train_py"
    description: str = (
        "Searches train.py for a text pattern (supports regex). "
        "Returns matching lines with line numbers. "
        "Use this to find where variables, functions, or patterns are used "
        "before making edits with edit_train_py."
    )

    def _run(self, pattern: str) -> str:
        """Search train.py for a pattern."""
        import re

        train_path = get_experiments_dir() / "train.py"
        try:
            content = train_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: train.py not found at {train_path}"

        lines = content.split("\n")
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Fall back to literal search if regex is invalid
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        matches: list[str] = []
        for i, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"  L{i:4d}: {line.rstrip()}")

        if not matches:
            return f"No matches for '{pattern}' in train.py ({len(lines)} lines)."

        return f"Found {len(matches)} match(es) for '{pattern}':\n" + "\n".join(matches)


class CodeCheckTool(BaseTool):
    """Tool to check train.py for syntax and linting errors."""

    name: str = "code_check"
    description: str = (
        "Runs ruff (linter) on train.py to check for syntax errors and issues. "
        "Returns 'OK' if no issues, or lists errors to fix. "
        "Pass any string (e.g. 'check'). "
        "ALWAYS run this AFTER writing code to ensure it will run correctly."
    )
    cache_function: Callable = lambda _args, _result: False  # type: ignore[assignment]

    def _run(self, argument: str = "check") -> str:
        """Run ruff on train.py."""
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
        "Pass any string (e.g. 'run'). Returns score value or error message."
    )
    cache_function: Callable = lambda _args, _result: False  # type: ignore[assignment]

    def _run(self, argument: str = "run") -> str:
        """Run train.py and return score."""
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
    cache_function: Callable = lambda _args, _result: False  # type: ignore[assignment]

    _packages_installed: ClassVar[list[str]] = []
    max_installs_per_run: int = 3

    @classmethod
    def reset_counters(cls) -> None:
        cls._packages_installed = []

    def _run(self, package: str) -> str:
        """Install a package via uv add."""
        package = package.strip().lower()

        # Safety check - only allow whitelisted packages
        if package not in get_allowed_packages():
            return (
                f"Error: '{package}' is not in the allowed list. "
                f"Allowed packages: {', '.join(sorted(get_allowed_packages()))}"
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
