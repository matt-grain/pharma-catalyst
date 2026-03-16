"""Tests for training tools: EditTrainPyTool, WriteTrainPyTool, RunTrainPyTool."""

import json
import os

import pytest

SAMPLE_TRAIN_PY = '''\
"""Training pipeline."""

import numpy as np
from sklearn.metrics import roc_auc_score


def load_data():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y


def train(verbose=True):
    X, y = load_data()
    if verbose:
        print(f"Loaded {len(y)} samples")
        print(f"Features: {X.shape[1]}")
    score = 0.85
    if verbose:
        sep = "=" * 40
        print(sep)
        print(f"ROC_AUC: {score:.4f}")
    return score
'''


@pytest.fixture()
def experiments_dir(tmp_path):
    """Create a temp experiments directory with a train.py."""
    exp_dir = tmp_path / "experiments" / "test"
    exp_dir.mkdir(parents=True)

    # Write sample train.py
    (exp_dir / "train.py").write_text(SAMPLE_TRAIN_PY, encoding="utf-8")

    # Create baseline.json
    baseline = {
        "metric": "ROC_AUC",
        "score": 0.80,
        "direction": "higher_is_better",
        "property": "test",
    }
    (exp_dir / "baseline.json").write_text(json.dumps(baseline))

    # Set env vars for tools
    os.environ["PHARMA_EXPERIMENTS_DIR"] = str(exp_dir)
    os.environ["PHARMA_EXPERIMENT"] = "test"

    yield exp_dir

    os.environ.pop("PHARMA_EXPERIMENTS_DIR", None)
    os.environ.pop("PHARMA_EXPERIMENT", None)


class TestEditTrainPyTool:
    """Tests for the targeted edit tool."""

    def test_simple_replacement(self, experiments_dir):
        from pharma_agents.tools import EditTrainPyTool

        tool = EditTrainPyTool()
        result = tool._run(
            old_text="score = 0.85",
            new_text="score = 0.92",
        )

        assert "Successfully edited" in result
        content = (experiments_dir / "train.py").read_text()
        assert "score = 0.92" in content
        assert "score = 0.85" not in content

    def test_multiline_replacement(self, experiments_dir):
        from pharma_agents.tools import EditTrainPyTool

        tool = EditTrainPyTool()
        result = tool._run(
            old_text="def load_data():\n    X = np.random.rand(100, 10)\n    y = np.random.randint(0, 2, 100)\n    return X, y",
            new_text="def load_data():\n    X = np.random.rand(200, 20)\n    y = np.random.randint(0, 2, 200)\n    return X, y",
        )

        assert "Successfully edited" in result
        content = (experiments_dir / "train.py").read_text()
        assert "rand(200, 20)" in content

    def test_old_text_not_found_shows_hint(self, experiments_dir):
        from pharma_agents.tools import EditTrainPyTool

        tool = EditTrainPyTool()
        result = tool._run(
            old_text="this text does not exist",
            new_text="replacement",
        )

        assert "Error: old_text not found" in result

    def test_old_text_not_found_similar_lines(self, experiments_dir):
        from pharma_agents.tools import EditTrainPyTool

        tool = EditTrainPyTool()
        # Search for something close to an actual line
        result = tool._run(
            old_text="score = 0.99",  # Actual is 0.85
            new_text="score = 1.0",
        )

        assert "Error: old_text not found" in result
        assert "Similar lines" in result
        assert "score = 0.85" in result

    def test_ambiguous_match_rejected(self, experiments_dir):
        from pharma_agents.tools import EditTrainPyTool

        tool = EditTrainPyTool()
        # "return" appears multiple times (return X, y AND return score)
        result = tool._run(
            old_text="return",
            new_text="return None",
        )

        assert "appears" in result
        assert "times" in result

    def test_dangerous_pattern_blocked(self, experiments_dir):
        from pharma_agents.tools import EditTrainPyTool

        tool = EditTrainPyTool()
        result = tool._run(
            old_text="score = 0.85",
            new_text="os.system('rm -rf /')",
        )

        assert "Dangerous pattern" in result
        # Original content unchanged
        content = (experiments_dir / "train.py").read_text()
        assert "score = 0.85" in content

    def test_preserves_rest_of_file(self, experiments_dir):
        from pharma_agents.tools import EditTrainPyTool

        original = (experiments_dir / "train.py").read_text()
        tool = EditTrainPyTool()
        tool._run(old_text="score = 0.85", new_text="score = 0.90")

        modified = (experiments_dir / "train.py").read_text()
        # Only the target line should differ
        assert modified.replace("score = 0.90", "score = 0.85") == original


class TestWriteTrainPyDoubleEncoding:
    """Tests for the double-encoding fix in WriteTrainPyTool."""

    def test_double_encoded_newlines_fixed(self, experiments_dir):
        """When content has \\n but no real newlines, unescape them."""
        from pharma_agents.tools import WriteTrainPyTool

        tool = WriteTrainPyTool()
        # Simulate double-encoded content (all on one line with literal \\n)
        double_encoded = (
            '"""Training."""\\n'
            "import numpy as np\\n"
            "\\n"
            "def train(verbose=True):\\n"
            "    score = 0.85\\n"
            '    print(f"Score: {score:.4f}")\\n'
            "    return score\\n"
        )

        result = tool._run(double_encoded)
        assert "Successfully wrote" in result

        content = (experiments_dir / "train.py").read_text()
        # Should have real newlines, not literal \\n
        assert "\\n" not in content
        assert "\n" in content
        assert "def train" in content

    def test_normal_content_not_mangled(self, experiments_dir):
        """Content with proper newlines should not be double-unescaped."""
        from pharma_agents.tools import WriteTrainPyTool

        tool = WriteTrainPyTool()
        normal_content = SAMPLE_TRAIN_PY

        result = tool._run(normal_content)
        assert "Successfully wrote" in result

        content = (experiments_dir / "train.py").read_text()
        assert "def train" in content
        assert "def load_data" in content

    def test_fstring_through_json_survives(self, experiments_dir):
        """Simulate the exact path: Python code → json.dumps → tool input."""
        from pharma_agents.tools import WriteTrainPyTool

        tool = WriteTrainPyTool()

        # This is what happens when an LLM builds code in Python then
        # sends it through a JSON tool call
        code = (
            '"""Train."""\n'
            "import numpy as np\n\n"
            "def train(verbose=True):\n"
            "    score = 0.85\n"
            "    if verbose:\n"
            '        print(f"Score: {score:.4f}")\n'
            '        print(f"Samples: {100}")\n'
            "    return score\n"
        )

        # json.dumps then json.loads simulates the round-trip
        serialized = json.dumps(code)
        deserialized = json.loads(serialized)

        result = tool._run(deserialized)
        assert "Successfully wrote" in result

        content = (experiments_dir / "train.py").read_text()
        assert 'print(f"Score: {score:.4f}")' in content
        assert "def train" in content

    def test_double_encoded_tabs_fixed(self, experiments_dir):
        """When content has \\t but no real tabs, unescape them."""
        from pharma_agents.tools import WriteTrainPyTool

        tool = WriteTrainPyTool()
        double_encoded = (
            '"""Train."""\\n'
            "import numpy as np\\n"
            "\\n"
            "def train(verbose=True):\\n"
            "\\treturn 0.85\\n"
        )

        result = tool._run(double_encoded)
        assert "Successfully wrote" in result

        content = (experiments_dir / "train.py").read_text()
        assert "\\t" not in content
        assert "\t" in content


class TestSearchTrainPyTool:
    """Tests for the grep/search tool."""

    def test_literal_search(self, experiments_dir):
        from pharma_agents.tools import SearchTrainPyTool

        tool = SearchTrainPyTool()
        result = tool._run("score = 0.85")

        assert "1 match" in result
        assert "score = 0.85" in result
        assert "L" in result  # Line number

    def test_regex_search(self, experiments_dir):
        from pharma_agents.tools import SearchTrainPyTool

        tool = SearchTrainPyTool()
        result = tool._run(r"def \w+")

        assert "match" in result
        assert "def load_data" in result
        assert "def train" in result

    def test_case_insensitive(self, experiments_dir):
        from pharma_agents.tools import SearchTrainPyTool

        tool = SearchTrainPyTool()
        result = tool._run("TRAINING")

        assert "match" in result
        assert "Training" in result  # Finds the docstring

    def test_no_matches(self, experiments_dir):
        from pharma_agents.tools import SearchTrainPyTool

        tool = SearchTrainPyTool()
        result = tool._run("xyznonexistent")

        assert "No matches" in result

    def test_invalid_regex_falls_back_to_literal(self, experiments_dir):
        from pharma_agents.tools import SearchTrainPyTool

        tool = SearchTrainPyTool()
        # Invalid regex (unbalanced bracket) should fall back to literal
        result = tool._run("[invalid")

        assert "No matches" in result  # Won't find literal "[invalid"


class TestReadTrainPyModes:
    """Tests for ReadTrainPyTool outline and line-range modes."""

    def test_outline_mode(self, experiments_dir):
        from pharma_agents.tools import ReadTrainPyTool

        tool = ReadTrainPyTool()
        result = tool._run("outline")

        assert "outline" in result
        assert "def train" in result
        assert "def load_data" in result
        assert "lines total" in result

    def test_line_range(self, experiments_dir):
        from pharma_agents.tools import ReadTrainPyTool

        tool = ReadTrainPyTool()
        result = tool._run("lines 1-5")

        assert "lines 1-5" in result
        assert "Training pipeline" in result
        # Should NOT contain content from line 10+
        lines = result.strip().split("\n")
        # Header line + 5 content lines
        assert len(lines) <= 7

    def test_full_mode_has_line_numbers(self, experiments_dir):
        from pharma_agents.tools import ReadTrainPyTool

        tool = ReadTrainPyTool()
        result = tool._run("read")

        assert "lines):" in result
        assert "   1 |" in result  # Line number format


class TestRunTrainPyToolNoCache:
    """Tests for RunTrainPyTool cache bypass (anti-loop fix)."""

    def test_cache_function_returns_false(self):
        """RunTrainPyTool should never cache (anti-loop fix)."""
        from pharma_agents.tools import RunTrainPyTool

        tool = RunTrainPyTool()
        # cache_function should always return False
        assert tool.cache_function(None, None) is False
        assert tool.cache_function("run", "output") is False

    def test_code_check_cache_function_returns_false(self):
        """CodeCheckTool should also never cache."""
        from pharma_agents.tools import CodeCheckTool

        tool = CodeCheckTool()
        assert tool.cache_function(None, None) is False
