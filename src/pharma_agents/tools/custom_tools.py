"""Custom CrewAI tools for pharma-agents."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from crewai.tools import BaseTool

from pharma_agents.memory import get_experiments_dir, get_metric_name


# Skills directory relative to project root
SKILLS_DIR = Path(__file__).parent.parent.parent.parent / ".claude" / "skills"


# =============================================================================
# Archivist Tools - Literature Research
# =============================================================================


def get_literature_dir() -> Path:
    """Get the literature directory for the current experiment."""
    return get_experiments_dir() / "literature"


class AlphaxivTool(BaseTool):
    """Tool to fetch arxiv papers as markdown from alphaxiv.org."""

    name: str = "fetch_arxiv_paper"
    description: str = (
        "Fetches an arxiv paper summary from alphaxiv.org as markdown. "
        "Input: arxiv paper ID (e.g., '2401.12345' or '2401.12345v2'). "
        "Returns structured AI-generated overview optimized for LLM consumption. "
        "Much faster than reading PDFs."
    )
    cache_function: None = None

    # Rate limiting
    max_papers_per_run: int = 10
    min_interval_seconds: float = 1.0
    _papers_fetched: ClassVar[int] = 0
    _last_fetch: ClassVar[float] = 0.0

    def _run(self, paper_id: str) -> str:
        """Fetch paper from alphaxiv."""
        import urllib.request
        import urllib.error

        paper_id = paper_id.strip()

        # Remove URL prefixes if present
        for prefix in [
            "https://arxiv.org/abs/",
            "https://arxiv.org/pdf/",
            "https://alphaxiv.org/overview/",
            "arxiv.org/abs/",
        ]:
            if paper_id.startswith(prefix):
                paper_id = paper_id[len(prefix) :]

        # Remove .pdf suffix
        paper_id = paper_id.replace(".pdf", "")

        # Check limits
        if AlphaxivTool._papers_fetched >= self.max_papers_per_run:
            return (
                f"Error: Max papers limit ({self.max_papers_per_run}) reached this run."
            )

        # Rate limiting
        elapsed = time.time() - AlphaxivTool._last_fetch
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

        # Try overview first (structured summary)
        url = f"https://alphaxiv.org/overview/{paper_id}.md"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read().decode("utf-8")
                AlphaxivTool._papers_fetched += 1
                AlphaxivTool._last_fetch = time.time()
                return f"=== Paper {paper_id} (Overview) ===\n\n{content}"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Try full text as fallback
                url = f"https://alphaxiv.org/abs/{paper_id}.md"
                try:
                    with urllib.request.urlopen(url, timeout=30) as response:
                        content = response.read().decode("utf-8")
                        AlphaxivTool._papers_fetched += 1
                        AlphaxivTool._last_fetch = time.time()
                        # Truncate if very long
                        if len(content) > 15000:
                            content = content[:15000] + "\n\n[... truncated ...]"
                        return f"=== Paper {paper_id} (Full Text) ===\n\n{content}"
                except urllib.error.HTTPError:
                    return f"Paper {paper_id} not found on alphaxiv. Try: https://arxiv.org/abs/{paper_id}"
            return f"Error fetching paper: {e}"
        except Exception as e:
            return f"Error: {e}"


class ArxivSearchTool(BaseTool):
    """Tool to search arxiv for recent papers."""

    name: str = "search_arxiv"
    description: str = (
        "Searches arxiv for recent papers on a topic. "
        "Input: search query (e.g., 'ADMET prediction graph neural network'). "
        "Returns list of paper IDs with titles and abstracts. "
        "Use this to find relevant papers, then fetch_arxiv_paper for details."
    )
    cache_function: None = None

    max_results: int = 10
    max_searches_per_run: int = 5
    _searches_done: ClassVar[int] = 0

    def _run(self, query: str) -> str:
        """Search arxiv API."""
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET

        if ArxivSearchTool._searches_done >= self.max_searches_per_run:
            return (
                f"Error: Max searches ({self.max_searches_per_run}) reached this run."
            )

        # Build arxiv API query
        # Focus on cs.LG (machine learning) and q-bio (quantitative biology)
        search_query = urllib.parse.quote(f"all:{query}")
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={search_query}&start=0&max_results={self.max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )

        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                xml_content = response.read().decode("utf-8")

            ArxivSearchTool._searches_done += 1

            # Parse XML
            root = ET.fromstring(xml_content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            results = []
            for entry in root.findall("atom:entry", ns):
                paper_id = entry.find("atom:id", ns)
                title = entry.find("atom:title", ns)
                summary = entry.find("atom:summary", ns)
                published = entry.find("atom:published", ns)

                if paper_id is not None and title is not None:
                    # Extract just the ID from the URL
                    pid = paper_id.text.split("/abs/")[-1] if paper_id.text else ""
                    ttl = " ".join(title.text.split()) if title.text else ""
                    abstract = (
                        summary.text[:300] + "..."
                        if summary is not None and summary.text
                        else ""
                    )
                    pub_date = (
                        published.text[:10]
                        if published is not None and published.text
                        else ""
                    )

                    results.append(f"- **{pid}** ({pub_date}): {ttl}\n  {abstract}\n")

            if not results:
                return f"No papers found for query: {query}"

            return f"Found {len(results)} papers for '{query}':\n\n" + "\n".join(
                results
            )

        except Exception as e:
            return f"Error searching arxiv: {e}"


class LiteratureStoreTool(BaseTool):
    """Tool to store paper summaries with embeddings."""

    name: str = "store_paper"
    description: str = (
        "Stores a paper summary in the literature database with embeddings. "
        "Input: JSON with 'paper_id', 'title', 'summary', and optional 'key_methods'. "
        "This enables semantic search for the Hypothesis Agent later."
    )

    def _run(self, paper_json: str) -> str:
        """Store paper in literature DB."""
        try:
            from fastembed import TextEmbedding
        except ImportError:
            return "Error: fastembed not installed. Run: uv add fastembed"

        lit_dir = get_literature_dir()
        lit_dir.mkdir(parents=True, exist_ok=True)

        # Parse input
        try:
            paper = json.loads(paper_json)
        except json.JSONDecodeError:
            return "Error: Invalid JSON. Expected: {paper_id, title, summary, key_methods?}"

        paper_id = paper.get("paper_id", "").strip()
        title = paper.get("title", "").strip()
        summary = paper.get("summary", "").strip()
        key_methods = paper.get("key_methods", [])

        if not paper_id or not summary:
            return "Error: paper_id and summary are required."

        # Create embedding
        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        text_to_embed = f"{title}. {summary}"
        embedding = list(model.embed([text_to_embed]))[0].tolist()

        # Load or create index
        index_path = lit_dir / "index.json"
        if index_path.exists():
            index: dict = json.loads(index_path.read_text())
        else:
            index = {"papers": {}, "created_at": datetime.now().isoformat()}

        # Store paper
        papers_dict: dict = index.get("papers", {})
        papers_dict[paper_id] = {
            "title": title,
            "summary": summary,
            "key_methods": key_methods,
            "embedding": embedding,
            "added_at": datetime.now().isoformat(),
        }
        index["papers"] = papers_dict
        index["updated_at"] = datetime.now().isoformat()

        index_path.write_text(json.dumps(index, indent=2))

        # Also save markdown summary
        papers_dir = lit_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        safe_id = paper_id.replace("/", "_").replace(":", "_")
        (papers_dir / f"{safe_id}.md").write_text(
            f"# {title}\n\n**Paper ID:** {paper_id}\n\n## Summary\n\n{summary}\n\n"
            f"## Key Methods\n\n" + "\n".join(f"- {m}" for m in key_methods)
        )

        return f"Stored paper {paper_id} with embedding ({len(embedding)} dims)."


class LiteratureQueryTool(BaseTool):
    """Tool to query literature database semantically."""

    name: str = "query_literature"
    description: str = (
        "Searches the literature database for relevant papers. "
        "Input: search query (e.g., 'gradient boosting molecular features'). "
        "Returns top matching papers with summaries. "
        "Use this to find techniques from recent research."
    )

    top_k: int = 5

    def _run(self, query: str) -> str:
        """Query literature DB."""
        try:
            from fastembed import TextEmbedding
            import numpy as np
        except ImportError:
            return "Error: fastembed not installed."

        lit_dir = get_literature_dir()
        index_path = lit_dir / "index.json"

        if not index_path.exists():
            return "No literature database found. Run the Archivist first to gather papers."

        index = json.loads(index_path.read_text())
        papers = index.get("papers", {})

        if not papers:
            return "Literature database is empty."

        # Create query embedding
        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        query_emb = list(model.embed([query]))[0]

        # Calculate similarities
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        results = []
        for paper_id, paper in papers.items():
            emb = paper.get("embedding")
            if emb:
                sim = cosine_sim(query_emb, emb)
                results.append((sim, paper_id, paper))

        # Sort by similarity
        results.sort(reverse=True, key=lambda x: x[0])

        # Format top results
        output = [f"Top {min(self.top_k, len(results))} papers for '{query}':\n"]
        for sim, paper_id, paper in results[: self.top_k]:
            output.append(
                f"**[{sim:.2f}] {paper_id}**: {paper['title']}\n"
                f"  {paper['summary'][:200]}...\n"
            )

        return "\n".join(output)


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


class SkillLoaderTool(BaseTool):
    """Tool to load scientific skills for context."""

    name: str = "load_skill"
    description: str = (
        "Loads a scientific skill to get domain knowledge and code examples. "
        "Available skills: rdkit, deepchem, datamol, molfeat, pytdc, "
        "chembl-database, pubchem-database, literature-review. "
        "Input: skill name (e.g., 'rdkit' or 'molfeat'). "
        "Returns the skill content with best practices and code examples."
    )

    # Constraints
    max_skills_per_run: int = 3
    _skills_loaded: list = []

    def _run(self, skill_name: str) -> str:
        """Load a skill by name."""
        skill_name = skill_name.strip().lower()

        # Check limit
        if len(self._skills_loaded) >= self.max_skills_per_run:
            return f"Error: Max skills limit ({self.max_skills_per_run}) reached. Already loaded: {self._skills_loaded}"

        # Check if already loaded
        if skill_name in self._skills_loaded:
            return f"Skill '{skill_name}' already loaded this session."

        # Try scientific skills first, then root skills
        skill_paths = [
            SKILLS_DIR / "scientific" / f"{skill_name}.md",
            SKILLS_DIR / f"{skill_name}.md",
        ]

        for skill_path in skill_paths:
            if skill_path.exists():
                try:
                    content = skill_path.read_text(encoding="utf-8")
                    self._skills_loaded.append(skill_name)
                    # Truncate if too long (keep first 8000 chars)
                    if len(content) > 8000:
                        content = (
                            content[:8000] + "\n\n[... truncated, skill too long ...]"
                        )
                    return f"=== SKILL: {skill_name} ===\n\n{content}"
                except Exception as e:
                    return f"Error loading skill: {e}"

        # List available skills
        available = []
        if (SKILLS_DIR / "scientific").exists():
            available = [f.stem for f in (SKILLS_DIR / "scientific").glob("*.md")]
        return f"Skill '{skill_name}' not found. Available: {available}"
