"""Literature storage and query tools."""

import json
import os
import re
import sys
import time as _time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from crewai.tools import BaseTool

from pharma_agents.memory import get_experiment_name, get_experiments_root

# Content limits
MAX_SUMMARY_LENGTH = 1000
MAX_KEY_METHODS = 5
MIN_FULL_CONTENT_LENGTH = 200


@contextmanager
def _locked_index(index_path: Path):
    """Read-modify-write index.json with file locking to prevent corruption.

    Uses fcntl on Unix and msvcrt on Windows.
    """
    lock_path = index_path.with_suffix(".lock")
    lock_path.touch(exist_ok=True)
    lock_file = open(lock_path)  # noqa: SIM115
    try:
        if sys.platform == "win32":
            import msvcrt

            # Retry lock acquisition on Windows (non-blocking by default)
            for _ in range(50):
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    _time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_EX)

        if index_path.exists():
            try:
                index = json.loads(index_path.read_text())
            except (json.JSONDecodeError, ValueError):
                index = {"papers": {}, "created_at": datetime.now().isoformat()}
        else:
            index = {"papers": {}, "created_at": datetime.now().isoformat()}
        yield index
        index["updated_at"] = datetime.now().isoformat()
        index_path.write_text(json.dumps(index, indent=2))
    finally:
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def get_literature_dir() -> Path:
    """Get the literature directory for the current experiment.

    Uses PHARMA_LITERATURE_DIR if set (for tests), otherwise uses
    the MAIN experiments dir (not worktree) so literature persists
    across runs.
    """
    # Test override only - NOT PHARMA_EXPERIMENTS_DIR (that's set to worktree)
    override = os.environ.get("PHARMA_LITERATURE_DIR")
    if override:
        return Path(override)

    # Always use main experiments dir (not worktree)
    return get_experiments_root() / get_experiment_name() / "literature"


class LiteratureStoreTool(BaseTool):
    """Tool to store paper summaries with embeddings."""

    name: str = "store_paper"
    description: str = (
        "Stores a paper in the literature database with embeddings. "
        "Input: JSON with 'paper_id', 'title', 'summary', and optional 'key_methods'. "
        "OR: Raw markdown content (will auto-extract paper_id and use content as summary). "
        "This enables semantic search for the Hypothesis Agent later."
    )

    def _extract_from_markdown(self, content: str) -> dict | None:
        """Try to extract paper info from markdown content."""
        paper_id = None
        title = None

        # Look for arxiv ID patterns (multiple formats)
        arxiv_match = re.search(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)", content)
        if arxiv_match:
            paper_id = arxiv_match.group(1)

        # Look for paper ID in "=== Paper XXX ===" format
        if not paper_id:
            header_match = re.search(r"=== Paper (\d{4}\.\d{4,5}(?:v\d+)?)", content)
            if header_match:
                paper_id = header_match.group(1)

        # Fallback: look for any arxiv-style ID anywhere (XXXX.XXXXX format)
        if not paper_id:
            fallback_match = re.search(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", content)
            if fallback_match:
                paper_id = fallback_match.group(1)

        # Look for title
        title_match = re.search(r"[Tt]itle[:\s]*([^\n]+)", content)
        if title_match:
            title = title_match.group(1).strip().strip("*#")

        # Extract abstract/summary (try multiple patterns, most specific first)
        summary = None
        for pattern in [
            r"> Abstract:\s*(.+?)(?:\n\n|\n\||$)",  # Quoted abstract (markdown.new)
            r"Summary[:\s]+(.+?)(?:\n\n|Key Techniques|Key Methods|$)",  # alphaxiv overview
            r"\nAbstract\s*\n(.+?)(?:\n\n|$)",  # Abstract as section header (alphaxiv full)
            r"Abstract\s*[:]\s*(.+?)(?:\n\n|\n1[\s.]|\n#|$)",  # "Abstract:" inline (alphaxiv full)
        ]:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                summary = match.group(1).strip()
                # Clean up whitespace from PDF text extraction
                summary = re.sub(r"\s+", " ", summary)
                break
        if not summary:
            # Fallback: skip headers and get content
            lines = [
                line
                for line in content.split("\n")
                if line.strip() and not line.startswith("===")
            ]
            summary = " ".join(lines[:5])[:MAX_SUMMARY_LENGTH]

        # Extract key techniques/methods
        key_methods = []
        techniques_match = re.search(
            r"Key Techniques[:\s]*(.+?)(?:\n\n|$)", content, re.DOTALL
        )
        if techniques_match:
            methods_text = techniques_match.group(1).strip()
            # Split on bullet points (lines starting with -)
            bullets = [
                line.lstrip("- ").strip()
                for line in methods_text.split("\n")
                if line.strip().startswith("-")
            ]
            if bullets:
                key_methods = [b for b in bullets if b]
            elif ", " in methods_text:
                # Comma-separated (alphaxiv overview single-line format)
                key_methods = [
                    m.strip() for m in methods_text.split(", ") if m.strip()
                ]
            else:
                # Fallback: split on periods for non-bulleted text
                key_methods = [
                    m.strip() for m in methods_text.split(".") if m.strip()
                ]

        if paper_id:
            return {
                "paper_id": paper_id,
                "title": title or f"Paper {paper_id}",
                "summary": summary,
                "key_methods": key_methods[:MAX_KEY_METHODS],
            }
        return None

    def _run(self, paper_input: str) -> str:
        """Store paper in literature DB."""
        try:
            from fastembed import TextEmbedding
        except ImportError:
            return "Error: fastembed not installed. Run: uv add fastembed"

        lit_dir = get_literature_dir()
        lit_dir.mkdir(parents=True, exist_ok=True)

        # Try JSON first
        paper = None
        full_content = None
        try:
            paper = json.loads(paper_input)
            # JSON may include full_content field
            full_content = paper.get("full_content")
        except json.JSONDecodeError:
            # Not JSON - try to extract from markdown
            # Save the raw input as full content
            full_content = paper_input
            paper = self._extract_from_markdown(paper_input)
            if not paper:
                return (
                    "Error: Could not parse input. Expected JSON {paper_id, title, summary} "
                    "or markdown with arxiv ID and abstract."
                )

        paper_id = paper.get("paper_id", "").strip()
        title = paper.get("title", "").strip()
        summary = paper.get("summary", "").strip()
        key_methods = paper.get("key_methods", [])

        if not paper_id or not summary:
            return "Error: paper_id and summary are required."

        # Quick pre-check (without lock) to skip embedding if already stored
        index_path = lit_dir / "index.json"
        if index_path.exists():
            try:
                existing = json.loads(index_path.read_text())
            except (json.JSONDecodeError, ValueError):
                existing = {"papers": {}}
            if paper_id in existing.get("papers", {}):
                return f"Paper {paper_id} already in database — skipped."
            if paper_id in existing.get("removed", []):
                return f"Paper {paper_id} was previously removed as irrelevant — skipped."

        # Create embedding (outside lock — this is the slow part)
        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        text_to_embed = f"{title}. {summary}"
        embedding = list(model.embed([text_to_embed]))[0].tolist()

        # Locked read-modify-write to prevent concurrent corruption
        with _locked_index(index_path) as index:
            # Re-check under lock (state may have changed during embedding)
            if paper_id in index.get("papers", {}):
                return f"Paper {paper_id} already in database — skipped."
            if paper_id in index.get("removed", []):
                return f"Paper {paper_id} was previously removed as irrelevant — skipped."

            papers_dict: dict = index.get("papers", {})
            papers_dict[paper_id] = {
                "title": title,
                "summary": summary,
                "key_methods": key_methods,
                "embedding": embedding,
                "added_at": datetime.now().isoformat(),
            }
            index["papers"] = papers_dict

        # Also save markdown summary
        papers_dir = lit_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        safe_id = paper_id.replace("/", "_").replace(":", "_")

        # Save full content only if it's real full content (not arxiv abstract fallback)
        has_full = False
        if full_content and len(full_content) > MIN_FULL_CONTENT_LENGTH:
            # Skip noisy arxiv abstract fallback - only save alphaxiv full content
            is_arxiv_abstract = "(arxiv abstract)" in full_content[:100]
            if not is_arxiv_abstract:
                (papers_dir / f"{safe_id}_full.md").write_text(full_content)
                has_full = True

        # Build summary markdown with links
        full_link = (
            f"**Full Content:** [{safe_id}_full.md]({safe_id}_full.md)\n"
            if has_full
            else ""
        )
        (papers_dir / f"{safe_id}.md").write_text(
            f"# {title}\n\n**Paper ID:** {paper_id}\n"
            f"**PDF:** https://arxiv.org/pdf/{paper_id}.pdf\n"
            f"{full_link}\n"
            f"## Summary\n\n{summary}\n\n"
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

    # Cache embedding model to avoid reloading on every query
    _model: ClassVar = None

    @classmethod
    def _get_model(cls):
        """Get or create cached embedding model."""
        if cls._model is None:
            from fastembed import TextEmbedding

            cls._model = TextEmbedding("BAAI/bge-small-en-v1.5")
        return cls._model

    def _run(self, query: str) -> str:
        """Query literature DB."""
        try:
            import numpy as np
        except ImportError:
            return "Error: numpy not installed."

        lit_dir = get_literature_dir()
        index_path = lit_dir / "index.json"

        if not index_path.exists():
            return "No literature database found. Run the Archivist first to gather papers."

        try:
            index = json.loads(index_path.read_text())
        except (json.JSONDecodeError, ValueError):
            return "Literature database is corrupted. Re-run the Archivist to rebuild."
        papers = index.get("papers", {})

        if not papers:
            return "Literature database is empty."

        # Create query embedding (uses cached model)
        try:
            model = self._get_model()
        except ImportError:
            return "Error: fastembed not installed."
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


class RemovePaperTool(BaseTool):
    """Tool to remove irrelevant papers from the literature database."""

    name: str = "remove_paper"
    description: str = (
        "Removes an irrelevant paper from the literature database. "
        "Input: paper ID (e.g., '2401.12345v1'). "
        "Use this after search_and_store to filter out papers that are "
        "NOT relevant to the target prediction task."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    def _run(self, paper_id: str) -> str:
        """Remove paper from literature DB."""
        paper_id = paper_id.strip()
        lit_dir = get_literature_dir()
        index_path = lit_dir / "index.json"

        if not index_path.exists():
            return "No literature database found."

        with _locked_index(index_path) as index:
            papers = index.get("papers", {})
            removed_list: list = index.get("removed", [])

            if paper_id not in papers:
                # Still add to blacklist so it won't be re-stored
                if paper_id not in removed_list:
                    removed_list.append(paper_id)
                    index["removed"] = removed_list
                return f"Paper {paper_id} not found in database (blacklisted from future storage)."

            # Remove from index and add to blacklist
            title = papers[paper_id].get("title", paper_id)
            del papers[paper_id]
            index["papers"] = papers
            if paper_id not in removed_list:
                removed_list.append(paper_id)
            index["removed"] = removed_list

        # Remove markdown files (outside lock — no index contention)
        papers_dir = lit_dir / "papers"
        safe_id = paper_id.replace("/", "_").replace(":", "_")
        for suffix in ["", "_full"]:
            md_file = papers_dir / f"{safe_id}{suffix}.md"
            if md_file.exists():
                md_file.unlink()

        return f"Removed '{title}' ({paper_id}) from literature database."


class FetchMorePapersTool(BaseTool):
    """Tool for Hypothesis Agent to fetch fresh papers on a specific topic."""

    name: str = "fetch_more_papers"
    description: str = (
        "Fetch fresh papers on a specific topic when you need more ideas. "
        "Use this when query_literature doesn't have enough relevant papers. "
        "Input: specific topic (e.g., 'attention mechanisms for molecules'). "
        "This searches arxiv, fetches via alphaxiv, and stores in literature DB."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    max_calls_per_run: int = 2
    _calls_done: ClassVar[int] = 0

    @classmethod
    def reset_counters(cls) -> None:
        cls._calls_done = 0

    def _run(self, topic: str) -> str:
        """Fetch more papers on a topic."""
        from .arxiv import AlphaxivTool, ArxivSearchTool

        if FetchMorePapersTool._calls_done >= self.max_calls_per_run:
            return f"Error: Max calls ({self.max_calls_per_run}) reached. Use existing literature."

        FetchMorePapersTool._calls_done += 1

        # Use the other tools internally
        search_tool = ArxivSearchTool()
        fetch_tool = AlphaxivTool()
        store_tool = LiteratureStoreTool()

        results = []

        # Search for papers
        search_result = search_tool._run(topic)
        if "No papers found" in search_result:
            return f"No papers found for '{topic}'. Try a different topic."

        results.append(f"Searched arxiv for '{topic}'")

        # Extract paper IDs and abstracts from search results
        # Format: - **XXXX.XXXXX** (date): Title\n  Abstract: ...
        paper_entries = re.findall(
            r"\*\*(\d{4}\.\d{4,5}(?:v\d+)?)\*\*[^:]*:\s*(.+?)\n\s*Abstract:\s*(.+?)(?=\n- \*\*|\n\nFound|\Z)",
            search_result,
            re.DOTALL,
        )

        if not paper_entries:
            # Fallback: just extract IDs
            paper_ids = re.findall(
                r"\*\*(\d{4}\.\d{4,5}(?:v\d+)?)\*\*", search_result
            )
            paper_entries = [(pid, pid, "") for pid in paper_ids]

        if not paper_entries:
            return "Could not extract paper IDs from search. Try different keywords."

        # Fetch and store top 3 papers
        papers_stored = 0
        for paper_id, title, abstract in paper_entries[:3]:
            title = title.strip()
            abstract = abstract.strip()

            # Try to fetch full content (may fail)
            paper_content = fetch_tool._run(paper_id)
            full_content = ""
            if "not available" not in paper_content.lower():
                full_content = paper_content

            # Use abstract from search if we have it
            summary = abstract if abstract else ""
            if not summary and full_content:
                # Fallback: extract from fetched content
                extracted = LiteratureStoreTool()._extract_from_markdown(
                    full_content
                )
                if extracted:
                    summary = extracted.get("summary", "")
                    title = extracted.get("title", title)

            if not summary:
                summary = f"Paper on {title}"

            store_result = store_tool._run(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "title": title[:200],
                        "summary": summary[:MAX_SUMMARY_LENGTH],
                        "key_methods": [],
                        "full_content": full_content,
                    }
                )
            )

            if "Stored" in store_result:
                papers_stored += 1
                results.append(f"  - Stored {paper_id}: {title[:50]}...")

        if papers_stored > 0:
            results.append(
                f"\nAdded {papers_stored} papers to literature DB. Use query_literature to search them."
            )
        else:
            results.append(
                "\nCould not store any papers. They may not be indexed on alphaxiv yet."
            )

        return "\n".join(results)
