"""Literature storage and query tools."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from crewai.tools import BaseTool

from pharma_agents.memory import get_experiment_name, get_experiments_root


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

        # Extract abstract/summary (try multiple patterns)
        summary = None
        for pattern in [
            r"> Abstract:(.+?)(?:\n\n|\n\|)",  # Quoted abstract (arxiv page)
            r"Summary[:\s]*(.+?)(?:\n\n|Key Techniques|Key Methods|$)",  # alphaxiv format
            r"Abstract[:\s>]*(.+?)(?:\n\n|\n\||$)",  # General abstract
        ]:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                summary = match.group(1).strip()
                break
        if not summary:
            # Fallback: skip headers and get content
            lines = [l for l in content.split("\n") if l.strip() and not l.startswith("===")]
            summary = " ".join(lines[:5])[:1000]

        # Extract key techniques/methods
        key_methods = []
        techniques_match = re.search(
            r"Key Techniques[:\s]*(.+?)(?:\n\n|$)", content, re.DOTALL
        )
        if techniques_match:
            # Split on commas or periods
            methods_text = techniques_match.group(1).strip()
            key_methods = [m.strip() for m in re.split(r"[,.]", methods_text) if m.strip()]

        if paper_id:
            return {
                "paper_id": paper_id,
                "title": title or f"Paper {paper_id}",
                "summary": summary,
                "key_methods": key_methods[:5],  # Limit to 5
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

        # Save full content only if it's real full content (not arxiv abstract fallback)
        has_full = False
        if full_content and len(full_content) > 200:
            # Skip noisy arxiv abstract fallback - only save alphaxiv full content
            is_arxiv_abstract = "(arxiv abstract)" in full_content[:100]
            if not is_arxiv_abstract:
                (papers_dir / f"{safe_id}_full.md").write_text(full_content)
                has_full = True

        # Build summary markdown with links
        full_link = f"**Full Content:** [{safe_id}_full.md]({safe_id}_full.md)\n" if has_full else ""
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


class FetchMorePapersTool(BaseTool):
    """Tool for Hypothesis Agent to fetch fresh papers on a specific topic."""

    name: str = "fetch_more_papers"
    description: str = (
        "Fetch fresh papers on a specific topic when you need more ideas. "
        "Use this when query_literature doesn't have enough relevant papers. "
        "Input: specific topic (e.g., 'attention mechanisms for molecules'). "
        "This searches arxiv, fetches via alphaxiv, and stores in literature DB."
    )
    cache_function: None = None

    max_calls_per_run: int = 2
    _calls_done: ClassVar[int] = 0

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

        # Extract paper IDs from search results (format: **XXXX.XXXXX**)
        paper_ids = re.findall(r"\*\*(\d{4}\.\d{4,5}(?:v\d+)?)\*\*", search_result)

        if not paper_ids:
            return "Could not extract paper IDs from search. Try different keywords."

        # Fetch and store top 3 papers
        papers_stored = 0
        for paper_id in paper_ids[:3]:
            # Fetch paper
            paper_content = fetch_tool._run(paper_id)
            if "not found" in paper_content.lower():
                continue

            # Extract title and summary from markdown content
            lines = paper_content.split("\n")
            title = paper_id  # Default
            summary = ""

            for line in lines:
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("## Summary") or line.startswith("## Abstract"):
                    # Get next few lines as summary
                    idx = lines.index(line)
                    summary = " ".join(lines[idx + 1 : idx + 5]).strip()
                    break

            if not summary:
                summary = " ".join(lines[5:10])[:500]

            # Store paper with full content
            store_result = store_tool._run(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "title": title[:200],
                        "summary": summary[:1000],
                        "key_methods": [],
                        "full_content": paper_content,
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
