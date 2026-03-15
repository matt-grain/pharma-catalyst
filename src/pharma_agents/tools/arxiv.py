"""Arxiv paper fetching and search tools."""

import time
from typing import ClassVar

from crewai.tools import BaseTool


class AlphaxivTool(BaseTool):
    """Tool to fetch arxiv papers - tries alphaxiv, falls back to markdown.new."""

    name: str = "fetch_arxiv_paper"
    description: str = (
        "Fetches an arxiv paper and extracts key content as markdown. "
        "Input: arxiv paper ID (e.g., '2401.12345' or '2401.12345v2'). "
        "Tries alphaxiv first, falls back to arxiv abstract page. "
        "Returns abstract, methods, and key techniques."
    )
    cache_function: None = None

    # Rate limiting
    max_papers_per_run: int = 10
    min_interval_seconds: float = 1.0
    timeout_seconds: float = 10.0
    max_retries: int = 2
    _papers_fetched: ClassVar[int] = 0
    _last_fetch: ClassVar[float] = 0.0

    @classmethod
    def reset_counters(cls) -> None:
        cls._papers_fetched = 0
        cls._last_fetch = 0.0

    def _fetch_url(self, url: str) -> str | None:
        """Fetch URL with retry logic."""
        import urllib.error
        import urllib.request

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout_seconds) as resp:
                    return resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None  # Not found, don't retry
                if attempt < self.max_retries - 1:
                    time.sleep(1)
            except Exception:
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        return None

    def _fetch_arxiv_abstract_page(self, paper_id: str) -> str | None:
        """Fetch arxiv abstract page via markdown.new for clean extraction."""
        import urllib.error
        import urllib.request

        # Use markdown.new to get clean markdown from arxiv abstract page
        url = f"https://markdown.new/https://arxiv.org/abs/{paper_id}"

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "pharma-agents/0.1")
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode("utf-8")

            if not content or len(content) < 100:
                return None

            return content

        except (urllib.error.HTTPError, urllib.error.URLError, Exception):
            return None

    def _run(self, paper_id: str) -> str:
        """Fetch paper - alphaxiv first, then arxiv abstract fallback."""
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
            return f"Error: Max papers limit ({self.max_papers_per_run}) reached."

        # Rate limiting
        elapsed = time.time() - AlphaxivTool._last_fetch
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

        AlphaxivTool._last_fetch = time.time()

        # Try alphaxiv overview first (fastest, structured)
        content = self._fetch_url(f"https://alphaxiv.org/overview/{paper_id}.md")
        if content:
            AlphaxivTool._papers_fetched += 1
            return f"=== Paper {paper_id} (alphaxiv overview) ===\n\n{content}"

        # Try alphaxiv full text
        content = self._fetch_url(f"https://alphaxiv.org/abs/{paper_id}.md")
        if content:
            AlphaxivTool._papers_fetched += 1
            if len(content) > 15000:
                content = content[:15000] + "\n\n[... truncated ...]"
            return f"=== Paper {paper_id} (alphaxiv full) ===\n\n{content}"

        # Fallback: fetch arxiv abstract page via markdown.new
        content = self._fetch_arxiv_abstract_page(paper_id)
        if content:
            AlphaxivTool._papers_fetched += 1
            return f"=== Paper {paper_id} (arxiv abstract) ===\n\n{content}"

        return f"Paper {paper_id} not available via alphaxiv or arxiv."


class ArxivSearchTool(BaseTool):
    """Tool to search arxiv for recent papers."""

    name: str = "search_arxiv"
    description: str = (
        "Searches arxiv for recent papers on a topic. "
        "Input: search query (e.g., 'ADMET prediction graph neural network'). "
        "Returns list of arxiv paper IDs with titles and abstracts. "
        "Use this to find papers, then fetch_arxiv_paper for full details."
    )
    cache_function: None = None

    max_results: int = 10
    max_searches_per_run: int = 8
    min_interval_seconds: float = 3.0  # arxiv recommends 3s between requests
    max_retries: int = 2
    _searches_done: ClassVar[int] = 0
    _last_search: ClassVar[float] = 0.0

    @classmethod
    def reset_counters(cls) -> None:
        cls._searches_done = 0
        cls._last_search = 0.0

    def _search_arxiv(self, query: str) -> list[dict] | None:
        """Try arxiv API. Returns list of papers or None on failure."""
        import urllib.error
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        search_query = urllib.parse.quote(f"all:{query}")
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={search_query}&start=0&max_results={self.max_results}"
            f"&sortBy=relevance&sortOrder=descending"
        )

        for attempt in range(self.max_retries):
            try:
                ArxivSearchTool._last_search = time.time()
                with urllib.request.urlopen(url, timeout=15) as response:
                    xml_content = response.read().decode("utf-8")

                root = ET.fromstring(xml_content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                results = []
                for entry in root.findall("atom:entry", ns):
                    paper_id = entry.find("atom:id", ns)
                    title = entry.find("atom:title", ns)
                    summary = entry.find("atom:summary", ns)
                    published = entry.find("atom:published", ns)

                    if paper_id is not None and title is not None:
                        pid = paper_id.text.split("/abs/")[-1] if paper_id.text else ""
                        results.append(
                            {
                                "id": pid,
                                "title": " ".join(title.text.split())
                                if title.text
                                else "",
                                "abstract": summary.text[:300]
                                if summary is not None and summary.text
                                else "",
                                "date": published.text[:10]
                                if published is not None and published.text
                                else "",
                                "source": "arxiv",
                            }
                        )
                return results if results else None

            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep((2**attempt) * 2)
                else:
                    return None
            except Exception:
                return None
        return None

    def _run(self, query: str) -> str:
        """Search arxiv for papers."""
        if ArxivSearchTool._searches_done >= self.max_searches_per_run:
            return (
                f"STOP SEARCHING. You have already done {self.max_searches_per_run} searches. "
                f"Do NOT call search_arxiv again. "
                f"Now use store_paper to save the papers you already found."
            )

        # Rate limiting
        elapsed = time.time() - ArxivSearchTool._last_search
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)

        results = self._search_arxiv(query)
        ArxivSearchTool._searches_done += 1

        if not results:
            return f"No papers found for query: {query}"

        formatted = []
        for r in results:
            abstract = r["abstract"] + "..." if r["abstract"] else ""
            formatted.append(
                f"- **{r['id']}** ({r['date']}): {r['title']}\n  {abstract}\n"
            )

        return f"Found {len(results)} papers for '{query}':\n\n" + "\n".join(formatted)
