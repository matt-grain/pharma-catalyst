"""Arxiv paper fetching and search tools."""

import json
import re
import time
from typing import ClassVar

from crewai.tools import BaseTool


# Common ML/pharma method keywords for extraction from abstracts
_METHOD_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(random\s*forest)",
        r"(gradient\s*boost\w*)",
        r"(XGBoost|LightGBM|CatBoost)",
        r"(graph\s*neural\s*network|GNN|GCN|GAT|MPNN)",
        r"(transformer|attention\s*mechanism|self-attention)",
        r"(convolutional\s*neural\s*network|CNN|1D-CNN)",
        r"(recurrent\s*neural\s*network|RNN|LSTM|GRU)",
        r"(support\s*vector\s*machine|SVM)",
        r"(logistic\s*regression)",
        r"(neural\s*network|deep\s*learning|MLP)",
        r"(molecular\s*fingerprint|Morgan\s*fingerprint|ECFP|MACCS)",
        r"(molecular\s*descriptor|RDKit\s*descriptor|physicochemical)",
        r"(SMILES|SELFIES|InChI)",
        r"(transfer\s*learning|pre-train\w*|fine-tun\w*)",
        r"(ensemble|stacking|bagging|boosting)",
        r"(cross-validation|k-fold)",
        r"(contrastive\s*learning|self-supervised)",
        r"(Bayesian\s*optim\w*|hyperparameter)",
        r"(variational\s*autoencoder|VAE|autoencoder)",
        r"(reinforcement\s*learning)",
        r"(data\s*augmentation|SMOTE|oversampling)",
        r"(feature\s*selection|feature\s*engineering)",
        r"(multi-task\s*learning)",
        r"(few-shot|zero-shot|meta-learning)",
    ]
]


def _extract_methods_from_text(text: str) -> list[str]:
    """Extract ML method names from abstract text using pattern matching."""
    found = []
    seen = set()
    for pattern in _METHOD_PATTERNS:
        match = pattern.search(text)
        if match:
            method = match.group(1).strip()
            key = method.lower()
            if key not in seen:
                seen.add(key)
                found.append(method)
    return found[:5]  # Max 5 methods


class AlphaxivTool(BaseTool):
    """Tool to fetch arxiv papers - tries alphaxiv, falls back to markdown.new."""

    name: str = "fetch_arxiv_paper"
    description: str = (
        "Fetches an arxiv paper and extracts key content as markdown. "
        "Input: arxiv paper ID (e.g., '2401.12345' or '2401.12345v2'). "
        "Tries alphaxiv first, falls back to arxiv abstract page. "
        "Returns abstract, methods, and key techniques."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    # Rate limiting
    max_papers_per_run: int = 30
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

    def _fetch_pdf_as_markdown(self, paper_id: str) -> str | None:
        """Download arxiv PDF and convert to markdown using pymupdf4llm."""
        import tempfile
        import urllib.error
        import urllib.request

        url = f"https://arxiv.org/pdf/{paper_id}.pdf"

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "pharma-agents/0.1")
            with urllib.request.urlopen(req, timeout=30) as response:
                pdf_bytes = response.read()

            if not pdf_bytes or len(pdf_bytes) < 1000:
                return None

            # Write to temp file and convert
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                import logging

                import fitz  # noqa: F401 — needed by pymupdf4llm
                import pymupdf4llm

                fitz.TOOLS.mupdf_display_errors(False)
                logging.getLogger("pymupdf4llm").setLevel(logging.WARNING)
                logging.getLogger("pymupdf").setLevel(logging.WARNING)
                # First 8 pages: title, abstract, intro, methods (skip refs/appendix)
                content = pymupdf4llm.to_markdown(
                    tmp_path, show_progress=False, pages=list(range(8))
                )
            finally:
                import os

                os.unlink(tmp_path)

            if not content or len(content) < 100:
                return None

            return content

        except Exception:
            return None

    @staticmethod
    def _clean_paper_id(paper_id: str) -> str:
        """Normalize paper ID by stripping URLs and suffixes."""
        paper_id = paper_id.strip()
        for prefix in [
            "https://arxiv.org/abs/",
            "https://arxiv.org/pdf/",
            "https://alphaxiv.org/overview/",
            "arxiv.org/abs/",
        ]:
            if paper_id.startswith(prefix):
                paper_id = paper_id[len(prefix):]
        return paper_id.replace(".pdf", "")

    def _run(self, paper_id: str) -> str:
        """Fetch paper - alphaxiv first, then arxiv abstract fallback."""
        paper_id = self._clean_paper_id(paper_id)

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

        # Fallback: download PDF and convert to markdown (pymupdf4llm)
        content = self._fetch_pdf_as_markdown(paper_id)
        if content:
            AlphaxivTool._papers_fetched += 1
            if len(content) > 15000:
                content = content[:15000] + "\n\n[... truncated ...]"
            return f"=== Paper {paper_id} (pdf) ===\n\n{content}"

        return f"Paper {paper_id} not available via alphaxiv or arxiv."


class ArxivSearchTool(BaseTool):
    """Tool to search arxiv for recent papers."""

    name: str = "search_arxiv"
    description: str = (
        "Searches arxiv for recent papers on a topic. "
        "Input: search query (e.g., 'ADMET prediction graph neural network'). "
        "Returns list of arxiv paper IDs with titles and abstracts. "
        "Use this to find papers, then fetch_and_store to store them."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    max_results: int = 3
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

        # Use abs: (abstract) field search with AND between terms
        # to get precise results. all: is too loose (matches astronomy, etc.)
        terms = [t.strip() for t in query.split() if t.strip()]
        abs_query = " AND ".join(f"abs:{t}" for t in terms)
        search_query = urllib.parse.quote(abs_query)
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
                                "abstract": " ".join(summary.text.split())
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
                f"Now use fetch_and_store to store the papers you already found."
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
            abstract = r["abstract"] if r["abstract"] else ""
            formatted.append(
                f"- **{r['id']}** ({r['date']}): {r['title']}\n"
                f"  Abstract: {abstract}\n"
            )

        return f"Found {len(results)} papers for '{query}':\n\n" + "\n".join(formatted)


class SearchAndStoreTool(BaseTool):
    """Combined tool: searches arxiv and stores ALL results automatically."""

    name: str = "search_and_store"
    description: str = (
        "Searches arxiv for papers on a topic, then AUTOMATICALLY stores "
        "all results in the literature database with embeddings. "
        "Input: search query (e.g., 'clinical toxicity prediction ML'). "
        "Returns summary of papers stored. No need to call store_paper separately."
    )
    cache_function: object = lambda _args, _result: False  # type: ignore[assignment]

    max_calls_per_run: int = 5
    _calls_done: ClassVar[int] = 0

    @classmethod
    def reset_counters(cls) -> None:
        cls._calls_done = 0

    def _run(self, query: str) -> str:
        """Search arxiv and store all results."""
        from .literature import LiteratureStoreTool

        if SearchAndStoreTool._calls_done >= self.max_calls_per_run:
            return f"Max search_and_store calls ({self.max_calls_per_run}) reached."

        SearchAndStoreTool._calls_done += 1

        # Search
        search_tool = ArxivSearchTool()
        results = search_tool._search_arxiv(query)

        if not results:
            return f"No papers found for '{query}'."

        # Store each paper
        store_tool = LiteratureStoreTool()
        fetch_tool = AlphaxivTool()
        stored = []
        skipped = []

        for r in results:
            paper_id = r["id"]
            title = r["title"]
            abstract = r["abstract"]

            if not paper_id or not abstract:
                continue

            # Try to fetch richer content (optional enrichment)
            full_content = ""
            if AlphaxivTool._papers_fetched < fetch_tool.max_papers_per_run:
                fetched = fetch_tool._run(paper_id)
                if "not available" not in fetched.lower() and "Error:" not in fetched:
                    full_content = fetched

            # Extract key methods from abstract + title + any fetched content
            text_for_methods = f"{title}. {abstract}"
            if full_content:
                text_for_methods += f" {full_content[:3000]}"
            key_methods = _extract_methods_from_text(text_for_methods)

            store_result = store_tool._run(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "title": title,
                        "summary": abstract,
                        "key_methods": key_methods,
                        "full_content": full_content,
                    }
                )
            )

            if "Stored" in store_result:
                snippet = abstract[:150] + "..." if len(abstract) > 150 else abstract
                methods = ", ".join(key_methods) if key_methods else "none detected"
                stored.append(
                    f"  + **{paper_id}**: {title[:80]}\n"
                    f"    Summary: {snippet}\n"
                    f"    Methods: {methods}"
                )
            elif "skipped" in store_result:
                skipped.append(paper_id)

        output = [f"Search '{query}': {len(stored)} stored, {len(skipped)} already in DB."]
        output.extend(stored)
        if skipped:
            output.append(f"  (skipped: {', '.join(skipped)})")
        output.append(
            "\nReview the papers above. Use remove_paper to delete any that are "
            "NOT relevant to the target property prediction task."
        )

        return "\n".join(output)
