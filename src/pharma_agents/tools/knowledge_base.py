"""Knowledge base RAG with hybrid retrieval (BM25 + dense + RRF)."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from crewai.tools import BaseTool

from pharma_agents.memory import get_experiment_name, get_experiments_root

# Chunking constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval constants
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75

# Module-level model cache for fastembed (loaded once per process)
_cached_model: object = None


def get_kb_dir() -> Path:
    """Get the knowledge base directory for the current experiment.

    Uses PHARMA_KB_DIR if set (for tests), otherwise uses the main
    experiments dir so the knowledge base persists across runs.
    """
    override = os.environ.get("PHARMA_KB_DIR")
    if override:
        return Path(override)
    return get_experiments_root() / get_experiment_name() / "knowledge_base"


def _write_index(index_path: Path, index: dict) -> None:
    """Write KB index as a full JSON replacement (not incremental like literature)."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _tokenize(text: str) -> list[str]:
    """BM25 tokenizer: lowercase, strip punctuation, filter short tokens."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return [t for t in cleaned.split() if len(t) > 1]


def _split_into_word_chunks(text: str, size: int, overlap: int) -> list[str]:
    """Split text into word-boundary chunks with overlap."""
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += size - overlap
    return chunks


def _apply_overlap(chunks: list[str]) -> list[str]:
    """Prepend last CHUNK_OVERLAP words of previous chunk to each subsequent chunk."""
    if len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        tail = (
            " ".join(prev_words[-CHUNK_OVERLAP:])
            if len(prev_words) >= CHUNK_OVERLAP
            else chunks[i - 1]
        )
        result.append(tail + " " + chunks[i])
    return result


def _split_section(text: str) -> list[str]:
    """Split a section that exceeds CHUNK_SIZE: first by paragraphs, then by words."""
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_chunks: list[str] = []
    for para in paragraphs:
        para_words = para.split()
        if len(para_words) > CHUNK_SIZE:
            raw_chunks.extend(_split_into_word_chunks(para, CHUNK_SIZE, 0))
        else:
            raw_chunks.append(para)
    return raw_chunks


def _extract_doc_title(text: str) -> str:
    """Extract document title from YAML frontmatter or first heading.

    Contextual retrieval: prepending the doc title to each chunk before
    embedding captures both local detail and global context (Anthropic cookbook).
    """
    # Try YAML frontmatter title first
    fm_match = re.search(
        r"^---\s*\n.*?^title:\s*[\"']?(.+?)[\"']?\s*$.*?^---",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if fm_match:
        return fm_match.group(1).strip()
    # Fall back to first # heading
    heading_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()
    return ""


def chunk_markdown(file_path: Path, kb_dir: Path) -> list[dict]:
    """Split a markdown file into overlapping chunks by heading sections."""
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc_title = _extract_doc_title(text)

    # Split into sections on ## headings; # header belongs to first section
    sections: list[tuple[str, str, int]] = []  # (heading, body, line_start)
    current_heading = "header"
    current_lines: list[str] = []
    current_start = 0
    for i, line in enumerate(lines):
        if line.startswith("## "):
            sections.append((current_heading, "\n".join(current_lines), current_start))
            current_heading = re.sub(r"^#+\s*", "", line).strip()
            current_lines = []
            current_start = i
        else:
            current_lines.append(line)
    sections.append((current_heading, "\n".join(current_lines), current_start))

    doc_id = file_path.stem
    source_file = str(file_path.relative_to(kb_dir))
    # Contextual retrieval prefix: document title prepended for embedding
    ctx_prefix = f"[From: {doc_title}] " if doc_title else ""
    result: list[dict] = []

    for heading, body, line_start in sections:
        if not body.strip():
            continue
        raw_chunks = _split_section(body)
        overlapped = _apply_overlap(raw_chunks)
        for chunk_text in overlapped:
            chunk_lines = chunk_text.count("\n") + 1
            raw_content = chunk_text.strip()
            result.append(
                {
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "section": heading,
                    "content": raw_content,
                    "embed_text": f"{ctx_prefix}{heading}: {raw_content}",
                    "line_start": line_start,
                    "line_end": line_start + chunk_lines,
                }
            )

    return result


def chunk_csv(file_path: Path, kb_dir: Path) -> list[dict]:
    """Split a CSV file into natural-language chunks of 10 rows each."""
    doc_id = file_path.stem
    source_file = str(file_path.relative_to(kb_dir))
    # Contextual retrieval: use filename as context for CSV data
    ctx_prefix = f"[From: {doc_id.replace('_', ' ')} data] "
    result: list[dict] = []

    with file_path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    batch_size = 10
    for start in range(0, len(reader), batch_size):
        batch = reader[start : start + batch_size]
        row_lines: list[str] = []
        for offset, row in enumerate(batch):
            fields = ", ".join(f"{col}: {val}" for col, val in row.items())
            row_lines.append(f"Row {start + offset + 1}: {fields}")
        content = "\n".join(row_lines)
        end = start + len(batch)
        section = f"rows_{start + 1}-{end}"
        result.append(
            {
                "doc_id": doc_id,
                "source_file": source_file,
                "section": section,
                "content": content,
                "embed_text": f"{ctx_prefix}{section}: {content}",
                "line_start": start + 1,
                "line_end": end,
            }
        )

    return result


def _build_index(kb_dir: Path) -> dict:
    """Scan kb_dir, chunk all .md and .csv files, embed and build BM25 stats."""
    all_chunks: list[dict] = []
    for md_file in sorted(kb_dir.glob("**/*.md")):
        if md_file.name in ("index.json", "index.lock"):
            continue
        all_chunks.extend(chunk_markdown(md_file, kb_dir))
    for csv_file in sorted(kb_dir.glob("**/*.csv")):
        if csv_file.name in ("index.json", "index.lock"):
            continue
        all_chunks.extend(chunk_csv(csv_file, kb_dir))

    if not all_chunks:
        return {
            "chunks": {},
            "bm25_doc_count": 0,
            "bm25_avg_dl": 0.0,
            "bm25_df": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    model = _get_cached_model()
    # Contextual retrieval: embed the enriched text (with doc title prefix),
    # but store raw content for display. This captures both local detail
    # and global document context in the embedding vector.
    embed_texts = [c.get("embed_text", c["content"]) for c in all_chunks]
    embeddings = list(model.embed(embed_texts))

    chunks_index: dict[str, dict] = {}
    bm25_df: Counter = Counter()
    doc_lengths: list[int] = []

    for i, chunk in enumerate(all_chunks):
        chunk_id = f"{chunk['doc_id']}::chunk_{i}"
        token_freqs = dict(Counter(_tokenize(chunk["content"])))
        dl = sum(token_freqs.values())
        doc_lengths.append(dl)
        bm25_df.update(set(token_freqs.keys()))
        chunks_index[chunk_id] = {
            **{
                k: chunk[k]
                for k in (
                    "doc_id",
                    "source_file",
                    "section",
                    "content",
                    "line_start",
                    "line_end",
                )
            },
            "embedding": embeddings[i].tolist(),
            "token_freqs": token_freqs,
        }

    bm25_avg_dl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

    return {
        "chunks": chunks_index,
        "bm25_doc_count": len(chunks_index),
        "bm25_avg_dl": bm25_avg_dl,
        "bm25_df": dict(bm25_df),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


def _bm25_score(query_tokens: list[str], chunk: dict, index: dict) -> float:
    """Okapi BM25 score for a single chunk given query tokens."""
    n = index["bm25_doc_count"]
    avg_dl = index["bm25_avg_dl"]
    token_freqs: dict[str, int] = chunk["token_freqs"]
    dl = sum(token_freqs.values())
    score = 0.0
    for token in query_tokens:
        tf = token_freqs.get(token, 0)
        df = index["bm25_df"].get(token, 0)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        denom = (
            tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl) if avg_dl > 0 else 1.0
        )
        score += idf * (tf * (BM25_K1 + 1)) / denom
    return score


def _dense_search(
    query_embedding: list[float], index: dict, top_k: int = 20
) -> list[tuple[str, float]]:
    """Cosine similarity search over all chunk embeddings."""
    try:
        import numpy as np
    except ImportError as err:
        raise ImportError("numpy not installed.") from err

    q = np.array(query_embedding)
    q_norm = np.linalg.norm(q)
    results: list[tuple[str, float]] = []
    for chunk_id, chunk in index["chunks"].items():
        emb = np.array(chunk["embedding"])
        emb_norm = np.linalg.norm(emb)
        if q_norm == 0 or emb_norm == 0:
            sim = 0.0
        else:
            sim = float(np.dot(q, emb) / (q_norm * emb_norm))
        results.append((chunk_id, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _sparse_search(
    query_tokens: list[str], index: dict, top_k: int = 20
) -> list[tuple[str, float]]:
    """BM25 search over all chunks."""
    results: list[tuple[str, float]] = []
    for chunk_id, chunk in index["chunks"].items():
        score = _bm25_score(query_tokens, chunk, index)
        results.append((chunk_id, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def _rrf_merge(
    rankings: list[list[tuple[str, float]]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    rrf_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def _get_cached_model():
    """Get or create cached fastembed model (module-level singleton)."""
    global _cached_model  # noqa: PLW0603
    if _cached_model is None:
        from fastembed import TextEmbedding

        _cached_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _cached_model


def hybrid_search(query: str, index: dict, top_k: int = 5) -> list[dict]:
    """Hybrid BM25 + dense search with RRF merging.

    Returns top_k results as dicts with chunk metadata and rrf_score.
    """
    model = _get_cached_model()
    query_embedding = list(model.embed([query]))[0].tolist()
    query_tokens = _tokenize(query)

    dense_results = _dense_search(query_embedding, index, top_k=20)
    sparse_results = _sparse_search(query_tokens, index, top_k=20)
    merged = _rrf_merge([dense_results, sparse_results], k=RRF_K)

    output: list[dict] = []
    for chunk_id, rrf_score in merged[:top_k]:
        chunk = index["chunks"].get(chunk_id, {})
        output.append(
            {
                "chunk_id": chunk_id,
                "source_file": chunk.get("source_file", ""),
                "section": chunk.get("section", ""),
                "content": chunk.get("content", ""),
                "rrf_score": rrf_score,
            }
        )
    return output


# ---------------------------------------------------------------------------
# CrewAI Tools
# ---------------------------------------------------------------------------


class KnowledgeQueryTool(BaseTool):
    """Search the internal knowledge base with hybrid BM25 + dense retrieval."""

    name: str = "query_knowledge_base"
    description: str = (
        "Search the internal knowledge base (reports, assay data, SOPs, safety docs) "
        "for information relevant to your hypothesis. Returns top results with source "
        "attribution. Use specific queries like 'BBB physicochemical property ranges' "
        "or 'historical model performance ensemble methods'."
    )
    top_k: int = 5

    # Cached index (rebuilt once per session)
    _index: ClassVar[dict | None] = None

    @classmethod
    def _get_index(cls) -> dict | None:
        """Load or build the knowledge base index."""
        if cls._index is not None:
            return cls._index
        kb_dir = get_kb_dir()
        index_path = kb_dir / "index.json"
        if index_path.exists():
            try:
                cls._index = json.loads(index_path.read_text(encoding="utf-8"))
                return cls._index
            except (json.JSONDecodeError, ValueError):
                pass
        # Auto-build if files exist but no index
        if kb_dir.exists() and (
            list(kb_dir.glob("**/*.md")) or list(kb_dir.glob("**/*.csv"))
        ):
            cls._index = _build_index(kb_dir)
            _write_index(index_path, cls._index)
            return cls._index
        return None

    def _run(self, query: str) -> str:
        """Search the knowledge base with hybrid retrieval."""
        index = self._get_index()
        if index is None:
            return "No knowledge base found. Add documents to the knowledge_base directory."
        chunks = index.get("chunks", {})
        if not chunks:
            return "Knowledge base is empty."

        results = hybrid_search(query, index, top_k=self.top_k)
        if not results:
            return f"No results found for '{query}'."

        lines = [f"Top {len(results)} results for '{query}':\n"]
        for r in results:
            source = r["source_file"]
            section = r["section"]
            content = r["content"][:300]
            score = r["rrf_score"]
            lines.append(
                f"--- [Source: {source} § {section}] (score: {score:.4f}) ---\n"
                f"{content}...\n"
            )
        return "\n".join(lines)


def rebuild_index(kb_dir: Path | None = None) -> str:
    """Rebuild the knowledge base index from all source documents.

    Called by the CLI (ingest_kb module), not by agents.
    """
    kb_dir = kb_dir or get_kb_dir()
    if not kb_dir.exists():
        return f"Knowledge base directory not found: {kb_dir}"

    index = _build_index(kb_dir)
    index_path = kb_dir / "index.json"
    _write_index(index_path, index)

    # Clear cached index so next query picks up changes
    KnowledgeQueryTool._index = None

    n_chunks = len(index.get("chunks", {}))
    md_files = list(kb_dir.glob("**/*.md"))
    csv_files = list(kb_dir.glob("**/*.csv"))
    n_docs = len(md_files) + len(csv_files)
    return f"Indexed {n_chunks} chunks from {n_docs} documents."
