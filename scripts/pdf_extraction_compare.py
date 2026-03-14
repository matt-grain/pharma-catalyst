#!/usr/bin/env python
"""
Compare PDF extraction tools for arxiv papers.

Tools tested:
- pymupdf (basic text extraction)
- pymupdf4llm (LLM-optimized markdown output)
- marker (if installed - best quality but heavy)
- docling (if installed - MIT license, good tables)

Usage:
    uv run python scripts/pdf_extraction_compare.py [arxiv_id]

Example:
    uv run python scripts/pdf_extraction_compare.py 2507.18557v3
"""

import sys
import time
import tempfile
from pathlib import Path


def download_arxiv_pdf(paper_id: str) -> Path:
    """Download PDF from arxiv."""
    import urllib.request

    url = f"https://arxiv.org/pdf/{paper_id}"
    pdf_path = Path(tempfile.gettempdir()) / f"arxiv_{paper_id.replace('/', '_')}.pdf"

    if not pdf_path.exists():
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, pdf_path)
        print(f"Saved to {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"Using cached {pdf_path}")

    return pdf_path


def extract_pymupdf(pdf_path: Path, pages: int = 5) -> tuple[str, float]:
    """Basic pymupdf text extraction."""
    import pymupdf

    start = time.perf_counter()
    doc = pymupdf.open(str(pdf_path))
    text = ""
    for page in doc[:pages]:
        text += page.get_text()
    elapsed = time.perf_counter() - start

    return text, elapsed


def extract_pymupdf4llm(pdf_path: Path, pages: int = 5) -> tuple[str, float]:
    """pymupdf4llm markdown extraction."""
    import pymupdf4llm

    start = time.perf_counter()
    # Extract as markdown with page limit
    md = pymupdf4llm.to_markdown(str(pdf_path), pages=list(range(pages)))
    elapsed = time.perf_counter() - start

    return md, elapsed


def extract_marker(pdf_path: Path, pages: int = 5) -> tuple[str, float] | None:
    """Marker PDF extraction (best quality, requires model download)."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError:
        return None

    start = time.perf_counter()
    models = create_model_dict()
    converter = PdfConverter(artifact_dict=models)
    result = converter(str(pdf_path))
    elapsed = time.perf_counter() - start

    return result.markdown, elapsed


def extract_docling(pdf_path: Path, pages: int = 5) -> tuple[str, float] | None:
    """Docling extraction (MIT license, good tables)."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return None

    start = time.perf_counter()
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    md = result.document.export_to_markdown()
    elapsed = time.perf_counter() - start

    return md, elapsed


def find_section(text: str, keywords: list[str], context: int = 1500) -> str:
    """Find text around keywords."""
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw.lower())
        if idx > 0:
            start = max(0, idx - 200)
            end = min(len(text), idx + context)
            return f"[Found '{kw}' at position {idx}]\n{text[start:end]}"
    return "[Keywords not found]"


def main():
    paper_id = sys.argv[1] if len(sys.argv) > 1 else "2507.18557v3"
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"\n{'=' * 60}")
    print(f"PDF Extraction Comparison: arxiv {paper_id}")
    print(f"{'=' * 60}\n")

    pdf_path = download_arxiv_pdf(paper_id)

    results = {}

    # 1. Basic pymupdf
    print("\n--- pymupdf (basic) ---")
    text, elapsed = extract_pymupdf(pdf_path, pages)
    results["pymupdf"] = {"text": text, "time": elapsed, "chars": len(text)}
    print(f"Time: {elapsed:.3f}s | Chars: {len(text)}")
    print(find_section(text, ["fingerprint", "ECFP", "attention", "GNN"]))

    # 2. pymupdf4llm
    print("\n--- pymupdf4llm (markdown) ---")
    md, elapsed = extract_pymupdf4llm(pdf_path, pages)
    results["pymupdf4llm"] = {"text": md, "time": elapsed, "chars": len(md)}
    print(f"Time: {elapsed:.3f}s | Chars: {len(md)}")
    print(find_section(md, ["fingerprint", "ECFP", "attention", "GNN"]))

    # 3. Marker (if available)
    print("\n--- marker (best quality) ---")
    marker_result = extract_marker(pdf_path, pages)
    if marker_result:
        md, elapsed = marker_result
        results["marker"] = {"text": md, "time": elapsed, "chars": len(md)}
        print(f"Time: {elapsed:.3f}s | Chars: {len(md)}")
        print(find_section(md, ["fingerprint", "ECFP", "attention", "GNN"]))
    else:
        print("Not installed (uv add marker-pdf)")

    # 4. Docling (if available)
    print("\n--- docling (MIT license) ---")
    docling_result = extract_docling(pdf_path, pages)
    if docling_result:
        md, elapsed = docling_result
        results["docling"] = {"text": md, "time": elapsed, "chars": len(md)}
        print(f"Time: {elapsed:.3f}s | Chars: {len(md)}")
        print(find_section(md, ["fingerprint", "ECFP", "attention", "GNN"]))
    else:
        print("Not installed (uv add docling)")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for name, data in results.items():
        print(f"{name:15} | Time: {data['time']:.3f}s | Chars: {data['chars']:,}")

    # Save outputs for comparison
    output_dir = Path("scripts/pdf_outputs")
    output_dir.mkdir(exist_ok=True)
    for name, data in results.items():
        out_file = output_dir / f"{paper_id.replace('/', '_')}_{name}.md"
        out_file.write_text(data["text"], encoding="utf-8")
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
