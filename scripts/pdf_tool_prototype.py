"""Prototype: Compare CrewAI PDFSearchTool vs our approach."""

import time
from pathlib import Path
import urllib.request

# Download a test PDF (arxiv paper on BBB prediction)
PDF_URL = "https://arxiv.org/pdf/2107.06773.pdf"
PDF_PATH = Path("scripts/test_paper.pdf")


def download_pdf():
    """Download test PDF if not exists."""
    if not PDF_PATH.exists():
        print(f"Downloading {PDF_URL}...")
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
        print(f"Saved to {PDF_PATH}")
    else:
        print(f"Using cached {PDF_PATH}")
    return PDF_PATH


def test_crewai_pdfsearchtool():
    """Test CrewAI's PDFSearchTool."""
    try:
        from crewai_tools import PDFSearchTool
    except ImportError:
        print("PDFSearchTool not available. Install with: uv add 'crewai[tools]'")
        return None

    pdf_path = download_pdf()

    # Try with HuggingFace embeddings (no API key needed)
    # Note: This might still need configuration
    print("\n=== CrewAI PDFSearchTool ===")
    start = time.perf_counter()

    try:
        # Try with HuggingFace embeddings (local, no API key)
        tool = PDFSearchTool(
            pdf=str(pdf_path),
            config={
                "embedding_model": {
                    "provider": "huggingface",
                    "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
                },
            },
        )

        # Search for BBB-related content
        result = tool._run("What methods are used for blood-brain barrier prediction?")

        elapsed = time.perf_counter() - start
        print(f"Time: {elapsed:.2f}s")
        print(f"Result: {result[:500]}...")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_pymupdf4llm():
    """Test our approach: pymupdf4llm for extraction."""
    try:
        import pymupdf4llm
    except ImportError:
        print("pymupdf4llm not available. Install with: uv add pymupdf4llm")
        return None

    pdf_path = download_pdf()

    print("\n=== pymupdf4llm (our approach) ===")
    start = time.perf_counter()

    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    elapsed = time.perf_counter() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Length: {len(md_text)} chars")
    print(f"Preview: {md_text[:500]}...")

    return md_text


def test_basic_pymupdf():
    """Test basic pymupdf extraction."""
    try:
        import pymupdf
    except ImportError:
        print("pymupdf not available")
        return None

    pdf_path = download_pdf()

    print("\n=== pymupdf (basic) ===")
    start = time.perf_counter()

    doc = pymupdf.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text()

    elapsed = time.perf_counter() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Length: {len(text)} chars")
    print(f"Preview: {text[:500]}...")

    return text


if __name__ == "__main__":
    print("PDF Tool Comparison Prototype")
    print("=" * 50)

    # Test each approach
    test_basic_pymupdf()
    test_pymupdf4llm()
    test_crewai_pdfsearchtool()
