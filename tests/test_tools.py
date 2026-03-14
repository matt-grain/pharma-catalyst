"""
Integration tests for pharma-agents tools.

These tests hit real APIs to verify the tools actually work.
Run with: uv run pytest tests/test_tools.py -v
"""

import json
import os
import pytest

# Set a test experiment directory
TEST_EXPERIMENT = "test_experiment"


@pytest.fixture(scope="module")
def experiments_dir(tmp_path_factory):
    """Create a temporary experiments directory."""
    exp_dir = tmp_path_factory.mktemp("experiments") / TEST_EXPERIMENT
    exp_dir.mkdir(parents=True)

    # Create a minimal baseline.json
    baseline = {
        "metric": "RMSE",
        "score": 1.0,
        "direction": "lower_is_better",
        "property": "test property",
    }
    (exp_dir / "baseline.json").write_text(json.dumps(baseline))

    # Set environment for tools
    os.environ["PHARMA_EXPERIMENT"] = TEST_EXPERIMENT
    os.environ["PHARMA_EXPERIMENTS_DIR"] = str(exp_dir)
    os.environ["PHARMA_LITERATURE_DIR"] = str(exp_dir / "literature")

    yield exp_dir

    # Cleanup
    if "PHARMA_EXPERIMENT" in os.environ:
        del os.environ["PHARMA_EXPERIMENT"]
    if "PHARMA_EXPERIMENTS_DIR" in os.environ:
        del os.environ["PHARMA_EXPERIMENTS_DIR"]
    if "PHARMA_LITERATURE_DIR" in os.environ:
        del os.environ["PHARMA_LITERATURE_DIR"]


class TestArxivSearchTool:
    """Test ArxivSearchTool against real arxiv API."""

    @pytest.mark.timeout(30)
    def test_search_returns_results(self):
        """Search arxiv for a common ML topic."""
        from pharma_agents.tools.custom_tools import ArxivSearchTool

        tool = ArxivSearchTool()
        result = tool._run("molecular property prediction")

        assert "Found" in result
        assert "papers" in result
        # Should contain paper IDs in format YYMM.NNNNN
        assert "**" in result  # Markdown bold for paper IDs

    @pytest.mark.timeout(30)
    def test_search_admet_topic(self):
        """Search for ADMET-specific papers."""
        from pharma_agents.tools.custom_tools import ArxivSearchTool

        tool = ArxivSearchTool()
        result = tool._run("ADMET prediction deep learning")

        assert "Found" in result or "No papers found" in result
        # Either we find papers or gracefully report none

    @pytest.mark.timeout(30)
    def test_search_limit_respected(self):
        """Verify max_searches_per_run limit."""
        from pharma_agents.tools.custom_tools import ArxivSearchTool

        # Reset the class counter
        ArxivSearchTool._searches_done = 0

        tool = ArxivSearchTool()
        tool.max_searches_per_run = 2

        # First two should work
        tool._run("test query 1")
        tool._run("test query 2")

        # Third should be blocked
        result = tool._run("test query 3")
        assert "STOP SEARCHING" in result


class TestAlphaxivTool:
    """Test AlphaxivTool against real alphaxiv API."""

    @pytest.mark.timeout(30)
    def test_fetch_known_paper(self):
        """Fetch a well-known paper that should exist on alphaxiv."""
        from pharma_agents.tools.custom_tools import AlphaxivTool

        # Reset counters
        AlphaxivTool._papers_fetched = 0

        tool = AlphaxivTool()
        # Use a popular paper that's likely indexed
        result = tool._run("1706.03762")  # Attention Is All You Need

        # Either we get content or a 404 (not all papers are indexed)
        assert "Paper" in result or "not found" in result.lower()

    @pytest.mark.timeout(30)
    def test_fetch_with_url_prefix(self):
        """Test that URL prefixes are stripped correctly."""
        from pharma_agents.tools.custom_tools import AlphaxivTool

        AlphaxivTool._papers_fetched = 0

        tool = AlphaxivTool()
        result = tool._run("https://arxiv.org/abs/1706.03762")

        # Should work the same as bare ID
        assert "Paper" in result or "not found" in result.lower()

    @pytest.mark.timeout(30)
    def test_rate_limiting(self):
        """Verify rate limiting works (doesn't error)."""
        from pharma_agents.tools.custom_tools import AlphaxivTool
        import time

        AlphaxivTool._papers_fetched = 0

        tool = AlphaxivTool()
        tool.min_interval_seconds = 0.5

        start = time.time()
        tool._run("1706.03762")
        tool._run("1706.03762")  # Same paper, tests rate limiting
        elapsed = time.time() - start

        # Should take at least 0.5 seconds due to rate limiting
        assert elapsed >= 0.4


class TestLiteratureStoreTool:
    """Test LiteratureStoreTool with real embeddings."""

    @pytest.mark.timeout(60)
    def test_store_paper_creates_files(self, experiments_dir):
        """Store a paper and verify files are created."""
        from pharma_agents.tools.custom_tools import LiteratureStoreTool

        tool = LiteratureStoreTool()

        paper_data = json.dumps(
            {
                "paper_id": "test.12345",
                "title": "Test Paper on Molecular ML",
                "summary": "This paper presents a novel approach to molecular property prediction using graph neural networks.",
                "key_methods": ["GNN", "Morgan fingerprints", "attention pooling"],
            }
        )

        result = tool._run(paper_data)

        assert "Stored paper" in result
        assert "384 dims" in result  # BGE-small embedding size

        # Verify files exist
        lit_dir = experiments_dir / "literature"
        assert (lit_dir / "index.json").exists()
        assert (lit_dir / "papers" / "test.12345.md").exists()

        # Verify index content
        index = json.loads((lit_dir / "index.json").read_text())
        assert "test.12345" in index["papers"]
        assert len(index["papers"]["test.12345"]["embedding"]) == 384

    @pytest.mark.timeout(60)
    def test_store_multiple_papers(self, experiments_dir):
        """Store multiple papers and verify they accumulate."""
        from pharma_agents.tools.custom_tools import LiteratureStoreTool

        tool = LiteratureStoreTool()

        # Store first paper
        tool._run(
            json.dumps(
                {
                    "paper_id": "multi.001",
                    "title": "First Paper",
                    "summary": "Summary one",
                }
            )
        )

        # Store second paper
        tool._run(
            json.dumps(
                {
                    "paper_id": "multi.002",
                    "title": "Second Paper",
                    "summary": "Summary two",
                }
            )
        )

        # Verify both are in index
        lit_dir = experiments_dir / "literature"
        index = json.loads((lit_dir / "index.json").read_text())

        assert "multi.001" in index["papers"]
        assert "multi.002" in index["papers"]

    @pytest.mark.timeout(60)
    def test_store_from_alphaxiv_markdown(self, experiments_dir):
        """Store paper from raw alphaxiv markdown format."""
        from pharma_agents.tools.custom_tools import LiteratureStoreTool

        tool = LiteratureStoreTool()

        # Alphaxiv format with Summary and Key Techniques
        alphaxiv_markdown = """=== Paper 2310.07351 ===
Title: Atom-Motif Contrastive Transformer for Molecular Property Prediction
Summary: Proposes AMCT, a Graph Transformer that considers both atom-level and motif-level (functional group) interactions to capture critical molecular patterns.
Key Techniques: Motif-level interaction modeling, Atom-Motif Contrastive learning, Property-aware attention mechanism, Graph Transformer (GT) backbone.
"""

        result = tool._run(alphaxiv_markdown)

        assert "Stored paper" in result
        assert "2310.07351" in result

        # Verify correct extraction
        lit_dir = experiments_dir / "literature"
        index = json.loads((lit_dir / "index.json").read_text())

        paper = index["papers"]["2310.07351"]
        assert "Atom-Motif Contrastive Transformer" in paper["title"]
        assert "AMCT" in paper["summary"]
        assert len(paper["key_methods"]) >= 3


class TestLiteratureQueryTool:
    """Test LiteratureQueryTool with real semantic search."""

    @pytest.mark.timeout(60)
    def test_query_empty_db(self, experiments_dir):
        """Query when no literature exists."""
        from pharma_agents.tools.custom_tools import LiteratureQueryTool

        # Remove any existing literature
        lit_dir = experiments_dir / "literature"
        if (lit_dir / "index.json").exists():
            (lit_dir / "index.json").unlink()

        tool = LiteratureQueryTool()
        result = tool._run("test query")

        assert "No literature database" in result or "empty" in result.lower()

    @pytest.mark.timeout(90)
    def test_query_finds_relevant_papers(self, experiments_dir):
        """Store papers and verify semantic search finds relevant ones."""
        from pharma_agents.tools.custom_tools import (
            LiteratureStoreTool,
            LiteratureQueryTool,
        )

        store_tool = LiteratureStoreTool()

        # Store papers with different topics
        store_tool._run(
            json.dumps(
                {
                    "paper_id": "gnn.001",
                    "title": "Graph Neural Networks for ADMET",
                    "summary": "We use graph neural networks with attention mechanisms for ADMET property prediction on molecular graphs.",
                }
            )
        )

        store_tool._run(
            json.dumps(
                {
                    "paper_id": "nlp.001",
                    "title": "Transformers for NLP",
                    "summary": "Large language models and transformers for natural language processing and text generation.",
                }
            )
        )

        store_tool._run(
            json.dumps(
                {
                    "paper_id": "mol.001",
                    "title": "Morgan Fingerprints Analysis",
                    "summary": "Molecular fingerprints like Morgan and MACCS keys for cheminformatics and drug discovery.",
                }
            )
        )

        # Query for molecular topics
        query_tool = LiteratureQueryTool()
        result = query_tool._run("molecular fingerprints drug discovery")

        # The molecular/fingerprint paper should rank higher than NLP paper
        assert "mol.001" in result or "gnn.001" in result
        # Check that we get similarity scores
        assert "[0." in result  # Similarity scores like [0.87]


class TestSkillLoaderTool:
    """Test SkillLoaderTool."""

    def test_load_existing_skill(self):
        """Load a skill that exists."""
        from pharma_agents.tools.custom_tools import SkillLoaderTool

        SkillLoaderTool._skills_loaded = []  # Reset class-level state
        tool = SkillLoaderTool()

        result = tool._run("rdkit")

        assert "SKILL: rdkit" in result or "not found" in result.lower()

    def test_load_nonexistent_skill(self):
        """Try to load a skill that doesn't exist."""
        from pharma_agents.tools.custom_tools import SkillLoaderTool

        SkillLoaderTool._skills_loaded = []  # Reset class-level state
        tool = SkillLoaderTool()

        # Use valid format (alphanumeric + hyphens only)
        result = tool._run("nonexistent-skill-xyz")

        assert "not found" in result.lower()
        assert "Available:" in result

    def test_skill_limit_respected(self):
        """Verify max_skills_per_run limit."""
        from pharma_agents.tools.custom_tools import SkillLoaderTool

        SkillLoaderTool._skills_loaded = []  # Reset class-level state
        tool = SkillLoaderTool()
        tool.max_skills_per_run = 2

        # Load two skills
        tool._run("rdkit")
        tool._run("deepchem")

        # Third should be blocked
        result = tool._run("molfeat")
        assert "Max skills limit" in result

    def test_path_traversal_blocked(self):
        """Verify path traversal attempts are rejected."""
        from pharma_agents.tools.custom_tools import SkillLoaderTool

        SkillLoaderTool._skills_loaded = []
        tool = SkillLoaderTool()

        # Try path traversal patterns
        for malicious in ["../../../etc/passwd", "..\\..\\secrets", "skill/../bad"]:
            result = tool._run(malicious)
            assert "Invalid skill name" in result


class TestEndToEndWorkflow:
    """Test the full Archivist workflow."""

    @pytest.mark.timeout(120)
    def test_search_fetch_store_query(self, experiments_dir):
        """Full workflow: search -> fetch -> store -> query."""
        from pharma_agents.tools.custom_tools import (
            ArxivSearchTool,
            AlphaxivTool,
            LiteratureStoreTool,
            LiteratureQueryTool,
        )

        # Reset counters
        ArxivSearchTool._searches_done = 0
        AlphaxivTool._papers_fetched = 0

        # 1. Search arxiv
        search_tool = ArxivSearchTool()
        search_result = search_tool._run("molecular property prediction GNN")
        print(f"Search result:\n{search_result[:500]}...")

        assert "Found" in search_result or "papers" in search_result.lower()

        # 2. Store a synthetic paper (since alphaxiv might not have all papers)
        store_tool = LiteratureStoreTool()
        store_tool._run(
            json.dumps(
                {
                    "paper_id": "e2e.test",
                    "title": "End-to-End Test: GNN for Molecular Property Prediction",
                    "summary": "This test paper validates the full archivist workflow with graph neural networks for predicting molecular properties like solubility and ADMET.",
                    "key_methods": ["GNN", "message passing", "molecular graphs"],
                }
            )
        )

        # 3. Query the stored paper
        query_tool = LiteratureQueryTool()
        query_result = query_tool._run("GNN molecular solubility prediction")
        print(f"Query result:\n{query_result}")

        assert "e2e.test" in query_result
        assert "[0." in query_result  # Similarity score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
