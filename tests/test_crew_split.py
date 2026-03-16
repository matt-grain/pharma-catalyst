"""Tests for the split crew methods (hypothesis_crew, implementation_crew)."""

from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def crew_instance():
    """Create a PharmaAgentsCrew with mocked LLM."""
    mock_llm = MagicMock()
    mock_llm.model = "gemini/gemini-2.0-flash"

    with patch("pharma_agents.crew.get_llm", return_value=mock_llm):
        with patch("pharma_agents.crew.LoggingLLM", return_value=mock_llm):
            with patch.dict(
                "os.environ",
                {
                    "PHARMA_EXPERIMENT": "bbbp",
                    "LLM_MODEL": "gemini/gemini-2.0-flash",
                    "GOOGLE_API_KEY": "test-key",
                },
            ):
                from pharma_agents.crew import PharmaAgentsCrew

                return PharmaAgentsCrew()


def test_hypothesis_crew_has_one_agent(crew_instance):
    """hypothesis_crew should contain only the hypothesis agent."""
    h_crew = crew_instance.hypothesis_crew()
    assert len(h_crew.agents) == 1
    assert len(h_crew.tasks) == 1


def test_implementation_crew_has_two_agents(crew_instance):
    """implementation_crew should contain model + evaluator agents."""
    impl_crew = crew_instance.implementation_crew()
    assert len(impl_crew.agents) == 2
    assert len(impl_crew.tasks) == 2


def test_original_crew_still_works(crew_instance):
    """The original crew() method should still return a 3-agent crew."""
    original = crew_instance.crew()
    assert len(original.agents) == 3
    assert len(original.tasks) == 3
