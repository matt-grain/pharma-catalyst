"""Tests for the AutoGen expert review panel."""

import json
from unittest.mock import MagicMock, patch

import pytest

from pharma_agents.crew import HypothesisOutput
from pharma_agents.review_panel import (
    ReviewVerdict,
    _parse_verdict_from_json,
    _parse_verdict_from_text,
    format_approved_hypothesis,
    run_review_panel,
)


# --- ReviewVerdict model tests ---


def test_review_verdict_approved():
    verdict = ReviewVerdict(
        decision="approved",
        feedback="Looks good",
        confidence=0.9,
        concerns=["minor concern"],
    )
    assert verdict.decision == "approved"
    assert verdict.revised_proposal is None


def test_review_verdict_revised_requires_proposal():
    verdict = ReviewVerdict(
        decision="revised",
        revised_proposal="Use XGBoost instead",
        revised_reasoning="Better for small datasets",
        feedback="Good idea but needs adjustment",
        confidence=0.7,
        concerns=["overfitting risk"],
    )
    assert verdict.decision == "revised"
    assert verdict.revised_proposal == "Use XGBoost instead"


def test_review_verdict_rejected():
    verdict = ReviewVerdict(
        decision="rejected",
        feedback="Fundamentally flawed approach",
        confidence=0.95,
        concerns=["data leakage", "overfitting"],
    )
    assert verdict.decision == "rejected"


def test_review_verdict_invalid_decision():
    with pytest.raises(Exception):
        ReviewVerdict(
            decision="maybe",
            feedback="unsure",
            confidence=0.5,
            concerns=[],
        )


# --- JSON parsing tests ---


def test_parse_verdict_from_json_with_code_block():
    text = '''Here is my verdict:
```json
{
    "decision": "approved",
    "feedback": "Solid proposal",
    "confidence": 0.85,
    "concerns": ["minor risk"]
}
```'''
    verdict = _parse_verdict_from_json(text)
    assert verdict is not None
    assert verdict.decision == "approved"
    assert verdict.confidence == 0.85


def test_parse_verdict_from_json_raw():
    text = '{"decision": "rejected", "feedback": "Bad idea", "confidence": 0.9, "concerns": ["fatal flaw"]}'
    verdict = _parse_verdict_from_json(text)
    assert verdict is not None
    assert verdict.decision == "rejected"


def test_parse_verdict_from_json_invalid():
    verdict = _parse_verdict_from_json("No JSON here at all")
    assert verdict is None


def test_parse_verdict_from_json_malformed():
    verdict = _parse_verdict_from_json('{"decision": "approved", bad json}')
    assert verdict is None


# --- Text fallback parsing tests ---


def test_parse_verdict_from_text_approved():
    verdict = _parse_verdict_from_text("I think this should be approved because...")
    assert verdict.decision == "approved"
    assert verdict.confidence == 0.3


def test_parse_verdict_from_text_rejected():
    verdict = _parse_verdict_from_text("This proposal should be rejected due to data leakage")
    assert verdict.decision == "rejected"


def test_parse_verdict_from_text_revised():
    verdict = _parse_verdict_from_text("The proposal needs to be revised to address...")
    assert verdict.decision == "revised"


def test_parse_verdict_from_text_no_keyword_defaults_approved():
    verdict = _parse_verdict_from_text("I have no strong opinion on this matter")
    assert verdict.decision == "approved"


def test_parse_verdict_from_text_truncates_long_feedback():
    long_text = "x" * 1000
    verdict = _parse_verdict_from_text(long_text)
    assert len(verdict.feedback) == 500


# --- format_approved_hypothesis tests ---


def _make_hypothesis(**kwargs) -> HypothesisOutput:
    defaults = {
        "proposal": "Add RDKit descriptors",
        "reasoning": "More molecular features",
        "change_description": "Compute 200 descriptors",
        "literature_insight": "Recent papers suggest this",
    }
    defaults.update(kwargs)
    return HypothesisOutput(**defaults)


def test_format_approved_hypothesis_approved():
    verdict = ReviewVerdict(
        decision="approved",
        feedback="Panel agrees",
        confidence=0.9,
        concerns=[],
    )
    text = format_approved_hypothesis(verdict, _make_hypothesis())
    assert "APPROVED PROPOSAL:" in text
    assert "Add RDKit descriptors" in text
    assert "PANEL FEEDBACK: Panel agrees" in text


def test_format_approved_hypothesis_revised():
    verdict = ReviewVerdict(
        decision="revised",
        revised_proposal="Add only top-50 RDKit descriptors",
        feedback="Too many features, reduce",
        confidence=0.8,
        concerns=["feature explosion"],
    )
    text = format_approved_hypothesis(verdict, _make_hypothesis())
    assert "revised by review panel" in text
    assert "Add only top-50 RDKit descriptors" in text
    assert "ORIGINAL PROPOSAL: Add RDKit descriptors" in text


# --- run_review_panel integration test (mocked AG2) ---


@patch("autogen.GroupChatManager")
@patch("autogen.GroupChat")
@patch("autogen.ConversableAgent")
def test_run_review_panel_returns_verdict(
    mock_agent_cls, mock_gc_cls, mock_manager_cls
):
    """Test that run_review_panel returns a valid verdict when AG2 works."""
    # Setup: mock the GroupChat to return a moderator message with JSON
    mock_gc = MagicMock()
    mock_gc.messages = [
        {
            "name": "Moderator",
            "role": "assistant",
            "content": json.dumps(
                {
                    "decision": "approved",
                    "feedback": "Solid approach",
                    "confidence": 0.85,
                    "concerns": ["minor overfitting risk"],
                }
            ),
        }
    ]
    mock_gc_cls.return_value = mock_gc

    # Mock agent to not actually call LLM
    mock_agent = MagicMock()
    mock_agent_cls.return_value = mock_agent

    verdict = run_review_panel(
        hypothesis=_make_hypothesis(),
        experiment_context="No previous experiments",
        baseline_score=0.85,
        metric="AUC-ROC",
        direction="higher_is_better",
    )

    assert verdict.decision == "approved"
    assert verdict.confidence == 0.85


@patch("autogen.GroupChatManager")
@patch("autogen.GroupChat")
@patch("autogen.ConversableAgent")
def test_run_review_panel_handles_error_gracefully(
    mock_agent_cls, mock_gc_cls, mock_manager_cls
):
    """Test that AG2 errors result in a safe default (approved with low confidence)."""
    mock_agent = MagicMock()
    mock_agent.initiate_chat.side_effect = RuntimeError("LLM API error")
    mock_agent_cls.return_value = mock_agent

    verdict = run_review_panel(
        hypothesis=_make_hypothesis(),
        experiment_context="",
        baseline_score=1.0,
        metric="RMSE",
        direction="lower_is_better",
    )

    assert verdict.decision == "approved"
    assert verdict.confidence <= 0.3


@patch("autogen.GroupChatManager")
@patch("autogen.GroupChat")
@patch("autogen.ConversableAgent")
def test_run_review_panel_reraises_rate_limit_errors(
    mock_agent_cls, mock_gc_cls, mock_manager_cls
):
    """Rate limit errors must propagate to main.py's retry handler, not be swallowed."""
    mock_agent = MagicMock()
    mock_agent.initiate_chat.side_effect = RuntimeError(
        "429 RESOURCE_EXHAUSTED. quota exceeded"
    )
    mock_agent_cls.return_value = mock_agent

    with pytest.raises(RuntimeError, match="429"):
        run_review_panel(
            hypothesis=_make_hypothesis(),
            experiment_context="",
            baseline_score=1.0,
            metric="RMSE",
            direction="lower_is_better",
        )
