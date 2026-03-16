"""
AutoGen expert review panel for hypothesis validation.

Uses AG2's GroupChat to run an adversarial multi-perspective debate
between domain experts before committing to model training.

Why AutoGen instead of more CrewAI agents:
  CrewAI excels at sequential task pipelines. But adversarial debate —
  multiple agents arguing over the same topic, building on each other's
  points — is a fundamentally different interaction pattern. AutoGen's
  GroupChat is purpose-built for this.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from .crew import HypothesisOutput, get_api_key, get_provider, is_free_tier
from .log_utils import log_box
from .review_config import (
    REVIEW_TEMPERATURE,
    get_agent_config,
    get_agent_keys_ordered,
    get_max_rounds,
)

# Free tier: 20 RPM shared across CrewAI + AG2.
# CrewAI's LoggingLLM enforces a 4s gap between its calls, but AG2 has its own
# LLM client so we need a separate cooldown + per-call delay here.
_FREE_TIER_COOLDOWN: float = 10.0  # Let RPM window reset after CrewAI calls
_FREE_TIER_DELAY: float = 5.0  # Between each AG2 agent's LLM call


class ReviewVerdict(BaseModel):
    """Structured output from the expert review panel."""

    decision: Literal["approved", "revised", "rejected"]
    revised_proposal: str | None = None
    revised_reasoning: str | None = None
    feedback: str
    confidence: float
    concerns: list[str]


# AG2 provider configs: provider → (api_type, base_url, strip_prefix)
_AG2_PROVIDERS: dict[str, dict] = {
    "gemini": {"api_type": "google"},
    "groq": {"base_url": "https://api.groq.com/openai/v1"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1"},
    "cerebras": {"base_url": "https://api.cerebras.ai/v1"},
    "openai": {},
    "anthropic": {"api_type": "anthropic"},
}


def _build_llm_config() -> dict:
    """Build AG2 LLM config from environment variables (provider-agnostic).

    Detects the provider from LLM_MODEL prefix and configures AG2 accordingly.
    For Google: uses api_type='google' with bare model name.
    For OpenAI-compatible providers (Groq, etc.): uses base_url.
    """
    model = os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash")
    provider = get_provider(model)
    api_key = get_api_key(model)

    # Strip provider prefix — AG2 needs bare model name
    bare_model = model.split("/", 1)[1] if "/" in model else model

    config_entry: dict = {
        "model": bare_model,
        "api_key": api_key,
    }

    # Apply provider-specific settings
    provider_config = _AG2_PROVIDERS.get(provider, {})
    config_entry.update(provider_config)

    # Honor OPENAI_API_BASE for OpenAI-compatible providers (e.g. mock server)
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base and "base_url" not in config_entry and "api_type" not in config_entry:
        config_entry["base_url"] = api_base

    return {
        "config_list": [config_entry],
        "temperature": REVIEW_TEMPERATURE,
    }


def _parse_verdict_from_json(text: str) -> ReviewVerdict | None:
    """Try to parse ReviewVerdict from JSON in the moderator's response."""
    # Extract JSON from markdown code blocks or raw text
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{[^{}]*\"decision\"[^{}]*\}", text, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        else:
            return None

    try:
        data = json.loads(raw)
        return ReviewVerdict(**data)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _parse_verdict_from_text(text: str) -> ReviewVerdict:
    """Fallback: extract verdict from free text when JSON parsing fails."""
    decision: Literal["approved", "revised", "rejected"] = "approved"
    for keyword in ("rejected", "revised", "approved"):
        if keyword in text.lower():
            decision = keyword  # type: ignore[assignment]
            break

    return ReviewVerdict(
        decision=decision,
        feedback=text[:500],
        confidence=0.3,
        concerns=[],
    )


def _format_review_prompt(
    hypothesis: HypothesisOutput,
    experiment_context: str,
    baseline_score: float,
    metric: str,
    direction: str,
) -> str:
    """Format the hypothesis into a debate prompt for the panel."""
    return (
        f"## Proposal Under Review\n\n"
        f"**Proposal:** {hypothesis.proposal}\n"
        f"**Reasoning:** {hypothesis.reasoning}\n"
        f"**Change Description:** {hypothesis.change_description}\n"
        f"**Literature Insight:** {hypothesis.literature_insight or 'None'}\n\n"
        f"## Context\n\n"
        f"- Current baseline {metric}: {baseline_score:.4f}\n"
        f"- Direction: {direction}\n\n"
        f"## Experiment History\n\n"
        f"{experiment_context}\n\n"
        f"---\n"
        f"Please evaluate this proposal from your expert perspective."
    )


def format_approved_hypothesis(
    verdict: ReviewVerdict, hypothesis: HypothesisOutput
) -> str:
    """Format the approved/revised hypothesis as text for the implementation crew."""
    if verdict.decision == "revised" and verdict.revised_proposal:
        return (
            f"APPROVED PROPOSAL (revised by review panel):\n"
            f"{verdict.revised_proposal}\n\n"
            f"ORIGINAL PROPOSAL: {hypothesis.proposal}\n"
            f"PANEL FEEDBACK: {verdict.feedback}"
        )
    return (
        f"APPROVED PROPOSAL:\n"
        f"{hypothesis.proposal}\n\n"
        f"REASONING: {hypothesis.reasoning}\n"
        f"CHANGE: {hypothesis.change_description}\n"
        f"PANEL FEEDBACK: {verdict.feedback}"
    )


def run_review_panel(
    hypothesis: HypothesisOutput,
    experiment_context: str,
    baseline_score: float,
    metric: str,
    direction: str,
) -> ReviewVerdict:
    """Run the AutoGen expert review panel on a hypothesis.

    Five specialists debate the proposal in a round-robin GroupChat,
    then a Moderator issues a structured go/no-go verdict.
    """
    from autogen import ConversableAgent, GroupChat, GroupChatManager

    llm_config = _build_llm_config()

    # On free tier, throttle to stay under 20 RPM (shared with CrewAI)
    throttle = is_free_tier()
    if throttle:
        logger.info(
            f"Free tier — cooldown {_FREE_TIER_COOLDOWN}s, "
            f"then {_FREE_TIER_DELAY}s between panel calls"
        )
        time.sleep(_FREE_TIER_COOLDOWN)

    def _throttle_hook(agent: ConversableAgent, messages: list | None = None) -> None:
        """Log previous speaker's message and throttle on free tier."""
        # Log the latest message from the previous speaker (real-time debate logging)
        if messages:
            last = messages[-1]
            speaker = last.get("name", last.get("role", "unknown"))
            content = last.get("content", "")
            # Truncate for loguru (full content in post-debate log)
            preview = content[:300] + "..." if len(content) > 300 else content
            box = log_box(f"🔬 Review Panel — {speaker}", preview.split("\n"))
            print(box)
            logger.info(f"[REVIEW] [{speaker}] {content}")

        if throttle:
            logger.debug(f"[{agent.name}] Throttling {_FREE_TIER_DELAY}s (free tier)")
            time.sleep(_FREE_TIER_DELAY)

    # Create all agents dynamically from YAML config (review_agents.yaml)
    def _make_agent(key: str) -> ConversableAgent:
        cfg = get_agent_config(key)
        return ConversableAgent(
            name=cfg["name"],
            system_message=cfg["system_message"],
            llm_config=llm_config,
            human_input_mode="NEVER",
            update_agent_state_before_reply=[_throttle_hook],
        )

    agent_keys = get_agent_keys_ordered()
    agents = [_make_agent(key) for key in agent_keys]

    group_chat = GroupChat(
        agents=agents,
        messages=[],
        max_round=get_max_rounds(),
        speaker_selection_method="round_robin",
        send_introductions=True,
    )
    manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    # Build the debate prompt
    prompt = _format_review_prompt(
        hypothesis, experiment_context, baseline_score, metric, direction
    )

    agents_list = ", ".join(str(a.name) for a in group_chat.agents)
    print(log_box("🧪 Review Panel Started", [
        "AutoGen Expert Review Panel",
        f"Agents: {agents_list}",
        f"Max Rounds: {get_max_rounds()}",
        f"Proposal: {hypothesis.proposal[:80]}...",
    ], emoji="🧪"))
    logger.info("Starting AutoGen review panel debate...")

    debate_error = None
    try:
        # Initiate chat from first panelist (first speaker in round-robin)
        agents[0].initiate_chat(manager, message=prompt)
    except Exception as e:
        error_msg = str(e).lower()

        rate_limit_markers = ["429", "rate limit", "quota", "resource_exhausted"]
        is_rate_limit = any(marker in error_msg for marker in rate_limit_markers)

        if not is_rate_limit:
            logger.error(f"Review panel error: {e}")
            return ReviewVerdict(
                decision="approved",
                feedback=f"Review panel failed ({e}), defaulting to approved",
                confidence=0.2,
                concerns=["Review panel encountered an error"],
            )

        # Rate limit hit — but Moderator may have already spoken in an earlier round.
        # Try to salvage a verdict from partial debate before giving up.
        logger.warning(f"Review panel rate limited: {e}")
        debate_error = e

    # Log all debate messages to loguru (AG2 only prints to stdout)
    chat_messages = group_chat.messages
    logger.info(f"Review panel debate: {len(chat_messages)} messages")
    for msg in chat_messages:
        speaker = msg.get("name", msg.get("role", "unknown"))
        content = msg.get("content", "")
        logger.info(f"[REVIEW] [{speaker}] {content}")

    # Extract Moderator's verdict from messages
    moderator_text = ""
    for msg in reversed(chat_messages):
        if msg.get("name") == "Moderator" or msg.get("role") == "assistant":
            moderator_text = msg.get("content", "")
            if "decision" in moderator_text.lower():
                break

    # If rate-limited AND no moderator verdict found, re-raise for main.py retry
    if debate_error and not moderator_text:
        logger.error("No moderator verdict to salvage — re-raising rate limit error")
        raise debate_error

    if debate_error and moderator_text:
        logger.info("Salvaged moderator verdict from partial debate despite rate limit")

    if not moderator_text:
        logger.warning("No moderator response found, defaulting to approved")
        return ReviewVerdict(
            decision="approved",
            feedback="Moderator did not respond, defaulting to approved",
            confidence=0.2,
            concerns=["Moderator response missing"],
        )

    # Parse verdict: try JSON first, then fallback to text
    verdict = _parse_verdict_from_json(moderator_text)
    if verdict is None:
        logger.warning("Could not parse JSON verdict, using text fallback")
        verdict = _parse_verdict_from_text(moderator_text)

    verdict_emoji = {"approved": "✅", "revised": "🔄", "rejected": "❌"}.get(verdict.decision, "❓")
    print(log_box(f"{verdict_emoji} Review Panel Verdict", [
        f"Decision: {verdict.decision.upper()}",
        f"Confidence: {verdict.confidence:.2f}",
        f"Feedback: {verdict.feedback[:200]}",
        f"Concerns: {', '.join(verdict.concerns) if verdict.concerns else 'None'}",
    ]))
    logger.info(
        f"Review panel verdict: {verdict.decision} "
        f"(confidence: {verdict.confidence:.2f})"
    )
    return verdict
