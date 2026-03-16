"""
Configuration for the AutoGen expert review panel.

Agent definitions (names + system prompts) live in review_agents.yaml,
matching the CrewAI pattern of agents.yaml / tasks.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import yaml

REVIEW_TEMPERATURE: Final[float] = 0.5

_CONFIG_DIR: Final[Path] = Path(__file__).parent
_AGENTS_YAML: Final[Path] = _CONFIG_DIR / "review_agents.yaml"


def _load_agents_config() -> dict[str, Any]:
    """Load review agent definitions from YAML."""
    return yaml.safe_load(_AGENTS_YAML.read_text(encoding="utf-8"))


def get_agent_config(agent_key: str) -> dict[str, str]:
    """Get name and system_message for a review agent by key."""
    config = _load_agents_config()
    if agent_key not in config:
        raise KeyError(f"Unknown review agent: {agent_key}")
    return config[agent_key]


def get_agent_keys_ordered() -> list[str]:
    """Get all agent keys in order: panelists first, then moderator(s).

    Order is determined by the YAML file order and the 'role' field.
    Adding a new agent to review_agents.yaml automatically includes it.
    """
    config = _load_agents_config()
    panelists = [k for k, v in config.items() if v.get("role") != "moderator"]
    moderators = [k for k, v in config.items() if v.get("role") == "moderator"]
    return panelists + moderators


def get_max_rounds() -> int:
    """Calculate max rounds dynamically from the number of agents in YAML."""
    return len(get_agent_keys_ordered()) + 1  # +1 for the initial prompt
