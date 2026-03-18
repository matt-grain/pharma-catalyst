"""Tool safety and package configuration.

Loads defaults from src/pharma_agents/tool_defaults.yaml, then merges
per-experiment overrides from experiments/<exp>/tool_config.yaml.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

import yaml

from pharma_agents.memory import get_experiments_dir

_DEFAULTS_PATH: Final = Path(__file__).parent / "tool_defaults.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict if missing or invalid."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def _load_config() -> dict:
    """Load defaults + per-experiment overrides (cached for session).

    WARNING: Returns a mutable dict. Do NOT mutate the return value directly —
    use get_allowed_packages() / get_dangerous_patterns() which return copies.
    """
    config = _load_yaml(_DEFAULTS_PATH)

    # Per-experiment override (merges lists, not replaces)
    try:
        exp_dir = get_experiments_dir()
        exp_config = _load_yaml(exp_dir / "tool_config.yaml")
    except (ValueError, KeyError):
        exp_config = {}

    if "allowed_packages" in exp_config:
        base = set(config.get("allowed_packages", []))
        base.update(exp_config["allowed_packages"])
        config["allowed_packages"] = sorted(base)

    if "dangerous_patterns" in exp_config:
        base = list(config.get("dangerous_patterns", []))
        base.extend(exp_config["dangerous_patterns"])
        config["dangerous_patterns"] = base

    return config


def get_allowed_packages() -> set[str]:
    """Return the set of allowed packages for InstallPackageTool."""
    return set(_load_config().get("allowed_packages", []))


def get_dangerous_patterns() -> tuple[str, ...]:
    """Return dangerous patterns blocked in train.py writes/edits."""
    return tuple(_load_config().get("dangerous_patterns", []))


def reload_config() -> None:
    """Clear cached config (call after changing experiment)."""
    _load_config.cache_clear()
