"""Logging utilities for framed output matching CrewAI's visual style."""

from __future__ import annotations


def log_box(title: str, lines: list[str], width: int = 120, emoji: str = "") -> str:
    """Format a framed log box matching CrewAI's visual style.

    Returns the formatted string (caller decides whether to print or log).
    """
    header = f" {emoji} {title} " if emoji else f" {title} "
    pad = max(0, width - len(header) - 2)
    left_pad = pad // 2
    right_pad = pad - left_pad

    parts = [
        f"╭{'─' * left_pad}{header}{'─' * right_pad}╮",
        f"│{' ' * width}│",
    ]
    for line in lines:
        # Wrap long lines
        while len(line) > width - 4:
            parts.append(f"│  {line[: width - 4]:<{width - 2}}│")
            line = line[width - 4 :]
        parts.append(f"│  {line:<{width - 2}}│")
    parts.extend(
        [
            f"│{' ' * width}│",
            f"╰{'─' * width}╯",
        ]
    )
    return "\n".join(parts)
