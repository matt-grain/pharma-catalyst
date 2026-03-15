"""Scientific skill loading tool."""

import re
from pathlib import Path
from typing import ClassVar

from crewai.tools import BaseTool

# Skills directory relative to project root
SKILLS_DIR = Path(__file__).parent.parent.parent.parent / ".claude" / "skills"


class SkillLoaderTool(BaseTool):
    """Tool to load scientific skills for context."""

    name: str = "load_skill"
    description: str = (
        "Loads a scientific skill to get domain knowledge and code examples. "
        "Available skills: rdkit, deepchem, datamol, molfeat, pytdc, "
        "chembl-database, pubchem-database, literature-review. "
        "Input: skill name (e.g., 'rdkit' or 'molfeat'). "
        "Returns the skill content with best practices and code examples."
    )

    # Constraints
    max_skills_per_run: int = 3
    _skills_loaded: ClassVar[list[str]] = []  # Shared across instances

    @classmethod
    def reset_counters(cls) -> None:
        cls._skills_loaded = []

    def _run(self, skill_name: str) -> str:
        """Load a skill by name."""
        skill_name = skill_name.strip().lower()

        # Sanitize: only allow alphanumeric and hyphens (prevent path traversal)
        if not re.match(r"^[a-z0-9-]+$", skill_name):
            return f"Error: Invalid skill name '{skill_name}'. Use only lowercase letters, numbers, and hyphens."

        # Check limit
        if len(self._skills_loaded) >= self.max_skills_per_run:
            return f"Error: Max skills limit ({self.max_skills_per_run}) reached. Already loaded: {self._skills_loaded}"

        # Check if already loaded
        if skill_name in self._skills_loaded:
            return f"Skill '{skill_name}' already loaded this session."

        # Try scientific skills first, then root skills
        skill_paths = [
            SKILLS_DIR / "scientific" / f"{skill_name}.md",
            SKILLS_DIR / f"{skill_name}.md",
        ]

        for skill_path in skill_paths:
            if skill_path.exists():
                try:
                    content = skill_path.read_text(encoding="utf-8")
                    self._skills_loaded.append(skill_name)
                    # Truncate if too long (keep first 8000 chars)
                    if len(content) > 8000:
                        content = (
                            content[:8000] + "\n\n[... truncated, skill too long ...]"
                        )
                    return f"=== SKILL: {skill_name} ===\n\n{content}"
                except Exception as e:
                    return f"Error loading skill: {e}"

        # List available skills
        available = []
        if (SKILLS_DIR / "scientific").exists():
            available = [f.stem for f in (SKILLS_DIR / "scientific").glob("*.md")]
        return f"Skill '{skill_name}' not found. Available: {available}"
