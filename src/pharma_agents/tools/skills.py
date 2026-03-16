"""Scientific skill loading and discovery tools."""

import re
from pathlib import Path
from typing import ClassVar

from crewai.tools import BaseTool

# Skills directory relative to project root
SKILLS_DIR = Path(__file__).parent.parent.parent.parent / ".claude" / "skills"


def _parse_frontmatter(path: Path) -> dict[str, str]:
    """Extract YAML frontmatter (name, description) from a skill file."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fm[key.strip()] = val.strip()
    return fm


def _list_all_skills() -> list[dict[str, str]]:
    """Scan skills directories and return name + description for each."""
    skills = []
    seen = set()

    # Scientific skills (flat .md files)
    sci_dir = SKILLS_DIR / "scientific"
    if sci_dir.exists():
        for f in sorted(sci_dir.glob("*.md")):
            fm = _parse_frontmatter(f)
            name = fm.get("name", f.stem)
            if name not in seen:
                seen.add(name)
                skills.append({"name": name, "description": fm.get("description", "")})

    # Root-level .md skills
    if SKILLS_DIR.exists():
        for f in sorted(SKILLS_DIR.glob("*.md")):
            fm = _parse_frontmatter(f)
            name = fm.get("name", f.stem)
            if name not in seen:
                seen.add(name)
                skills.append({"name": name, "description": fm.get("description", "")})

    # Directory-based skills (tooluniverse-*)
    if SKILLS_DIR.exists():
        for d in sorted(SKILLS_DIR.iterdir()):
            if d.is_dir() and (d / "SKILL.md").exists():
                fm = _parse_frontmatter(d / "SKILL.md")
                name = fm.get("name", d.name)
                if name not in seen:
                    seen.add(name)
                    skills.append(
                        {"name": name, "description": fm.get("description", "")}
                    )

    return skills


class SkillDiscoveryTool(BaseTool):
    """Tool to discover available skills by keyword."""

    name: str = "discover_skills"
    description: str = (
        "Lists available skills with descriptions, optionally filtered by keyword. "
        "Input: keyword to filter (e.g., 'drug', 'compound', 'molecular') or 'all' to list everything. "
        "Returns skill names and descriptions. Use load_skill to load a specific skill."
    )

    def _run(self, keyword: str = "all") -> str:
        keyword = keyword.strip().lower()
        skills = _list_all_skills()

        if keyword and keyword != "all":
            skills = [
                s
                for s in skills
                if keyword in s["name"].lower() or keyword in s["description"].lower()
            ]

        if not skills:
            return f"No skills found matching '{keyword}'."

        lines = [f"Found {len(skills)} skill(s):\n"]
        for s in skills:
            desc = (
                s["description"][:120] + "..."
                if len(s["description"]) > 120
                else s["description"]
            )
            lines.append(f"- **{s['name']}**: {desc}")

        lines.append('\nUse load_skill("<name>") to load any skill above.')
        return "\n".join(lines)


class SkillLoaderTool(BaseTool):
    """Tool to load scientific skills for context."""

    name: str = "load_skill"
    description: str = (
        "Loads a skill by name to get domain knowledge, code examples, or workflow guides. "
        "Use discover_skills first to find available skills and their descriptions. "
        "Input: skill name (e.g., 'rdkit' or 'tooluniverse-drug-research'). "
        "Returns the full skill content."
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
            SKILLS_DIR
            / skill_name
            / "SKILL.md",  # directory-based skills (tooluniverse-*)
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
        # Also list directory-based skills (tooluniverse-*)
        for skill_dir in SKILLS_DIR.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                available.append(skill_dir.name)
        return f"Skill '{skill_name}' not found. Available: {sorted(available)}"
