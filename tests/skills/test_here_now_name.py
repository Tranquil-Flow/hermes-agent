"""Regression test for #53382: here-now skill frontmatter name must be valid.

The optional ``here-now`` skill declared ``name: here.now`` in its frontmatter.
The skill-name validator (``UrlSource._VALID_NAME_RE``) only allows lowercase
letters, digits, hyphens and underscores (``^[a-z][a-z0-9_-]*$``), so the dot
made the skill fail to load/validate. This guards against regressions by
validating the real frontmatter through the real project validator.
"""
from pathlib import Path

from tools.skills_hub import UrlSource

SKILL_MD = (
    Path(__file__).resolve().parent.parent.parent
    / "optional-skills"
    / "productivity"
    / "here-now"
    / "SKILL.md"
)


def _parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter between ``---`` markers."""
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    import yaml

    return yaml.safe_load(parts[1]) or {}


def test_here_now_skill_name_is_valid():
    """The here-now skill frontmatter name must pass the project validator."""
    assert SKILL_MD.exists(), f"SKILL.md not found at {SKILL_MD}"

    frontmatter = _parse_frontmatter(SKILL_MD.read_text(encoding="utf-8"))
    name = frontmatter.get("name", "")
    assert name, "Skill name is missing from frontmatter"

    assert UrlSource._is_valid_skill_name(name), (
        f"Skill name {name!r} is invalid per UrlSource._is_valid_skill_name "
        f"(must match {UrlSource._VALID_NAME_RE.pattern} — no dots)"
    )
