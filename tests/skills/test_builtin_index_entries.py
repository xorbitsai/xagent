"""Built-in skill routing fields must survive the index entry cap.

The skill index is the only thing the model sees when deciding whether to load
a skill, and ``_index_text`` truncates silently, so anything over the cap never
reaches the routing decision.
"""

from pathlib import Path

import pytest

from xagent.core.agent.context.skill_tool import INDEX_ENTRY_MAX_CHARS, _index_text
from xagent.skills.parser import SkillParser

BUILTIN_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "xagent" / "skills" / "builtin"
)
ROUTING_FIELDS = ("description", "when_to_use")


def _builtin_skills() -> list[Path]:
    skills = sorted(p for p in BUILTIN_DIR.iterdir() if (p / "SKILL.md").is_file())
    assert skills, f"no built-in skills found under {BUILTIN_DIR}"
    return skills


@pytest.mark.parametrize("skill_dir", _builtin_skills(), ids=lambda p: p.name)
@pytest.mark.parametrize("field", ROUTING_FIELDS)
def test_routing_field_is_present_and_untruncated(skill_dir: Path, field: str) -> None:
    value = SkillParser.parse(skill_dir).get(field)
    collapsed = " ".join(str(value or "").split())

    assert collapsed, (
        f"{skill_dir.name}: {field} is empty, so the index entry says nothing"
    )

    entry = _index_text(value)
    assert entry == collapsed, (
        f"{skill_dir.name}: {field} is {len(collapsed)} chars, "
        f"{len(collapsed) - INDEX_ENTRY_MAX_CHARS} over the "
        f"{INDEX_ENTRY_MAX_CHARS}-char cap; the index entry keeps only "
        f"{len(entry) - 1} of them and drops the rest"
    )


def test_frontmatter_wins_over_body_section() -> None:
    """A body section is unbounded prose; the bounded frontmatter field routes."""
    skill = SkillParser.parse_bundle(
        name="fixture",
        files={
            "SKILL.md": (
                "---\n"
                "description: from frontmatter\n"
                "when_to_use: also from frontmatter\n"
                "---\n\n"
                "## Description\n\nfrom section\n\n"
                "## When to Use\n\nalso from section\n"
            ).encode()
        },
    )

    assert skill["description"] == "from frontmatter"
    assert skill["when_to_use"] == "also from frontmatter"


def test_whitespace_only_frontmatter_falls_through_to_the_section() -> None:
    """Otherwise it wins and then collapses to an empty index entry."""
    skill = SkillParser.parse_bundle(
        name="fixture",
        files={
            "SKILL.md": (
                "---\n"
                'description: "   "\n'
                "when_to_use: |\n"
                "  \n"
                "---\n\n"
                "## Description\n\nfrom section\n\n"
                "## When to Use\n\nalso from section\n"
            ).encode()
        },
    )

    assert skill["description"] == "from section"
    assert skill["when_to_use"] == "also from section"
