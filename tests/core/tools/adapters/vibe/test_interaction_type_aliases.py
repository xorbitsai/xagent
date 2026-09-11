"""``INTERACTION_TYPE_ALIASES`` is a hand-copy of a frontend table.

``normalizeInteractions`` (``frontend/src/contexts/app-context-chat.tsx``) maps
off-contract type names onto the seven the render surface implements, and the
engine applies the same map before it decides whether an interaction is
answerable. The two tables have to agree: an alias only the frontend knows
reads as an unsupported type engine-side, and one only the engine knows gets
published and then dropped on render. Nothing but this cell notices when one
side is edited alone.

The parse is deliberately literal. If the ternary chain is restructured this
fails with "could not parse", which is the right outcome -- the map has to be
re-read by a human either way.
"""

import re
from pathlib import Path

import pytest

from xagent.core.tools.adapters.vibe.interaction_types import (
    INTERACTION_TYPE_ALIASES,
    INTERACTION_TYPES,
)

SOURCE = (
    Path(__file__).resolve().parents[5]
    / "frontend"
    / "src"
    / "contexts"
    / "app-context-chat.tsx"
)

# One arm of the chain: the `rawType === "x" || rawType === "y"` test, then the
# canonical name it resolves to. Matched against whitespace-collapsed source so
# a reformat of the same expression still parses.
_ARM = re.compile(r'((?:rawType === "\w+"(?: \|\| )?)+) \? "(\w+)"')
_ALIAS = re.compile(r'rawType === "(\w+)"')


def _frontend_aliases() -> dict[str, str]:
    if not SOURCE.exists():
        pytest.skip(f"frontend source not present: {SOURCE}")
    flattened = re.sub(r"\s+", " ", SOURCE.read_text(encoding="utf-8"))
    body = flattened.split("const rawType = item.type", 1)
    assert len(body) == 2, "could not parse: normalizeInteractions moved"
    region = body[1].split("const rawField", 1)[0]
    arms = _ARM.findall(region)
    assert arms, "could not parse: the alias ternary chain changed shape"
    return {
        alias: canonical for tests, canonical in arms for alias in _ALIAS.findall(tests)
    }


def test_the_engine_alias_table_matches_the_frontend_one() -> None:
    assert _frontend_aliases() == INTERACTION_TYPE_ALIASES


def test_every_alias_resolves_to_a_supported_type() -> None:
    assert set(INTERACTION_TYPE_ALIASES.values()) <= set(INTERACTION_TYPES)
    assert not set(INTERACTION_TYPE_ALIASES) & set(INTERACTION_TYPES)
