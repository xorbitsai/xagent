"""Fixture keys: the per-environment, per-identity values a check reads.

A check that needs a vendor object (a document id, a workspace id, a seeded
row) names it by a fixture key instead of a literal. The identity adapter the
engine runs under resolves each key from server-side configuration, so a
value differs per environment and identity and never comes from the caller.

A key is ``<namespace>.<name>``, at most 80 characters: the namespace is
kebab-case, by convention the app id of the connector or tenant family that
owns the value (``google-docs``), and the name is one or more dot-separated
snake_case parts (``document_id``, ``seed.project_key``). Errors name the
key, never a value.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .codes import is_code

FIXTURE_KEY_PATTERN = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:_[a-z0-9]+)*)+"
)

# What a resolver may return for a key: a JSON scalar.
FixtureValue = str | int | float | bool


@dataclass(frozen=True)
class FixtureRef:
    """A value the identity's resolver supplies at run time."""

    key: str

    def __post_init__(self) -> None:
        # A ``fixture_missing`` row reports the key as its reason, a code of
        # at most 80 characters.
        if not FIXTURE_KEY_PATTERN.fullmatch(self.key) or not is_code(self.key):
            raise ValueError(f"invalid fixture key: {self.key!r}")


class FixtureResolver(Protocol):
    def resolve_fixture(self, key: str) -> FixtureValue | None:
        """The value for ``key`` in this environment and identity, or None."""


class FixtureMissing(Exception):
    """The resolver has no value for ``key``; the check reports
    ``fixture_missing``."""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def resolve_fixtures(value: Any, resolver: FixtureResolver) -> Any:
    """``value`` with every ``FixtureRef`` replaced by the resolver's value.

    Mappings and sequences are walked; a tuple comes back as a list, as JSON
    has no tuple. Raises ``FixtureMissing`` for the first unresolved key and
    ``TypeError`` for a value that is not a JSON scalar.
    """
    if isinstance(value, FixtureRef):
        resolved = resolver.resolve_fixture(value.key)
        if resolved is None:
            raise FixtureMissing(value.key)
        if not isinstance(resolved, (str, int, float, bool)) or (
            isinstance(resolved, float) and not math.isfinite(resolved)
        ):
            raise TypeError(f"fixture {value.key} did not resolve to a JSON scalar")
        return resolved
    if isinstance(value, Mapping):
        return {key: resolve_fixtures(item, resolver) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [resolve_fixtures(item, resolver) for item in value]
    return value
