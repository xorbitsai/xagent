"""Structural exactly-one-entity invariant on ``ShareChatAccessContext`` (#1225).

The context documents that exactly one of ``agent`` / ``workforce`` is set.
That guarantee is load-bearing for run-quota attribution: the entity
rate-limit key prefers workforce when both ids are present, so a both-set
context would silently charge an agent-share run to a workforce bucket.
These tests pin the ``__post_init__`` guard that makes the illegal states
unconstructible instead of merely narrated in docstrings.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from xagent.web.api.public_chat_access import ShareChatAccessContext


def _context_kwargs(**entity: Any) -> dict[str, Any]:
    return {
        "user": SimpleNamespace(id=1),
        "share_token": "tok",
        "guest_id": "guest",
        **entity,
    }


def test_agent_only_context_constructs() -> None:
    agent = SimpleNamespace(id=7)
    context = ShareChatAccessContext(**_context_kwargs(agent=agent))
    assert context.agent is agent
    assert context.workforce is None


def test_workforce_only_context_constructs() -> None:
    workforce = SimpleNamespace(id=7)
    context = ShareChatAccessContext(**_context_kwargs(workforce=workforce))
    assert context.workforce is workforce
    assert context.agent is None


def test_neither_entity_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ShareChatAccessContext(**_context_kwargs())


def test_both_entities_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ShareChatAccessContext(
            **_context_kwargs(
                agent=SimpleNamespace(id=7),
                workforce=SimpleNamespace(id=9),
            )
        )


class _FalsyEntity:
    """Truthy-ness must not matter: the guard must compare against ``None``.

    Tests elsewhere construct the context with ``MagicMock()`` / ``object()``
    stand-ins; a stand-in whose ``__bool__`` is falsy is still a *set* entity.
    """

    def __bool__(self) -> bool:
        return False


def test_falsy_entity_object_counts_as_set() -> None:
    agent = _FalsyEntity()
    context = ShareChatAccessContext(**_context_kwargs(agent=agent))
    assert context.agent is agent
    assert context.workforce is None
