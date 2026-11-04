"""The agent's current goal — the unit of work in progress.

A turn establishes the user's request as the active goal; a finer unit of work
(e.g. a single DAG step) can override it with its own objective for the duration
of that unit. Nests naturally via context-var tokens.

This is a general execution signal, not a routing concept per se — it just
happens that the ``auto`` model router is its first consumer: it judges
difficulty from what the agent is actually trying to do, rather than from the
scaffolded sub-prompt a given LLM call happens to carry.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator, Optional

_GOAL: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "xagent_active_goal", default=None
)


def current_goal() -> Optional[str]:
    """The goal in effect right now, if any (innermost scope wins)."""
    return _GOAL.get()


def enter_goal(text: Optional[str]) -> Optional[contextvars.Token]:
    """Set the active goal; returns a token for exit_goal(), or None if blank.

    A blank/whitespace goal is a no-op (leaves the current goal unchanged).
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    return _GOAL.set(cleaned)


def exit_goal(token: Optional[contextvars.Token]) -> None:
    """Restore the goal that was in effect before the matching enter_goal()."""
    if token is not None:
        _GOAL.reset(token)


@contextmanager
def goal_scope(text: Optional[str]) -> Iterator[None]:
    """Scope the active goal to ``text`` for the enclosed block."""
    token = enter_goal(text)
    try:
        yield
    finally:
        exit_goal(token)
