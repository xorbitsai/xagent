"""Overlay hook to hide app-managed service users from admin surfaces.

Standalone xagent has no hidden users. The SaaS overlay installs a provider
that returns its synthetic team storage-principal user ids — accounts that own
knowledge-base storage but are not real, login-capable users and therefore must
not appear in, or be deletable from, the global admin user directory.
"""

from __future__ import annotations

from typing import Any, Callable

HiddenUserFilter = Callable[[Any], list[int]]

_hidden_user_filter: HiddenUserFilter | None = None


def set_hidden_user_filter(fn: HiddenUserFilter | None) -> None:
    """Install (or clear, with ``None``) the hidden-user id provider."""
    global _hidden_user_filter
    _hidden_user_filter = fn


def hidden_user_ids(db: Any) -> list[int]:
    """Return user ids to exclude from admin enumeration/deletion (empty if unset)."""
    if _hidden_user_filter is None:
        return []
    return [int(uid) for uid in _hidden_user_filter(db)]
