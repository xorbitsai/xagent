"""Application-owned access seam for personal management API keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PersonalKeyAccessScope:
    """The personal-key owners an authenticated actor may manage."""

    owner_user_ids: tuple[int, ...]
    can_manage_others: bool


_personal_key_scope_hook: Any = None


def set_personal_key_scope_hook(hook: Any) -> None:
    """Install or clear the application-owned personal-key scope resolver."""
    global _personal_key_scope_hook
    _personal_key_scope_hook = hook


def get_personal_key_access_scope(db: Any, actor: Any) -> PersonalKeyAccessScope:
    """Resolve a fail-closed scope, always retaining the actor's own keys."""
    actor_id = int(actor.id)
    self_scope = PersonalKeyAccessScope((actor_id,), False)
    if _personal_key_scope_hook is None:
        return self_scope

    try:
        proposed = _personal_key_scope_hook(db, actor)
        if not isinstance(proposed, PersonalKeyAccessScope):
            return self_scope
        if type(proposed.can_manage_others) is not bool:
            return self_scope
        if type(proposed.owner_user_ids) is not tuple:
            return self_scope
        if any(
            type(owner_id) is not int or owner_id <= 0
            for owner_id in proposed.owner_user_ids
        ):
            return self_scope
        if not proposed.can_manage_others:
            return self_scope
        owner_ids = tuple(dict.fromkeys((actor_id, *proposed.owner_user_ids)))
        return PersonalKeyAccessScope(owner_ids, True)
    except (AttributeError, TypeError, ValueError):
        return self_scope
