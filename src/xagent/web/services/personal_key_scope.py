"""Application-owned access seam for personal management API keys."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


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


_personal_key_principal_policy: Callable[[Any], object] | None = None


def set_personal_key_principal_policy(policy: Callable[[Any], object] | None) -> None:
    """Install or clear the application-owned personal-key principal policy.

    The policy receives the authenticated user snapshot and runs in the
    authentication worker, so it should decide from that snapshot alone.
    """
    global _personal_key_principal_policy
    _personal_key_principal_policy = policy


def check_personal_key_principal(user: Any, *, key_prefix: str) -> bool:
    """Return whether the application policy admits an authenticated user.

    Only a policy result that is literally ``True`` admits the user. Any other
    result, and any ``Exception`` raised by the policy, denies. Both denials
    are logged with the key prefix and user id only. With no policy installed
    every user is admitted.
    """
    policy = _personal_key_principal_policy
    if policy is None:
        return True
    try:
        allowed = policy(user)
    except Exception:
        logger.error(
            "Personal key principal policy raised; denying key_prefix=%s user_id=%s",
            key_prefix,
            int(user.id),
        )
        return False
    if allowed is True:
        return True
    logger.warning(
        "Personal key principal policy denied key_prefix=%s user_id=%s",
        key_prefix,
        int(user.id),
    )
    return False
