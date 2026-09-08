"""Host extension point for transactional first-administrator setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from fastapi import FastAPI
from sqlalchemy.orm import Session

_HOOK_STATE_KEY = "_first_admin_setup_hook"


@dataclass(frozen=True)
class FirstAdminIdentity:
    """Stable identity exposed while the administrator row is still staged."""

    user_id: int


class FirstAdminSetupHook(Protocol):
    """Stage host-owned rows without ending the caller's transaction."""

    def __call__(self, db: Session, admin: FirstAdminIdentity) -> None: ...


def register_first_admin_setup_hook(app: FastAPI, hook: FirstAdminSetupHook) -> None:
    """Install this application's hook for first-administrator setup.

    The hook runs after the core user and setup setting are staged and before
    the route's sole commit. It may add or flush rows on ``db``, but must not
    commit, roll back, or close the caller-owned session. Raising aborts and
    rolls back the complete setup transaction.
    """
    setattr(app.state, _HOOK_STATE_KEY, hook)


def run_first_admin_setup_hook(
    app: FastAPI, db: Session, admin: FirstAdminIdentity
) -> None:
    """Run the hook registered on ``app``, if any."""
    hook = getattr(app.state, _HOOK_STATE_KEY, None)
    if hook is not None:
        hook(db, admin)


__all__ = [
    "FirstAdminIdentity",
    "FirstAdminSetupHook",
    "register_first_admin_setup_hook",
]
