"""Optional host-policy fence for actor OAuth callback persistence."""

from collections.abc import Callable

from sqlalchemy.orm import Session

OAuthPersistenceGuard = Callable[[Session, int, str], None]
_guard: OAuthPersistenceGuard | None = None


def set_oauth_persistence_guard(guard: OAuthPersistenceGuard | None) -> None:
    """Install the host guard at startup; None preserves standalone behavior."""
    global _guard
    _guard = guard


def require_oauth_owner_active(db: Session, user_id: int, owner_key: str) -> None:
    """Fence actor persistence after association locks, before flow/grant locks.

    The guard must raise ValueError to deny persistence. It may acquire locks,
    but must not commit, roll back, or perform network I/O. The caller retains
    those locks through credential commit and rolls back on failure.
    """
    if _guard is not None:
        _guard(db, user_id, owner_key)
