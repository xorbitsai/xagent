"""Request-scoped user context helpers for KB/RAG operations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator, Optional

_current_user_id: ContextVar[Optional[int]] = ContextVar(
    "rag_current_user_id", default=None
)
_current_is_admin: ContextVar[bool] = ContextVar("rag_current_is_admin", default=False)


@dataclass(frozen=True)
class UserScope:
    """Resolved user scope used by tenant-aware operations."""

    user_id: Optional[int]
    is_admin: bool


def get_user_scope() -> UserScope:
    """Return current request-scoped user scope."""
    return UserScope(
        user_id=_current_user_id.get(), is_admin=bool(_current_is_admin.get())
    )


def resolve_user_scope(
    user_id: Optional[int] = None, is_admin: Optional[bool] = None
) -> UserScope:
    """Resolve explicit scope with context fallback.

    Explicit arguments always take precedence. When no explicit user context is
    provided (both user_id and is_admin are None), fall back to request-scoped
    values set by API/service entrypoints.

    Note:
        is_admin uses Optional[bool] to distinguish three cases:
        - None: fallback to context
        - False: explicitly non-admin user
        - True: explicitly admin user
    """
    if user_id is not None or is_admin is not None:
        return UserScope(
            user_id=user_id, is_admin=bool(is_admin) if is_admin is not None else False
        )

    scoped = get_user_scope()
    return UserScope(user_id=scoped.user_id, is_admin=bool(scoped.is_admin))


@contextmanager
def user_scope_context(user_id: Optional[int], is_admin: bool) -> Iterator[None]:
    """Temporarily set request user scope for nested KB/RAG operations."""
    user_token: Token[Optional[int]] = _current_user_id.set(user_id)
    admin_token: Token[bool] = _current_is_admin.set(bool(is_admin))
    try:
        yield
    finally:
        _current_user_id.reset(user_token)
        _current_is_admin.reset(admin_token)
