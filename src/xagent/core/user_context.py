"""Shared user-scoped execution context."""

from __future__ import annotations

import contextvars
from typing import Optional

current_user_id: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "current_user_id", default=None
)
