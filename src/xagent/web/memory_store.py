"""User-isolated memory store access for the web application.

Importing this module must not build or publish a store. A store may only be
published once storage admission has certified it, so the names below resolve
through the lifecycle manager at attribute-access time (PEP 562) instead of at
import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .dynamic_memory_store import get_memory_store

if TYPE_CHECKING:
    from .dynamic_memory_store import MemoryStoreType

    # Declared for type checkers only; resolved at runtime by ``__getattr__``.
    base_memory_store: MemoryStoreType
    global_memory_store: MemoryStoreType

__all__ = ["base_memory_store", "global_memory_store"]

_LAZY_NAMES = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Resolve the published store on access, failing closed when there is none.

    Raises:
        MemoryUnavailableError: when admission has not certified a store.
    """
    if name in _LAZY_NAMES:
        return get_memory_store()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
