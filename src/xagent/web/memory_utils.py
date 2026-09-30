"""Memory utilities for web application."""

from __future__ import annotations

import logging
from typing import Optional, Union

from xagent.core.memory.in_memory import InMemoryMemoryStore
from xagent.core.memory.lancedb import LanceDBMemoryStore

from .dynamic_memory_store import get_memory_store_manager
from .revocable_memory_store import RevocableMemoryStore
from .user_isolated_memory import UserIsolatedMemoryStore

logger = logging.getLogger(__name__)

# Type alias for our memory store types that includes user isolation and the
# revocation wrapper the manager publishes.
MemoryStoreType = Union[
    InMemoryMemoryStore,
    LanceDBMemoryStore,
    UserIsolatedMemoryStore,
    RevocableMemoryStore,
]


def create_memory_store(
    similarity_threshold: Optional[float] = None,
) -> MemoryStoreType:
    """Return the admitted memory store for this process.

    The store is built from the explicit global memory embedding authority
    alone. It is never derived from whichever embedding model happens to be
    configured in the model hub, because an administrator's personal default is
    not an authority over everybody's stored vectors.

    Args:
        similarity_threshold: Threshold for vector search, honoured only when
            this call is what first creates the process-wide manager.

    Raises:
        MemoryUnavailableError: when admission has not certified a store.
    """
    return get_memory_store_manager(similarity_threshold).get_memory_store()
