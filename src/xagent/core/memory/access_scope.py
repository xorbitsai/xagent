"""Trusted memory identity that is independent of execution identity.

This module deliberately lives in ``core.memory``.  A fixed memory scope is a
storage capability, not a web-service policy decision: higher layers may
decide whether to grant it, while memory-store wrappers own how it is bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..execution_scope import ExecutionScope
from .base import MemoryStore
from .scope_columns import strict_user_id


@dataclass(frozen=True)
class MemoryAccessScope:
    """Immutable owner and complete dimensions for one memory namespace.

    ``owner_user_id`` is a real platform ``users.id``.  It therefore uses the
    persisted memory owner's positive signed-int64 domain; booleans and lossy
    numeric spellings are not accepted.  At least one dimension is required so
    opting in can never widen access to every memory owned by that user.
    """

    owner_user_id: int
    memory_dimensions: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.owner_user_id) is not int:
            raise TypeError("memory owner_user_id must be an integer")
        owner_user_id = strict_user_id(self.owner_user_id)
        if owner_user_id is None or owner_user_id <= 0:
            raise ValueError("memory owner_user_id must be a positive signed int64")

        if not isinstance(self.memory_dimensions, Mapping):
            raise TypeError("memory_dimensions must be a mapping")
        dimensions = dict(self.memory_dimensions)
        if not dimensions:
            raise ValueError("memory access scope requires memory_dimensions")
        validated = ExecutionScope(memory_dimensions=dimensions).memory_dimensions

        object.__setattr__(self, "owner_user_id", owner_user_id)
        object.__setattr__(
            self,
            "memory_dimensions",
            MappingProxyType(dict(validated)),
        )


def bind_memory_access_scope(
    store: MemoryStore, access_scope: MemoryAccessScope
) -> MemoryStore:
    """Return ``store`` rebound to ``access_scope``, or fail closed.

    Supported wrappers expose ``with_memory_access_scope`` themselves.  The
    helper does not inspect wrapper internals, and it rejects both unsupported
    compositions and implementations that return themselves: rebinding must
    produce a fresh capability so an earlier namespace cannot remain active.
    """

    if not isinstance(access_scope, MemoryAccessScope):
        raise TypeError("access_scope must be a MemoryAccessScope")
    binder = getattr(store, "with_memory_access_scope", None)
    if not callable(binder):
        raise TypeError("memory store does not support fixed access scopes")
    bound = binder(access_scope)
    if bound is store:
        raise TypeError("memory store did not create a rebound access scope")
    if not isinstance(bound, MemoryStore):
        raise TypeError("memory store returned an invalid rebound access scope")
    return bound


__all__ = ["MemoryAccessScope", "bind_memory_access_scope"]
