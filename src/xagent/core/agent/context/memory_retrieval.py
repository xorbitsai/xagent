"""Consistency helpers for cached automatic memory retrieval.

The scope fingerprint is an opaque token supplied by a trusted caller.  This
module does not construct or interpret it; it only uses equality to decide
whether retrieval output stored in an execution checkpoint is still valid.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Protocol


class _ExecutionContextMetadata(Protocol):
    metadata: dict[str, Any]


MEMORY_CONTEXT_METADATA_KEY = "retrieved_memory_context"
RETRIEVED_MEMORIES_METADATA_KEY = "retrieved_memories"
MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY = "memory_retrieval_scope_fingerprint"

_AUTOMATIC_RETRIEVAL_METADATA_KEYS = (
    RETRIEVED_MEMORIES_METADATA_KEY,
    MEMORY_CONTEXT_METADATA_KEY,
)


def reconcile_memory_retrieval_metadata(
    metadata: MutableMapping[str, Any],
    *,
    trusted_scope_fingerprint: str | None,
) -> bool:
    """Make one metadata mapping consistent with the current trusted scope.

    ``None`` means automatic memory retrieval is denied.  A non-empty string
    is an opaque cache-consistency token.  Cached retrieval is retained only
    when the stored token exactly matches the current token.  Legacy metadata
    with no token is therefore invalidated instead of being attributed to the
    current scope.

    Returns whether the mapping changed.
    """
    if trusted_scope_fingerprint == "":
        raise ValueError("trusted_scope_fingerprint must be non-empty or None")

    stored_fingerprint = metadata.get(MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY)
    scope_matches = (
        trusted_scope_fingerprint is not None
        and stored_fingerprint == trusted_scope_fingerprint
    )
    changed = False

    if not scope_matches:
        for key in _AUTOMATIC_RETRIEVAL_METADATA_KEYS:
            if key in metadata:
                metadata.pop(key)
                changed = True

    if trusted_scope_fingerprint is None:
        if MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY in metadata:
            metadata.pop(MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY)
            changed = True
    elif stored_fingerprint != trusted_scope_fingerprint:
        metadata[MEMORY_RETRIEVAL_SCOPE_FINGERPRINT_METADATA_KEY] = (
            trusted_scope_fingerprint
        )
        changed = True

    return changed


def reconcile_execution_memory_retrieval_cache(
    root_context: _ExecutionContextMetadata,
    active_step_contexts: Mapping[str, dict[str, Any]],
    *,
    trusted_scope_fingerprint: str | None,
) -> int:
    """Reconcile automatic retrieval on a root context and serialized DAG children.

    The function intentionally knows only the checkpoint nodes that are real
    execution contexts.  It does not recursively inspect arbitrary dictionaries.
    New DAG children created after this call inherit only the reconciled root
    metadata through ``ExecutionContext.create_child_context``.

    Returns the number of root/child metadata mappings changed.
    """
    changed_count = 0
    root_metadata = getattr(root_context, "metadata", None)
    if not isinstance(root_metadata, MutableMapping):
        raise TypeError("root_context.metadata must be a mutable mapping")
    if reconcile_memory_retrieval_metadata(
        root_metadata,
        trusted_scope_fingerprint=trusted_scope_fingerprint,
    ):
        changed_count += 1

    for child_payload in active_step_contexts.values():
        if not isinstance(child_payload, MutableMapping):
            continue
        child_metadata = child_payload.get("metadata")
        if not isinstance(child_metadata, MutableMapping):
            continue
        if reconcile_memory_retrieval_metadata(
            child_metadata,
            trusted_scope_fingerprint=trusted_scope_fingerprint,
        ):
            changed_count += 1

    return changed_count
