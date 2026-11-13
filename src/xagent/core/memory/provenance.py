"""Host-supplied provenance for model-stored memories.

A trusted host may describe *where* a memory was learned (for example a
Slack channel versus a DM) so its users can later see, per memory, the place
and time it came from. The host returns that description from its memory
policy resolver (``MemoryPolicyDecision.memory_provenance``); the web layer
binds it to the run's ``store_memory`` tool, which copies it into each new
note's metadata under :data:`MEMORY_PROVENANCE_METADATA_KEY`. The note's own
``timestamp`` records *when*.

Provenance is descriptive only. It is never a namespace dimension: stores do
not filter on it, it is not part of any scope fingerprint, and it cannot widen
or narrow which memories a run may read.

This module deliberately depends only on ``core.execution_scope`` so that
light host-facing policy modules can import it without pulling in the agent
runtime.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias

from ..execution_scope import validate_scope_component

# Note-metadata key under which host-supplied provenance is stored as a
# nested string mapping. Nested (not flat-prefixed) so provenance can never
# collide with ``user_id`` or the ``execution_scope_*`` isolation stamps, and
# is never mistaken for a namespace dimension by store filters.
MEMORY_PROVENANCE_METADATA_KEY = "provenance"

# Provenance values are identifiers (channel ids, kinds, task ids), never
# prose: no whitespace and a bounded length structurally keep conversation
# text out of a note that outlives its conversation (#2753).
_PROVENANCE_VALUE_RE = re.compile(r"\S{1,256}")

MemoryProvenance: TypeAlias = Mapping[str, str]
"""Non-content identifiers describing where a memory was learned, e.g.
``{"channel_kind": "dm", "channel_id": "D123", "task_id": "42"}``."""


def validate_memory_provenance(provenance: MemoryProvenance) -> MemoryProvenance:
    """Validate host provenance and return an immutable copy.

    Keys follow the execution-scope component grammar
    (``[a-zA-Z0-9_-]{1,63}``); values must be whitespace-free strings of at
    most 256 characters. The value grammar is what enforces the "identifiers,
    never conversation text" contract: a sentence cannot pass it. An empty
    mapping is rejected too -- a host with nothing to record passes ``None``.

    Raises:
        TypeError: ``provenance`` is not a mapping, is empty, or a value is
            not a string.
        ValueError: a key or value does not match its grammar
            (``InvalidScopeComponentError`` is a ``ValueError``).
    """
    if not isinstance(provenance, Mapping) or not provenance:
        raise TypeError("memory provenance must be a non-empty mapping")
    validated: dict[str, str] = {}
    for key, value in provenance.items():
        validate_scope_component(key, field_name="memory provenance key")
        if not isinstance(value, str):
            raise TypeError(f"memory provenance value for {key!r} must be a string")
        if not _PROVENANCE_VALUE_RE.fullmatch(value):
            raise ValueError(
                f"memory provenance value for {key!r} must be a whitespace-free "
                "identifier of at most 256 characters"
            )
        validated[key] = value
    return MappingProxyType(validated)


__all__ = [
    "MEMORY_PROVENANCE_METADATA_KEY",
    "MemoryProvenance",
    "validate_memory_provenance",
]
