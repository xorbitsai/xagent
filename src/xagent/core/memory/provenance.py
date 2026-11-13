"""Host-supplied provenance for model-stored memories.

A trusted host may describe *where* a memory was learned (for example a
Slack channel versus a DM) so its users can later see, per memory, the place
and time it came from. The host returns that description from its memory
policy resolver (``MemoryPolicyDecision.memory_provenance``); the web layer
binds it to the run's ``store_memory`` tool, which copies it into each new
note's metadata under :data:`MEMORY_PROVENANCE_METADATA_KEY`. The note's own
``timestamp`` records *when*.

Provenance means "originally learned in": it is written once, when the note
is first stored, and never re-stamped. ``update_memory`` keeps it while
replacing the content, ``UserIsolatedMemoryStore.update`` restores the stored
value over whatever a caller sends, and the REST create endpoint drops a
client-supplied value -- so it is server-owned like ``user_id`` and the
``execution_scope_*`` stamps.

Provenance is descriptive only. It is never a namespace dimension: stores do
not filter on it, it is not part of any scope fingerprint, and it cannot widen
or narrow which memories a run may read.

Imports stay limited to ``core.execution_scope`` (plus the standard
library) so that ``web.services.memory_policy`` does not have to import the
agent runtime merely to validate a decision.
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

# Provenance values must look like ASCII identifiers (Slack ``C…``/``D…``/
# ``G…`` ids, Teams ``19:…@thread.tacv2`` / ``a:1…`` ids and thread ids such
# as ``19:…@thread.tacv2;messageid=…``, Feishu ``oc_…`` ids, numeric task
# ids). This is defense in depth against a host bug that
# passes conversation text: it rejects whitespace, non-ASCII scripts (so
# spaceless CJK sentences fail) and punctuation typical of prose, but it
# cannot prove a value is not content -- the host remains responsible for
# passing identifiers only (#2753).
_PROVENANCE_VALUE_RE = re.compile(r"[A-Za-z0-9._:;@/=+-]{1,256}")

MemoryProvenance: TypeAlias = Mapping[str, str]
"""Non-content identifiers describing where a memory was learned, e.g.
``{"channel_kind": "dm", "channel_id": "D123", "task_id": "42"}``."""


def validate_memory_provenance(provenance: MemoryProvenance) -> MemoryProvenance:
    """Validate host provenance and return an immutable copy.

    Keys follow the execution-scope component grammar
    (``[a-zA-Z0-9_-]{1,63}``); values must match the ASCII identifier grammar
    ``[A-Za-z0-9._:;@/=+-]{1,256}``. The value grammar is defense in depth: it
    catches a host bug that passes free text (whitespace, non-ASCII scripts,
    prose punctuation), but an identifier-shaped string can still be content,
    so hosts must pass identifiers only. An empty mapping is rejected too -- a
    host with nothing to record passes ``None``.

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
                f"memory provenance value for {key!r} must be an ASCII "
                "identifier matching [A-Za-z0-9._:;@/=+-]{1,256}"
            )
        validated[key] = value
    return MappingProxyType(validated)


__all__ = [
    "MEMORY_PROVENANCE_METADATA_KEY",
    "MemoryProvenance",
    "validate_memory_provenance",
]
