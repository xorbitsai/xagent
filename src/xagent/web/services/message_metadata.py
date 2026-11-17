"""Server-set metadata attached to one user chat message.

A host settles a waiting interaction by appending a message flagged
``{"hidden": true, "kind": ..., "interaction_id": ...}`` (see
``task_interaction_settle``). Clients and server-derived surfaces treat such a
message as plumbing rather than conversation. The text still reaches the model,
still passes input review and still resumes the waiting run -- ``hidden`` only
affects display, titles, search and the task's last input.

Only server code can set this metadata: a client chat frame cannot carry it.
"""

from __future__ import annotations

import re
from typing import Any

_KIND_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_INTERACTION_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


def is_valid_metadata_kind(kind: Any) -> bool:
    """A lowercase identifier of at most 64 characters."""
    return isinstance(kind, str) and _KIND_PATTERN.fullmatch(kind) is not None


def is_valid_interaction_id(interaction_id: Any) -> bool:
    """At most 64 characters of ``[A-Za-z0-9_-]``."""
    return (
        isinstance(interaction_id, str)
        and _INTERACTION_ID_PATTERN.fullmatch(interaction_id) is not None
    )


def validate_server_message_metadata(raw: Any) -> dict[str, Any] | None:
    """Return ``raw`` if it is well-formed server metadata, ``None`` if absent.

    Raises ``ValueError`` for anything else: this value is set by server code,
    so a malformed one is a bug to surface, not input to clean up.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("message metadata must be an object")
    unknown = set(raw) - {"hidden", "kind", "interaction_id"}
    if unknown:
        raise ValueError(f"unsupported message metadata keys: {sorted(unknown)}")
    result: dict[str, Any] = {}
    if "kind" in raw:
        if not is_valid_metadata_kind(raw["kind"]):
            raise ValueError("message metadata kind is not a valid identifier")
        result["kind"] = raw["kind"]
    if "interaction_id" in raw:
        if not is_valid_interaction_id(raw["interaction_id"]):
            raise ValueError("message metadata interaction_id is not valid")
        result["interaction_id"] = raw["interaction_id"]
    if "hidden" in raw:
        if raw["hidden"] is not True or "kind" not in result:
            raise ValueError("hidden must be true and needs a kind")
        result["hidden"] = True
    return result or None


def is_hidden_message_metadata(metadata: Any) -> bool:
    """True only for an explicit ``hidden: true`` flag."""
    return isinstance(metadata, dict) and metadata.get("hidden") is True


def is_hidden_user_event(event_type: Any, data: Any) -> bool:
    """True for a ``user_message`` trace event that carries a hidden message."""
    return (
        event_type == "user_message"
        and isinstance(data, dict)
        and is_hidden_message_metadata(data.get("metadata"))
    )
