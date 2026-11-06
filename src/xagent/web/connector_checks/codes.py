"""The value shapes every smoke V2 module shares.

A result carries only codes, ids and tool names, so every value that reaches
it is held to one of these patterns, and ``result.schema.json`` repeats them
(a test pins the two equal). Declarations are held to the same patterns, so a
declared check id or fixture key always fits the result it ends up in.
"""

from __future__ import annotations

import hashlib
import re

# A reason, a skip reason, a check id or a fixture key: a public-safe code.
CODE_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,79}")
# A catalog app id as the builtin catalog writes it, and a connector's label
# in a result.
APP_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
# An MCP wire name: the in-repo servers use snake_case; remote servers may
# use hyphens, dots or slashes.
TOOL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_CHECK_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")

# The longest app id a result shows as itself; ``<label>.<check name>`` must
# still fit a code.
_MAX_LABEL_LENGTH = 64
# Every generated label starts with this, and no pass-through label does.
_GENERATED_LABEL_PREFIX = "custom-"


def is_code(value: object) -> bool:
    return isinstance(value, str) and CODE_PATTERN.fullmatch(value) is not None


def is_check_id(connector: str, check_id: str) -> bool:
    """Whether ``check_id`` is ``<connector>.<snake_case_name>`` and a code."""
    prefix = f"{connector}."
    return (
        check_id.startswith(prefix)
        and _CHECK_NAME_PATTERN.fullmatch(check_id[len(prefix) :]) is not None
        and is_code(check_id)
    )


def connector_label(app_id: str) -> str:
    """How a result names the connector of catalog app ``app_id``.

    A builtin app id is its own label. An admin-created app's id is free
    text (any string, up to the column's 100 characters), so one that is not
    a short kebab-case id, or that starts with ``custom-``, is named
    ``custom-<16 hex digits>`` from its SHA-256: stable across runs, so the
    result diff still matches it, and never the text itself. Generated labels
    always start with ``custom-`` and pass-through labels never do, so the two
    cannot collide; only a 64-bit SHA-256 prefix collision between two hashed
    ids remains.
    """
    if (
        len(app_id) <= _MAX_LABEL_LENGTH
        and APP_ID_PATTERN.fullmatch(app_id)
        and not app_id.startswith(_GENERATED_LABEL_PREFIX)
    ):
        return app_id
    digest = hashlib.sha256(app_id.encode("utf-8")).hexdigest()[:16]
    return f"{_GENERATED_LABEL_PREFIX}{digest}"
