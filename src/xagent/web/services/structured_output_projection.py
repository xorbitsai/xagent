"""A pure projection shared by every read path, including a downstream embedder's; it performs no database reads.

The structured result of a run is stored on ``tasks.structured_output_record``
as ``{"run_id", "status", "output", "reason"}``. :func:`build_record` makes that
record from the result of a run when the run is settled as COMPLETED.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Each list in a stored reason holds at most this many paths.
MAX_REASON_FIELDS = 20
_REASON_LIST_KEYS = ("missing_fields", "unexpressible_fields")
# The key of a run result that carries the structured result.
_RESULT_KEY = "structured_output"


def _stored_reason(reason: Any) -> dict[str, Any] | None:
    """The part of ``reason`` that is stored: the code, and each field list
    that is not empty, cut to :data:`MAX_REASON_FIELDS`. No free text is kept."""
    if not isinstance(reason, dict):
        return None
    stored: dict[str, Any] = {"code": reason.get("code")}
    for key in _REASON_LIST_KEYS:
        fields = reason.get(key)
        if isinstance(fields, list) and fields:
            stored[key] = fields[:MAX_REASON_FIELDS]
    return stored


def build_record(run_id: str | None, structured_output: Any) -> dict[str, Any] | None:
    """The record to store for ``run_id``, or None when the run has no
    structured result.

    ``structured_output`` is the value of the run result's ``structured_output``
    key: a mapping with ``status``, ``output`` and ``reason``. A value that is
    neither None nor a mapping is a server-side internal error: it is stored
    as a failed record with code ``internal_error``, and only its type name
    is logged, never the value.
    """
    if structured_output is None:
        return None
    if not isinstance(structured_output, dict):
        logger.warning(
            "The %s of run %s is a %s, not a mapping; storing a failed record",
            _RESULT_KEY,
            run_id,
            type(structured_output).__name__,
        )
        return {
            "run_id": run_id,
            "status": "failed",
            "output": None,
            "reason": {"code": "internal_error"},
        }
    return {
        "run_id": run_id,
        "status": structured_output.get("status"),
        "output": structured_output.get("output"),
        "reason": _stored_reason(structured_output.get("reason")),
    }
