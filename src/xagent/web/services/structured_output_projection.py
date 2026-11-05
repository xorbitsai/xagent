"""A pure projection shared by every read path, including a downstream embedder's; it performs no database reads.

The structured result of a run is stored on ``tasks.structured_output_record``
as ``{"run_id", "status", "output", "reason"}``. :func:`build_record` makes that
record from the result of a run when the run is settled as COMPLETED, and
:func:`project_structured_output` turns the stored record into the three
fields a task response carries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..models.task import TaskStatus

logger = logging.getLogger(__name__)

# The key of ``tasks.agent_config`` that holds the format of a task.
_FORMAT_KEY = "response_format"

# The message of each reason code. A response carries the text of its code
# from this table and never text from a record.
REASON_MESSAGES: dict[str, str] = {
    "agent_partial": (
        "The agent completed only part of the task, "
        "so no structured result was extracted."
    ),
    "agent_blocked": (
        "The agent stopped without completing the task, "
        "so no structured result was extracted."
    ),
    "information_missing": (
        "The final answer does not state some required values. "
        "missing_fields lists up to 20 of them."
    ),
    "value_not_expressible": (
        "The final answer states some values that cannot be written in the form "
        "the schema requires. unexpressible_fields lists up to 20 of them."
    ),
    "not_produced": (
        "No structured result was produced for the run shown in this response."
    ),
    "schema_mismatch": (
        "The extraction model did not return a result that matches the schema "
        "within 3 attempts."
    ),
    "result_too_large": (
        "The structured result is too large to return: it exceeds a size limit."
    ),
    "model_error": "The extraction model was unavailable or its call failed.",
    "invalid_stored_format": (
        "The response_format stored on this task is no longer valid, "
        "so no structured result was extracted."
    ),
    "extraction_timeout": (
        "Structured result extraction did not finish within 120 seconds."
    ),
    "internal_error": (
        "Structured result extraction failed because of an internal error."
    ),
}
# The codes a record may store, by the status that owns them. ``not_produced``
# is never stored: only the projection reports it.
_INCOMPLETE_CODES = frozenset(
    {"agent_partial", "agent_blocked", "information_missing", "value_not_expressible"}
)
_FAILED_CODES = frozenset(
    {
        "schema_mismatch",
        "result_too_large",
        "model_error",
        "invalid_stored_format",
        "extraction_timeout",
        "internal_error",
    }
)

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


@dataclass(frozen=True)
class StructuredOutputFields:
    """The three fields of a task response that come from the structured result."""

    structured_output: dict[str, Any] | None = None
    structured_output_status: str | None = None
    structured_output_reason: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "structured_output": self.structured_output,
            "structured_output_status": self.structured_output_status,
            "structured_output_reason": self.structured_output_reason,
        }


_NO_FIELDS = StructuredOutputFields()


def response_format_configured(agent_config: Any) -> bool:
    """Whether ``agent_config`` holds a format: a dict with the format key set
    to something other than None."""
    return type(agent_config) is dict and agent_config.get(_FORMAT_KEY) is not None


def _reason_fields(
    code: str, lists: dict[str, list[str]] | None = None
) -> dict[str, Any]:
    """A reason: the code, its fixed message, then the lists the record carried."""
    return {"code": code, "message": REASON_MESSAGES[code], **(lists or {})}


def _is_field_list(value: Any) -> bool:
    return (
        type(value) is list
        and 0 < len(value) <= MAX_REASON_FIELDS
        and all(type(item) is str for item in value)
    )


def _reason_problem(code: Any, status: str, reason: dict[str, Any]) -> str | None:
    """What is wrong with the reason of an incomplete or failed record, if anything."""
    allowed = _INCOMPLETE_CODES if status == "incomplete" else _FAILED_CODES
    if type(code) is not str or code not in allowed:
        return "reason code"
    for key in _REASON_LIST_KEYS:
        if key in reason and not _is_field_list(reason[key]):
            return key
    if "missing_fields" in reason and code != "information_missing":
        return "missing_fields"
    if "unexpressible_fields" in reason and code not in (
        "information_missing",
        "value_not_expressible",
    ):
        return "unexpressible_fields"
    if code == "information_missing" and "missing_fields" not in reason:
        return "missing_fields"
    if code == "value_not_expressible" and "unexpressible_fields" not in reason:
        return "unexpressible_fields"
    return None


def _record_problem(record: dict[str, Any]) -> str | None:
    """The first rule of the record shape that ``record`` breaks, or None.

    ``record`` is a dict whose ``run_id`` is a string. Extra keys, in the
    record and in its reason, are ignored.
    """
    if "status" not in record or "output" not in record or "reason" not in record:
        return "keys"
    status, output, reason = record["status"], record["output"], record["reason"]
    if type(status) is not str or status not in ("ready", "incomplete", "failed"):
        return "status"
    if status == "ready":
        if type(output) is not dict:
            return "output"
        return None if reason is None else "reason"
    if output is not None:
        return "output"
    if type(reason) is not dict:
        return "reason"
    return _reason_problem(reason.get("code"), status, reason)


def _not_produced(format_configured: bool) -> StructuredOutputFields:
    if not format_configured:
        return _NO_FIELDS
    return StructuredOutputFields(None, "incomplete", _reason_fields("not_produced"))


def _malformed(
    run_id: str | None, problem: str, format_configured: bool
) -> StructuredOutputFields:
    logger.warning(
        "The structured output record of run %s is malformed (%s)", run_id, problem
    )
    if not format_configured:
        return _NO_FIELDS
    return StructuredOutputFields(None, "failed", _reason_fields("internal_error"))


def project_structured_output(
    *,
    status: TaskStatus,
    run_id: str | None,
    record: Any,
    format_configured: bool,
) -> StructuredOutputFields:
    """The structured output fields of a task response.

    ``status`` is the effective status the response reports: the caller reads
    the queue of accepted-but-not-started turns before it reads the task row,
    and passes the status that results, so a task whose next turn is queued
    reads as pending here. ``run_id`` and ``record`` come from the same read
    of the task row, and ``format_configured`` is
    :func:`response_format_configured` of that row's ``agent_config``.

    Only a COMPLETED task reports anything. The record is the stored one when
    it belongs to ``run_id``; a record of another run, or none, reads as
    ``not_produced`` when a format is configured and as nothing otherwise. A
    record that does not have the stored shape reads as ``internal_error``
    when a format is configured, as nothing otherwise, and is logged. The
    run is compared first, so the shape of another run's record is not
    examined.
    """
    if status is not TaskStatus.COMPLETED:
        return _NO_FIELDS
    if record is None:
        return _not_produced(format_configured)
    if type(record) is not dict or type(record.get("run_id")) is not str:
        return _malformed(run_id, "run_id", format_configured)
    if record["run_id"] != run_id:
        return _not_produced(format_configured)
    problem = _record_problem(record)
    if problem is not None:
        return _malformed(run_id, problem, format_configured)
    if record["status"] == "ready":
        return StructuredOutputFields(record["output"], "ready", None)
    reason = record["reason"]
    lists = {key: list(reason[key]) for key in _REASON_LIST_KEYS if key in reason}
    return StructuredOutputFields(
        None, record["status"], _reason_fields(reason["code"], lists)
    )
