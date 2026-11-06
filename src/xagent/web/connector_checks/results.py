"""The smoke V2 run result and its JSON document.

One document per run and identity. It holds one ``session`` row per
connector (starting it and listing its tools) and one ``check`` row per
declared check, keyed by ``(identity, connector, check)`` so the result diff
can compare runs. When a session does not pass, none of its checks runs:
each reports ``skip`` (``session_unavailable``, or ``writes_disabled`` for a
write check) or stays ``unprobeable``, so one broken connector is one
failure, not one per check. A run that could not be set up (the identity or
its connectors could not be resolved) carries a ``failure`` code and no
rows, and is never ``complete``.

Outcomes are ``pass``, ``fail``, ``skip`` and ``unprobeable``. A ``fail`` or
``skip`` carries a ``FailureClass``; ``reason`` is an optional code that
refines it (the builder's credential reason, a load phase, an identity's
skip code, an assertion's code, a fixture key). The document is built to
carry codes, ids, tool names and timing only, never a token, a secret, a
response body, a fixture value or an end-user id: the rows and the packaged
JSON Schema (``result.schema.json``) refuse any value that is not a code or
a known label, and the callers that build rows put only codes in them.

``RESULT_SCHEMA_VERSION`` is bumped on any change a consumer must adapt to.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib import resources
from typing import Any, Literal

from ..connector_probes import UNPROBEABLE_REASONS
from ..services.mcp_connector_session import ConnectorToolsOutcome, ConnectorToolsResult
from .codes import APP_ID_PATTERN, TOOL_NAME_PATTERN, is_check_id, is_code
from .declarations import DECLARATION_FORMAT_VERSION

RESULT_SCHEMA_ID = "xagent.connector_smoke.result"
RESULT_SCHEMA_VERSION = 1

IDENTITY_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,31}")
ENVIRONMENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")

CheckKind = Literal["read", "write"]


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    UNPROBEABLE = "unprobeable"


class FailureClass(str, Enum):
    # Failures (fail). These first ones are typically a session's.
    NOT_FOUND = "not_found"
    # The user's credential is missing or unusable: a dead grant.
    NOT_CONNECTED = "not_connected"
    # The vendor rejected the platform's own client credential (#3580): every
    # user of the provider is affected, so it is routed apart.
    CLIENT_REJECTED = "client_rejected"
    UNSUPPORTED = "unsupported"
    FAILED_TO_START = "failed_to_start"
    NO_TOOLS = "no_tools"
    TIMED_OUT = "timed_out"
    # The run's total timeout ended before the connector was reached.
    NOT_REACHED = "not_reached"
    # Typically one check's.
    TOOL_MISSING = "tool_missing"
    CALL_FAILED = "call_failed"
    ASSERTION_FAILED = "assertion_failed"
    # Write-check safety (#3564).
    TENANT_MISMATCH = "tenant_mismatch"
    VERDICT_MISMATCH = "verdict_mismatch"
    CLEANUP_FAILED = "cleanup_failed"
    # Not run (skip).
    SKIPPED_BY_IDENTITY = "skipped_by_identity"
    SESSION_UNAVAILABLE = "session_unavailable"
    NOT_DECLARED = "not_declared"
    # ``reason`` is the unresolved fixture key.
    FIXTURE_MISSING = "fixture_missing"
    READ_ONLY_HINT_MISSING = "read_only_hint_missing"
    WRITES_DISABLED = "writes_disabled"


SKIP_CLASSES = frozenset(
    {
        FailureClass.SKIPPED_BY_IDENTITY,
        FailureClass.SESSION_UNAVAILABLE,
        FailureClass.NOT_DECLARED,
        FailureClass.FIXTURE_MISSING,
        FailureClass.READ_ONLY_HINT_MISSING,
        FailureClass.WRITES_DISABLED,
    }
)
FAIL_CLASSES = frozenset(FailureClass) - SKIP_CLASSES

_SESSION_CLASSES = {
    ConnectorToolsOutcome.NOT_FOUND: FailureClass.NOT_FOUND,
    ConnectorToolsOutcome.NOT_CONNECTED: FailureClass.NOT_CONNECTED,
    ConnectorToolsOutcome.UNSUPPORTED: FailureClass.UNSUPPORTED,
    ConnectorToolsOutcome.FAILED_TO_START: FailureClass.FAILED_TO_START,
    ConnectorToolsOutcome.NO_TOOLS: FailureClass.NO_TOOLS,
    ConnectorToolsOutcome.TIMED_OUT: FailureClass.TIMED_OUT,
}

# The reason the connector session reports when the vendor rejects the
# platform's client credential (#3580), whatever outcome carries it. It may
# name the credential's provider after a colon (``client_rejected:google``),
# which the row keeps so the alert can name the credential.
CLIENT_REJECTED_REASON = "client_rejected"


def classify_session(
    listing: ConnectorToolsResult,
) -> tuple[Outcome, FailureClass | None, str | None]:
    """A session row's outcome, class and reason for one listing.

    A reason that does not look like a code is dropped, so a message can
    never reach the result.
    """
    if listing.outcome is ConnectorToolsOutcome.OK:
        return Outcome.PASS, None, None
    reason = listing.reason if is_code(listing.reason) else None
    if reason is not None and reason.partition(":")[0] == CLIENT_REJECTED_REASON:
        return Outcome.FAIL, FailureClass.CLIENT_REJECTED, reason
    return Outcome.FAIL, _SESSION_CLASSES[listing.outcome], reason


@dataclass(frozen=True)
class SessionRow:
    """Starting one connector and listing its tools."""

    connector: str
    outcome: Outcome
    failure_class: FailureClass | None = None
    reason: str | None = None
    duration_ms: int = 0

    def __post_init__(self) -> None:
        _validate_connector(self.connector)
        _validate_outcome(
            self.outcome, self.failure_class, self.reason, unprobeable=False
        )
        _validate_duration(self.duration_ms)

    def to_json(self) -> dict[str, Any]:
        return _row_json("session", self.connector, None, None, None, self)


@dataclass(frozen=True)
class CheckRow:
    """One declared check. ``tool`` is None for a check with no tool to call
    (unprobeable, or no declaration)."""

    connector: str
    check: str
    kind: CheckKind
    tool: str | None
    outcome: Outcome
    failure_class: FailureClass | None = None
    reason: str | None = None
    duration_ms: int = 0

    def __post_init__(self) -> None:
        _validate_connector(self.connector)
        if not is_check_id(self.connector, self.check):
            raise ValueError(f"invalid check id for {self.connector}: {self.check!r}")
        if self.kind not in ("read", "write"):
            raise ValueError(f"invalid check kind: {self.kind!r}")
        if self.tool is not None and not TOOL_NAME_PATTERN.fullmatch(self.tool):
            raise ValueError(f"invalid tool name: {self.tool!r}")
        _validate_outcome(
            self.outcome, self.failure_class, self.reason, unprobeable=True
        )
        _validate_duration(self.duration_ms)

    def to_json(self) -> dict[str, Any]:
        return _row_json(
            "check", self.connector, self.check, self.kind, self.tool, self
        )


@dataclass(frozen=True)
class RunResult:
    """One run under one identity.

    ``complete`` is False when the run's total timeout cut it or ``failure``
    is set. ``failure`` is a code for a run that could not be set up
    (``<stage>:<reason>``, such as ``enumeration:no_connectors``); such a run
    has no rows, and the result diff must read it as no data rather than as
    every connector recovering. ``rows`` keep the order the engine produced
    them.
    """

    identity: str
    started_at: datetime
    finished_at: datetime
    complete: bool
    rows: tuple[SessionRow | CheckRow, ...]
    environment: str | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        if not IDENTITY_PATTERN.fullmatch(self.identity):
            raise ValueError(f"invalid identity label: {self.identity!r}")
        if self.environment is not None and not ENVIRONMENT_PATTERN.fullmatch(
            self.environment
        ):
            raise ValueError(f"invalid environment label: {self.environment!r}")
        for moment in (self.started_at, self.finished_at):
            if moment.utcoffset() is None:
                raise ValueError("run times must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("a run cannot finish before it starts")
        if self.failure is not None:
            if not is_code(self.failure):
                raise ValueError("a run failure must be a code")
            if self.complete or self.rows:
                raise ValueError("a run that failed to set up is incomplete and empty")
        self._check_rows()

    def _check_rows(self) -> None:
        sessions: dict[str, SessionRow] = {}
        checks: set[str] = set()
        for row in self.rows:
            if isinstance(row, SessionRow):
                if row.connector in sessions:
                    raise ValueError(f"two session rows for {row.connector}")
                sessions[row.connector] = row
                continue
            session = sessions.get(row.connector)
            if session is None:
                raise ValueError(f"{row.check} precedes its connector's session row")
            if row.check in checks:
                raise ValueError(f"duplicate check row: {row.check}")
            checks.add(row.check)
            if session.outcome is not Outcome.PASS and row.outcome not in (
                Outcome.SKIP,
                Outcome.UNPROBEABLE,
            ):
                raise ValueError(f"{row.check} ran without a session")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA_ID,
            "schema_version": RESULT_SCHEMA_VERSION,
            "declaration_format_version": DECLARATION_FORMAT_VERSION,
            "identity": self.identity,
            "environment": self.environment,
            "started_at": _timestamp(self.started_at),
            "finished_at": _timestamp(self.finished_at),
            "complete": self.complete,
            "failure": self.failure,
            "rows": [row.to_json() for row in self.rows],
        }


def load_result_schema() -> dict[str, Any]:
    """The JSON Schema every result document validates against."""
    package = __name__.rpartition(".")[0]
    text = (
        resources.files(package)
        .joinpath("result.schema.json")
        .read_text(encoding="utf-8")
    )
    schema: dict[str, Any] = json.loads(text)
    return schema


def _validate_connector(connector: str) -> None:
    if not APP_ID_PATTERN.fullmatch(connector):
        raise ValueError(f"invalid connector app id: {connector!r}")


def _validate_outcome(
    outcome: Outcome,
    failure_class: FailureClass | None,
    reason: str | None,
    *,
    unprobeable: bool,
) -> None:
    if reason is not None and not is_code(reason):
        raise ValueError("a reason must be a code")
    if outcome is Outcome.PASS:
        valid = failure_class is None
    elif outcome is Outcome.FAIL:
        valid = failure_class in FAIL_CLASSES
    elif outcome is Outcome.SKIP:
        valid = failure_class in SKIP_CLASSES
    else:
        valid = unprobeable and failure_class is None and reason in UNPROBEABLE_REASONS
    if not valid:
        raise ValueError(f"{outcome.value} cannot carry class {failure_class}")


def _validate_duration(duration_ms: int) -> None:
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, int)
        or duration_ms < 0
    ):
        raise ValueError("duration_ms must be a non-negative integer")


def _row_json(
    row_type: str,
    connector: str,
    check: str | None,
    kind: str | None,
    tool: str | None,
    row: SessionRow | CheckRow,
) -> dict[str, Any]:
    return {
        "row": row_type,
        "connector": connector,
        "check": check,
        "kind": kind,
        "tool": tool,
        "outcome": row.outcome.value,
        "failure_class": (
            row.failure_class.value if row.failure_class is not None else None
        ),
        "reason": row.reason,
        "duration_ms": row.duration_ms,
    }


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
