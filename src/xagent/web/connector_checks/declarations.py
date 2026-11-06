"""The smoke V2 declaration format: what a connector's checks are.

A connector declares read checks, write checks and tool exemptions. Every
check has a stable id, ``<app_id>.<name>``, which the run result and the
result diff key on, so an id is never derived from a position and never
reused for a different check.

A read check calls one tool with fixed arguments and asserts on the result.
A write check creates an object, verifies it and cleans it up; it also names
the tenant it must run in and the approval verdict the deployment's write
policy should give each write tool, because the probe path bypasses the
approval gate. Until the write-check safety rails land (#3564) the engine
runs no write check and reports each as ``writes_disabled``; the write
switch is per connector and keyed by its app id, so a check does not carry
one. A tool that has no check carries an exemption with its reason instead.

Arguments, and assertions' expected values, are JSON values, fixture keys
(``fixtures.FixtureRef``), or in a write check a run-scoped name
(``RunScopedName``). The caller of the engine never supplies either.

Assertions read a payload taken from the tool's normalized result
(``{"content": [...], "structured_content": ..., "is_error": ...}``):

- ``json_text``: the content is exactly one text item holding a JSON object,
  the in-repo servers' shape.
- ``json_text_items``: the list of every text item parsed as JSON.
- ``structured_content``: the MCP ``structuredContent`` object.

A path is a tuple of object keys and list indexes into that payload. A check
whose result is flagged ``is_error`` fails before any assertion runs.
``L3Envelope`` judges the result exactly as the L3 probe does
(``evaluate_probe_result`` in ``services/connector_probe.py``) and is the only
assertion of the checks derived from ``connector_probes.py``.

``validate_connector_checks`` enforces these rules; the registry test runs it
over every declaration.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from ..connector_probes import UNPROBEABLE_REASONS, UnprobeableReason
from .codes import APP_ID_PATTERN, TOOL_NAME_PATTERN, connector_label, is_check_id
from .fixtures import FixtureRef, FixtureValue

# Bumped on any change a declared check or a consumer must adapt to.
DECLARATION_FORMAT_VERSION = 1

_RUN_SCOPED_SUFFIX_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

RUN_SCOPED_PREFIX = "xagent-smoke-"

PayloadSource = Literal["json_text", "json_text_items", "structured_content"]
JsonType = Literal["object", "array", "string", "number", "boolean", "null"]
# The approval gate's decisions (GateDecisionValue in
# core/tools/adapters/vibe/mcp_approval_gate.py); a test pins the two equal.
ExpectedVerdict = Literal["allow", "require_approval", "deny"]

PayloadPath = tuple[str | int, ...]


class DeclarationError(ValueError):
    """A declaration breaks a rule of the format."""


@dataclass(frozen=True)
class RunScopedName:
    """A name the engine fills in as ``xagent-smoke-<run_id>-<suffix>``.

    Cleanup and the 24 h sweeper delete only objects carrying the prefix.
    """

    suffix: str

    def __post_init__(self) -> None:
        if not _RUN_SCOPED_SUFFIX_PATTERN.fullmatch(self.suffix):
            raise ValueError(f"invalid run-scoped name suffix: {self.suffix!r}")


# Assertions


@dataclass(frozen=True)
class StatusSuccess:
    """The payload's ``status`` is ``"success"``: the in-repo envelope."""


@dataclass(frozen=True)
class HasKeys:
    """The object at ``path`` carries every key in ``keys``."""

    path: PayloadPath
    keys: tuple[str, ...]


@dataclass(frozen=True)
class EachHasKeys:
    """Every item of the array at ``path`` carries every key in ``keys``."""

    path: PayloadPath
    keys: tuple[str, ...]


@dataclass(frozen=True)
class IsType:
    """The value at ``path`` has JSON type ``type``."""

    path: PayloadPath
    type: JsonType


@dataclass(frozen=True)
class Length:
    """The array at ``path`` has between ``min`` and ``max`` items."""

    path: PayloadPath
    min: int = 0
    max: int | None = None


@dataclass(frozen=True)
class Equals:
    """The value at ``path`` equals ``value``: typically a seeded fixture, or
    in a write step the run-scoped name it created."""

    path: PayloadPath
    value: FixtureValue | FixtureRef | RunScopedName | None


@dataclass(frozen=True)
class L3Envelope:
    """Judged as the L3 probe judges it (``evaluate_probe_result``)."""

    expect_keys: tuple[str, ...] = ()


Assertion = (
    StatusSuccess | HasKeys | EachHasKeys | IsType | Length | Equals | L3Envelope
)


# Checks


@dataclass(frozen=True)
class ReadCheck:
    """One read-only tool call and the assertions on its result.

    ``require_read_only_hint`` skips the check unless the server advertises
    the tool with ``readOnlyHint: true``, for servers whose tools are
    vendor-owned (#3566).
    """

    id: str
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    payload: PayloadSource = "json_text"
    assertions: tuple[Assertion, ...] = ()
    require_read_only_hint: bool = False


@dataclass(frozen=True)
class UnprobeableCheck:
    """A check slot that cannot run, with the L3 reason; its row is always
    ``unprobeable``."""

    id: str
    reason: UnprobeableReason


@dataclass(frozen=True)
class ToolStep:
    """One tool call inside a write check or a tenant pin."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    payload: PayloadSource = "json_text"
    assertions: tuple[Assertion, ...] = ()


@dataclass(frozen=True)
class TenantPin:
    """Before any write, ``step`` reads the connection's tenant id at
    ``path``; it must equal the registry's id, ``expected``."""

    step: ToolStep
    path: PayloadPath
    expected: FixtureRef


@dataclass(frozen=True)
class WriteCheck:
    """Create, verify and clean up one object inside a test tenant.

    ``create`` must name its object with a ``RunScopedName``.
    ``expected_verdicts`` names the approval verdict for exactly the write
    tools, ``create`` and ``cleanup``.
    """

    id: str
    create: ToolStep
    verify: ToolStep
    cleanup: ToolStep
    tenant_pin: TenantPin
    expected_verdicts: Mapping[str, ExpectedVerdict]


@dataclass(frozen=True)
class ToolExemption:
    """A tool deliberately left without a check, and why."""

    tool: str
    reason: str


@dataclass(frozen=True)
class ConnectorChecks:
    """Every check and exemption one connector declares."""

    app_id: str
    reads: tuple[ReadCheck | UnprobeableCheck, ...] = ()
    writes: tuple[WriteCheck, ...] = ()
    exemptions: tuple[ToolExemption, ...] = ()

    def checked_tools(self) -> frozenset[str]:
        """Tools a check calls as its subject; a tenant pin's read is not
        one."""
        tools = {check.tool for check in self.reads if isinstance(check, ReadCheck)}
        for write in self.writes:
            tools.update(
                step.tool for step in (write.create, write.verify, write.cleanup)
            )
        return frozenset(tools)


# Validation


def validate_connector_checks(checks: ConnectorChecks) -> None:
    """Raise ``DeclarationError`` naming the first rule ``checks`` breaks."""
    app_id = checks.app_id
    if not APP_ID_PATTERN.fullmatch(app_id):
        raise DeclarationError(f"invalid app id: {app_id!r}")
    if connector_label(app_id) != app_id:
        raise DeclarationError(f"app id {app_id!r} would not be its own result label")

    seen: set[str] = set()
    for read in checks.reads:
        if not isinstance(read, (ReadCheck, UnprobeableCheck)):
            raise DeclarationError(
                f"{app_id}: reads take a ReadCheck or UnprobeableCheck, "
                f"not {type(read).__name__}"
            )
        _validate_check_id(app_id, read.id, seen)
        if isinstance(read, ReadCheck):
            _validate_read(read)
        elif read.reason not in UNPROBEABLE_REASONS:
            raise DeclarationError(
                f"{read.id}: unknown unprobeable reason {read.reason!r}"
            )
    for write in checks.writes:
        if not isinstance(write, WriteCheck):
            raise DeclarationError(
                f"{app_id}: writes take a WriteCheck, not {type(write).__name__}"
            )
        _validate_check_id(app_id, write.id, seen)
        _validate_write(write)

    checked = checks.checked_tools()
    exempted: set[str] = set()
    for exemption in checks.exemptions:
        _validate_tool_name(f"{app_id} exemption", exemption.tool)
        if exemption.tool in exempted:
            raise DeclarationError(f"{app_id}: {exemption.tool} exempted twice")
        if exemption.tool in checked:
            raise DeclarationError(
                f"{app_id}: {exemption.tool} is both checked and exempted"
            )
        if not exemption.reason.strip():
            raise DeclarationError(
                f"{app_id}: {exemption.tool} exemption has no reason"
            )
        exempted.add(exemption.tool)


def _validate_check_id(app_id: str, check_id: str, seen: set[str]) -> None:
    prefix = f"{app_id}."
    if not check_id.startswith(prefix):
        raise DeclarationError(f"check id {check_id!r} must start with {prefix!r}")
    # The rule a result row holds the id to: a code, at most 80 characters.
    if not is_check_id(app_id, check_id):
        raise DeclarationError(f"invalid check id: {check_id!r}")
    if check_id in seen:
        raise DeclarationError(f"duplicate check id: {check_id}")
    seen.add(check_id)


def _validate_read(check: ReadCheck) -> None:
    _validate_call(
        check.id, check.tool, check.arguments, check.payload, check.assertions
    )
    _refuse_run_scoped_name(check.id, check.arguments, check.assertions)
    _require_assertion(check.id, "check", check.assertions)


def _validate_write(check: WriteCheck) -> None:
    where = check.id
    pin = check.tenant_pin
    for name in ("create", "verify", "cleanup"):
        if not isinstance(getattr(check, name), ToolStep):
            raise DeclarationError(f"{where}: {name} must be a ToolStep")
    if not isinstance(pin, TenantPin):
        raise DeclarationError(f"{where}: tenant pin must be a TenantPin")
    if not isinstance(pin.step, ToolStep):
        raise DeclarationError(f"{where}: tenant pin step must be a ToolStep")
    if not isinstance(pin.expected, FixtureRef):
        raise DeclarationError(f"{where}: tenant pin expected must be a FixtureRef")
    for step in (check.create, check.verify, check.cleanup, pin.step):
        _validate_call(where, step.tool, step.arguments, step.payload, step.assertions)
    _validate_path(where, pin.path)
    for name in ("create", "verify", "cleanup"):
        _require_assertion(where, name, getattr(check, name).assertions)
    _refuse_run_scoped_name(where, pin.step.arguments, pin.step.assertions)
    if not _contains_run_scoped_name(check.create.arguments):
        raise DeclarationError(
            f"{where}: create must name its object with a run-scoped name"
        )
    if not _contains_run_scoped_name(check.cleanup.arguments):
        raise DeclarationError(
            f"{where}: cleanup must name its object with a run-scoped name"
        )
    write_tools = {check.create.tool, check.cleanup.tool}
    verdicts = check.expected_verdicts
    if set(verdicts) != write_tools:
        raise DeclarationError(
            f"{where}: an expected verdict is needed for exactly {sorted(write_tools)}"
        )
    allowed = set(get_args(ExpectedVerdict))
    if any(verdict not in allowed for verdict in verdicts.values()):
        raise DeclarationError(f"{where}: unknown expected verdict")


def _require_assertion(
    where: str, name: str, assertions: tuple[Assertion, ...]
) -> None:
    if not assertions:
        raise DeclarationError(f"{where}: {name} needs at least one assertion")


def _validate_call(
    where: str,
    tool: str,
    arguments: Mapping[str, Any],
    payload: str,
    assertions: tuple[Assertion, ...],
) -> None:
    _validate_tool_name(where, tool)
    if not isinstance(arguments, Mapping):
        raise DeclarationError(f"{where}: arguments must be a mapping")
    _validate_json_value(where, arguments)
    if payload not in get_args(PayloadSource):
        raise DeclarationError(f"{where}: unknown payload source {payload!r}")
    for assertion in assertions:
        _validate_assertion(where, assertion)
    if any(isinstance(item, L3Envelope) for item in assertions):
        # evaluate_probe_result reads the whole result, as the L3 route does.
        if len(assertions) > 1:
            raise DeclarationError(f"{where}: L3Envelope must be the only assertion")
        if payload != "json_text":
            raise DeclarationError(f"{where}: L3Envelope reads the json_text payload")


def _validate_tool_name(where: str, tool: str) -> None:
    if not isinstance(tool, str) or not TOOL_NAME_PATTERN.fullmatch(tool):
        raise DeclarationError(f"{where}: invalid tool name {tool!r}")


def _validate_json_value(where: str, value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int, FixtureRef, RunScopedName)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DeclarationError(f"{where}: argument is not a finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise DeclarationError(f"{where}: argument keys must be strings")
            _validate_json_value(where, item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(where, item)
        return
    raise DeclarationError(
        f"{where}: argument of type {type(value).__name__} is not a JSON value"
    )


def _contains_run_scoped_name(value: Any) -> bool:
    if isinstance(value, RunScopedName):
        return True
    if isinstance(value, Mapping):
        return any(_contains_run_scoped_name(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_run_scoped_name(item) for item in value)
    return False


def _refuse_run_scoped_name(
    where: str, arguments: Mapping[str, Any], assertions: tuple[Assertion, ...]
) -> None:
    """Only a write step may use a run-scoped name, in an argument or in an
    ``Equals`` value."""
    values = [
        arguments,
        *(item.value for item in assertions if isinstance(item, Equals)),
    ]
    if _contains_run_scoped_name(values):
        raise DeclarationError(f"{where}: only a write step may use a run-scoped name")


def _validate_assertion(where: str, assertion: Assertion) -> None:
    if isinstance(assertion, StatusSuccess):
        return
    if isinstance(assertion, L3Envelope):
        _validate_keys(where, assertion.expect_keys, allow_empty=True)
        return
    _validate_path(where, assertion.path)
    if isinstance(assertion, (HasKeys, EachHasKeys)):
        _validate_keys(where, assertion.keys, allow_empty=False)
    elif isinstance(assertion, IsType):
        if assertion.type not in get_args(JsonType):
            raise DeclarationError(f"{where}: unknown JSON type {assertion.type!r}")
    elif isinstance(assertion, Length):
        if assertion.min < 0 or (
            assertion.max is not None and assertion.max < assertion.min
        ):
            raise DeclarationError(f"{where}: invalid length bounds")
    elif isinstance(assertion, Equals):
        value = assertion.value
        if isinstance(value, float) and not math.isfinite(value):
            raise DeclarationError(f"{where}: Equals value is not a finite number")
        if value is not None and not isinstance(
            value, (str, bool, int, float, FixtureRef, RunScopedName)
        ):
            raise DeclarationError(f"{where}: Equals takes a scalar value")
    else:
        raise DeclarationError(f"{where}: unknown assertion {type(assertion).__name__}")


def _validate_keys(where: str, keys: tuple[str, ...], *, allow_empty: bool) -> None:
    if (not keys and not allow_empty) or not all(
        isinstance(key, str) and key for key in keys
    ):
        raise DeclarationError(f"{where}: keys must be non-empty strings")


def _validate_path(where: str, path: PayloadPath) -> None:
    for segment in path:
        if isinstance(segment, bool) or not (
            isinstance(segment, str) or (isinstance(segment, int) and segment >= 0)
        ):
            raise DeclarationError(f"{where}: invalid path segment {segment!r}")
