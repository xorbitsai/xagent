"""The smoke V2 declaration format: what a connector's checks are.

A connector declares read checks, write checks, tool exemptions and, when
it has write checks, a sweep. Every check has a stable id, ``<app_id>.<name>``, which the run result and the
result diff key on, so an id is never derived from a position and never
reused for a different check.

A read check calls one tool with fixed arguments and asserts on the result.
A write check creates an object, verifies it and cleans it up. It also
names the tenant it must run in, and the approval verdict the deployment's
write policy gives a restricted actor for each write tool, because the probe
path bypasses both the approval gate and the per-user tool overrides. Its
create writes into the connector's write area (a fixture key
``<app_id>.write_...``) and names its object with a ``RunScopedName``. Its
cleanup deletes that object by the same name, or by a ``CreatedRef`` to the
id create returned once a cleanup guard has read the object by that id and
confirmed its name. A connector with write checks declares a ``Sweep`` for
what aborted runs leave behind. The engine runs a write check only for a
connector listed in ``XAGENT_CONNECTOR_SMOKE_WRITE_APPS`` (``writes.py``) and
reports it as ``writes_disabled`` otherwise; the switch is per connector and
keyed by its app id, so a check does not carry one. A tool that has no check carries an
exemption with its reason instead.

Arguments, and assertions' expected values, are JSON values or fixture keys
(``fixtures.FixtureRef``). A write check's steps may also use a
``RunScopedName``, and its verify, cleanup and cleanup guard a
``CreatedRef``; a sweep's delete takes an ``ItemRef``. The caller of the
engine never supplies any of them. No string in the arguments of a write
check's or a sweep's step contains an ``@``: an address comes from a fixture.

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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, get_args

from ..connector_probes import UNPROBEABLE_REASONS, UnprobeableReason
from .codes import APP_ID_PATTERN, TOOL_NAME_PATTERN, connector_label, is_check_id
from .fixtures import FixtureRef, FixtureValue

# Bumped on any change a declared check or a consumer must adapt to.
DECLARATION_FORMAT_VERSION = 2

_RUN_SCOPED_SUFFIX_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

RUN_SCOPED_PREFIX = "xagent-smoke-"

# How a write-area fixture key's name starts (``google-drive.write_folder_id``).
_WRITE_AREA_PREFIX = "write_"

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
    """A name the engine fills in as ``xagent-smoke-<run_id>-<suffix>``, so
    every object a write check creates carries its run's prefix."""

    suffix: str

    def __post_init__(self) -> None:
        if not _RUN_SCOPED_SUFFIX_PATTERN.fullmatch(self.suffix):
            raise ValueError(f"invalid run-scoped name suffix: {self.suffix!r}")


@dataclass(frozen=True)
class CreatedRef:
    """The value at ``path`` in the create step's payload: typically the id
    the vendor gave the new object (``("issue", "key")`` for Jira).

    Only a write check's verify, cleanup and cleanup guard may use one, and a
    cleanup that deletes by one needs a cleanup guard.
    """

    path: PayloadPath

    def __post_init__(self) -> None:
        if not _is_reference_path(self.path):
            raise ValueError(f"invalid created-object path: {self.path!r}")


@dataclass(frozen=True)
class ItemRef:
    """The value at ``path`` in one item the sweep's ``find`` listed: the id
    its ``delete`` takes. Only a sweep's ``delete`` may use one."""

    path: PayloadPath

    def __post_init__(self) -> None:
        if not _is_reference_path(self.path):
            raise ValueError(f"invalid listed-item path: {self.path!r}")


def _is_path_segment(segment: object) -> bool:
    # A bool is an int, and never a list index.
    return not isinstance(segment, bool) and (
        isinstance(segment, str) or (isinstance(segment, int) and segment >= 0)
    )


def _is_reference_path(path: object) -> bool:
    """A non-empty tuple of segments: a tuple keeps a reference hashable."""
    return (
        isinstance(path, tuple)
        and bool(path)
        and all(_is_path_segment(segment) for segment in path)
    )


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
class Contains:
    """The string at ``path`` contains ``value``."""

    path: PayloadPath
    value: str


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
    in a write step the run-scoped name or the id it created."""

    path: PayloadPath
    value: FixtureValue | FixtureRef | RunScopedName | CreatedRef | None


@dataclass(frozen=True)
class L3Envelope:
    """Judged as the L3 probe judges it (``evaluate_probe_result``)."""

    expect_keys: tuple[str, ...] = ()


Assertion = (
    StatusSuccess
    | HasKeys
    | EachHasKeys
    | Contains
    | IsType
    | Length
    | Equals
    | L3Envelope
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
    """One tool call inside a write check, a tenant pin or a sweep."""

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    payload: PayloadSource = "json_text"
    assertions: tuple[Assertion, ...] = ()


@dataclass(frozen=True)
class TenantPin:
    """Before any write, ``step`` reads, at ``path``, a value only this
    tenant's account has: its tenant id where a tool of the connector returns
    one, or the id of the tenant's own write-area container, which the seed
    keeps unshared. It must equal the registry's value, ``expected``."""

    step: ToolStep
    path: PayloadPath
    expected: FixtureRef


@dataclass(frozen=True)
class WriteCheck:
    """Create, verify and clean up one object inside a test tenant.

    ``create`` must write into the connector's write area (a fixture key
    ``<app_id>.write_...``) and name its object with a ``RunScopedName``.
    ``cleanup`` deletes that object by the same name or by a ``CreatedRef``.
    When it deletes by a ``CreatedRef``, ``cleanup_guard`` must first read
    the object by that same reference, with a tool none of the connector's
    write checks creates or cleans up with, and assert the name create gave
    it. As long as the guard finds the object by that reference, which review
    checks, a wrongly declared path is caught before anything is deleted.
    ``expected_verdicts`` names the approval verdict for exactly the write
    tools, ``create`` and ``cleanup``.
    """

    id: str
    create: ToolStep
    verify: ToolStep
    cleanup: ToolStep
    tenant_pin: TenantPin
    expected_verdicts: Mapping[str, ExpectedVerdict]
    cleanup_guard: ToolStep | None = None


@dataclass(frozen=True)
class Sweep:
    """How the sweeper finds and deletes what write checks left behind.

    ``find`` lists the connector's write area. ``items`` is the path of the
    array of objects in its payload, and ``name`` the path of an item's name,
    which the sweeper (#3564) is to match against ``xagent-smoke-`` and read
    the object's age from. ``delete`` removes one item, taken by an ``ItemRef``,
    with a tool one of the connector's write checks cleans up with, so its
    approval verdict is declared. ``tenant_pin`` names the same tenant as the
    write checks' pins.
    """

    tenant_pin: TenantPin
    find: ToolStep
    items: PayloadPath
    name: PayloadPath
    delete: ToolStep


@dataclass(frozen=True)
class ToolExemption:
    """A tool deliberately left without a check, and why."""

    tool: str
    reason: str


@dataclass(frozen=True)
class ConnectorChecks:
    """Every check and exemption one connector declares, and the sweep of a
    connector with write checks."""

    app_id: str
    reads: tuple[ReadCheck | UnprobeableCheck, ...] = ()
    writes: tuple[WriteCheck, ...] = ()
    exemptions: tuple[ToolExemption, ...] = ()
    sweep: Sweep | None = None

    def checked_tools(self) -> frozenset[str]:
        """Tools a check calls as its subject; a tenant pin's read, a cleanup
        guard and the sweep are not."""
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
        _validate_write(app_id, write)
    _refuse_shared_names(checks)
    _validate_sweep(checks)
    _refuse_writes_in_read_steps(checks)

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
    if not isinstance(check_id, str):
        raise DeclarationError(
            f"{app_id}: a check id must be a string, not {type(check_id).__name__}"
        )
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
    _refuse_references(check.id, "a read check", check.arguments, check.assertions)
    _require_assertion(check.id, "check", check.assertions)


def _validate_write(app_id: str, check: WriteCheck) -> None:
    where = check.id
    for name in ("create", "verify", "cleanup"):
        if not isinstance(getattr(check, name), ToolStep):
            raise DeclarationError(f"{where}: {name} must be a ToolStep")
    guard = check.cleanup_guard
    if guard is not None and not isinstance(guard, ToolStep):
        raise DeclarationError(f"{where}: cleanup guard must be a ToolStep")
    _validate_pin(where, check.tenant_pin)
    steps = {"create": check.create, "verify": check.verify, "cleanup": check.cleanup}
    if guard is not None:
        steps["cleanup guard"] = guard
    for name, step in steps.items():
        # Only create may not read what it is about to create.
        references = (
            (RunScopedName,) if name == "create" else (RunScopedName, CreatedRef)
        )
        _validate_step(where, name, step, allowed=references)

    suffixes = {
        ref.suffix for ref in _references(check.create.arguments, RunScopedName)
    }
    if not suffixes:
        raise DeclarationError(
            f"{where}: create must name its object with a run-scoped name"
        )
    _require_write_area(where, "create", check.create, app_id)
    deleted_by_id = set(_references(check.cleanup.arguments, CreatedRef))
    deleted_by_name = {
        ref.suffix for ref in _references(check.cleanup.arguments, RunScopedName)
    }
    if not deleted_by_id and not deleted_by_name:
        raise DeclarationError(
            f"{where}: cleanup must name its object with a run-scoped name "
            "or a created-object reference"
        )
    if not deleted_by_name <= suffixes:
        raise DeclarationError(
            f"{where}: cleanup must name the object create named, not another "
            "run-scoped name"
        )
    pin_key = check.tenant_pin.expected.key
    _refuse_other_fixtures(where, "cleanup", check.cleanup, pin_key, app_id)
    _validate_guard(where, guard, deleted_by_id, suffixes)
    if guard is not None:
        _refuse_other_fixtures(where, "cleanup guard", guard, pin_key, app_id)

    write_tools = {check.create.tool, check.cleanup.tool}
    verdicts = check.expected_verdicts
    if not isinstance(verdicts, Mapping):
        raise DeclarationError(f"{where}: expected verdicts must be a mapping")
    if set(verdicts) != write_tools:
        raise DeclarationError(
            f"{where}: an expected verdict is needed for exactly {sorted(write_tools)}"
        )
    allowed = set(get_args(ExpectedVerdict))
    if any(verdict not in allowed for verdict in verdicts.values()):
        raise DeclarationError(f"{where}: unknown expected verdict")


def _validate_guard(
    where: str,
    guard: ToolStep | None,
    deleted_by_id: set[CreatedRef],
    suffixes: set[str],
) -> None:
    """A cleanup that deletes by ``CreatedRef`` first reads the same object
    and confirms the name create gave it; no other cleanup has a guard."""
    if not deleted_by_id:
        if guard is not None:
            raise DeclarationError(
                f"{where}: a cleanup guard is only for a cleanup that deletes "
                "by a created-object reference"
            )
        return
    if guard is None:
        raise DeclarationError(
            f"{where}: a cleanup that deletes by a created-object reference "
            "needs a cleanup guard"
        )
    if not deleted_by_id <= set(_references(guard.arguments, CreatedRef)):
        raise DeclarationError(
            f"{where}: the cleanup guard must read the object the cleanup deletes"
        )
    if _references(guard.arguments, RunScopedName):
        # Looked up by name, it would confirm the name whichever object the
        # cleanup's reference points at.
        raise DeclarationError(
            f"{where}: the cleanup guard must find the object by its "
            "created-object reference, not by a run-scoped name"
        )
    if not any(
        isinstance(item, Equals)
        and isinstance(item.value, RunScopedName)
        and item.value.suffix in suffixes
        for item in guard.assertions
    ):
        raise DeclarationError(
            f"{where}: the cleanup guard must assert the run-scoped name "
            "create gave the object"
        )


def _refuse_writes_in_read_steps(checks: ConnectorChecks) -> None:
    """A step that only reads (a read check, verify, a tenant pin, a cleanup
    guard, the sweep's ``find``) may not call a tool any of the connector's
    write checks creates or cleans up with; the sweep's ``delete`` is one of
    the latter."""
    write_tools = {
        tool
        for write in checks.writes
        for tool in (write.create.tool, write.cleanup.tool)
    }
    reads: list[tuple[str, str, ToolStep | ReadCheck]] = [
        (read.id, "a read check", read)
        for read in checks.reads
        if isinstance(read, ReadCheck)
    ]
    for write in checks.writes:
        reads.append((write.id, "verify", write.verify))
        reads.append((write.id, "the tenant pin", write.tenant_pin.step))
        if write.cleanup_guard is not None:
            reads.append((write.id, "the cleanup guard", write.cleanup_guard))
    if checks.sweep is not None:
        where = f"{checks.app_id} sweep"
        reads.append((where, "the tenant pin", checks.sweep.tenant_pin.step))
        reads.append((where, "find", checks.sweep.find))
    for where, name, step in reads:
        if step.tool in write_tools:
            raise DeclarationError(
                f"{where}: {name} only reads, so it may not call a create or "
                "cleanup tool of the connector's write checks"
            )


def _refuse_shared_names(checks: ConnectorChecks) -> None:
    """Two write checks of a connector never give their objects the same
    run-scoped name: in one run a cleanup by name, or a guard's name check,
    could then match the other check's object."""
    named: set[str] = set()
    for write in checks.writes:
        suffixes = {
            ref.suffix for ref in _references(write.create.arguments, RunScopedName)
        }
        if shared := sorted(suffixes & named):
            raise DeclarationError(
                f"{write.id}: another write check of {checks.app_id} already "
                f"names its object with the run-scoped suffix {shared[0]!r}"
            )
        named |= suffixes


def _validate_sweep(checks: ConnectorChecks) -> None:
    app_id = checks.app_id
    sweep = checks.sweep
    if sweep is None:
        if checks.writes:
            raise DeclarationError(
                f"{app_id}: a connector with a write check must declare a sweep"
            )
        return
    if not isinstance(sweep, Sweep):
        raise DeclarationError(
            f"{app_id}: sweep must be a Sweep, not {type(sweep).__name__}"
        )
    if not checks.writes:
        raise DeclarationError(f"{app_id}: a sweep needs a write check")
    where = f"{app_id} sweep"
    for name in ("find", "delete"):
        if not isinstance(getattr(sweep, name), ToolStep):
            raise DeclarationError(f"{where}: {name} must be a ToolStep")
    _validate_pin(where, sweep.tenant_pin)
    if any(
        write.tenant_pin.expected != sweep.tenant_pin.expected
        for write in checks.writes
    ):
        raise DeclarationError(
            f"{app_id}: every tenant pin of a connector must name the same fixture key"
        )
    _validate_step(where, "find", sweep.find, allowed=())
    _validate_step(where, "delete", sweep.delete, allowed=(ItemRef,))
    _require_write_area(where, "find", sweep.find, app_id)
    _validate_path(where, sweep.items)
    if not _is_reference_path(sweep.name):
        raise DeclarationError(
            f"{where}: an item's name path must be a non-empty tuple of keys "
            "and indexes"
        )
    if not _references(sweep.delete.arguments, ItemRef):
        raise DeclarationError(
            f"{where}: delete must take the item by a listed-item reference"
        )
    cleanup_tools = {write.cleanup.tool for write in checks.writes}
    if cleanup_tools != {sweep.delete.tool}:
        # A sweep lists one kind of object, so it can only find what one
        # cleanup tool would have deleted.
        raise DeclarationError(
            f"{where}: delete must use the one cleanup tool of the connector's "
            f"write checks, not {sweep.delete.tool!r} for {sorted(cleanup_tools)}"
        )
    _refuse_other_fixtures(
        where, "delete", sweep.delete, sweep.tenant_pin.expected.key, app_id
    )


def _validate_pin(where: str, pin: TenantPin) -> None:
    if not isinstance(pin, TenantPin):
        raise DeclarationError(f"{where}: tenant pin must be a TenantPin")
    if not isinstance(pin.step, ToolStep):
        raise DeclarationError(f"{where}: tenant pin step must be a ToolStep")
    if not isinstance(pin.expected, FixtureRef):
        raise DeclarationError(f"{where}: tenant pin expected must be a FixtureRef")
    step = pin.step
    _validate_call(where, step.tool, step.arguments, step.payload, step.assertions)
    _validate_path(where, pin.path)
    _refuse_references(where, "the tenant pin", step.arguments, step.assertions)


def _validate_step(
    where: str, name: str, step: ToolStep, *, allowed: tuple[type, ...]
) -> None:
    """One step of a write check or a sweep."""
    _validate_call(where, step.tool, step.arguments, step.payload, step.assertions)
    _require_assertion(where, name, step.assertions)
    _refuse_references(where, name, step.arguments, step.assertions, allowed=allowed)
    if any(isinstance(leaf, str) and "@" in leaf for leaf in _leaves(step.arguments)):
        raise DeclarationError(
            f"{where}: {name} has an '@' in its arguments; an address comes "
            "from a fixture in the test tenant"
        )


def _refuse_other_fixtures(
    where: str, name: str, step: ToolStep, pin_key: str, app_id: str
) -> None:
    """A step that deletes, or picks what is deleted, takes no fixture key
    but the tenant pin's and write-area keys, so no seeded object outside the
    write area is its target."""
    for ref in _references(step.arguments, FixtureRef):
        if ref.key != pin_key and not _is_write_area_key(ref.key, app_id):
            raise DeclarationError(
                f"{where}: {name} may take only the tenant pin's or a "
                f"write-area fixture key, not {ref.key!r}"
            )


def _is_write_area_key(key: str, app_id: str) -> bool:
    return key.startswith(f"{app_id}.{_WRITE_AREA_PREFIX}")


def _require_write_area(where: str, name: str, step: ToolStep, app_id: str) -> None:
    """``step`` targets the connector's write area: an argument is a fixture
    key ``<app_id>.write_...``.

    The tenant registry names the container it sets aside for write checks
    that way, in the namespace of the connector that writes there
    (``google-drive.write_folder_id``), which can differ from the tenant
    pin's (``google.tenant_id``).
    """
    if not any(
        _is_write_area_key(ref.key, app_id)
        for ref in _references(step.arguments, FixtureRef)
    ):
        prefix = f"{app_id}.{_WRITE_AREA_PREFIX}"
        raise DeclarationError(
            f"{where}: {name} must target the connector's write area, a fixture "
            f"key starting with {prefix!r}"
        )


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
    if value is None or isinstance(value, (str, bool, int, FixtureRef, *_REFERENCES)):
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


_T = TypeVar("_T")

_REFERENCE_LABELS: dict[type, str] = {
    RunScopedName: "a run-scoped name",
    CreatedRef: "a created-object reference",
    ItemRef: "a listed-item reference",
}
_REFERENCES = tuple(_REFERENCE_LABELS)


def _leaves(value: Any) -> Iterator[Any]:
    """Every value in ``value`` that is not a mapping or a sequence, mapping
    keys included, at any depth."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _references(value: Any, kind: type[_T]) -> list[_T]:
    """Every ``kind`` in ``value``, at any depth."""
    return [leaf for leaf in _leaves(value) if isinstance(leaf, kind)]


def _refuse_references(
    where: str,
    name: str,
    arguments: Mapping[str, Any],
    assertions: tuple[Assertion, ...],
    *,
    allowed: tuple[type, ...] = (),
) -> None:
    """``name`` uses no reference outside ``allowed``, in an argument or in
    an ``Equals`` value."""
    values = [
        arguments,
        *(item.value for item in assertions if isinstance(item, Equals)),
    ]
    for kind, label in _REFERENCE_LABELS.items():
        if kind not in allowed and _references(values, kind):
            raise DeclarationError(f"{where}: {name} may not use {label}")


def _validate_assertion(where: str, assertion: Assertion) -> None:
    if isinstance(assertion, StatusSuccess):
        return
    if isinstance(assertion, L3Envelope):
        _validate_keys(where, assertion.expect_keys, allow_empty=True)
        return
    _validate_path(where, assertion.path)
    if isinstance(assertion, (HasKeys, EachHasKeys)):
        _validate_keys(where, assertion.keys, allow_empty=False)
    elif isinstance(assertion, Contains):
        if not isinstance(assertion.value, str) or not assertion.value:
            raise DeclarationError(f"{where}: Contains needs a non-empty string")
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
            value, (str, bool, int, float, FixtureRef, RunScopedName, CreatedRef)
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
    # A string would pass as a path of one-letter keys.
    if not isinstance(path, tuple):
        raise DeclarationError(
            f"{where}: a path must be a tuple, not {type(path).__name__}"
        )
    for segment in path:
        if not _is_path_segment(segment):
            raise DeclarationError(f"{where}: invalid path segment {segment!r}")
