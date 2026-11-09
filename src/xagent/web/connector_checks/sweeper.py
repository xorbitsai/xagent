"""The smoke V2 sweeper (#3564): removes what write checks left behind.

A write check names every object it creates
``xagent-smoke-<yyyymmddhhmm>-<6 hex digits>-<suffix>`` (``writes.py``) and
deletes it again, but a create that failed, a cleanup that failed and a run
that was killed leave the object in the tenant. ``run_sweep`` removes it
later. It sweeps each connector that declares a ``Sweep`` and whose writes
are switched on (``XAGENT_CONNECTOR_SMOKE_WRITE_APPS``). A connector whose
writes are off is ``skip``/``writes_disabled``; one without a sweep has no
row.

Before any connector starts, it asks the identity's write policy for the
sweep's delete tool, which is the write checks' cleanup tool, and resolves
the sweep's fixtures. Then, for each connector, it refuses without calling
anything when the policy's verdict is not the one the write checks declare
for that tool (``verdict_mismatch``) or a fixture has no value
(``tenant_mismatch``, the key). Otherwise it opens one session, finds the
pin, find and delete tools (``tool_missing``), pins the tenant
(``tenant_mismatch``: ``pin_failed`` or ``mismatch``) and lists the write
area with ``find``: its first page only, so further leftovers wait for the
next sweep.

An item whose name starts with ``xagent-smoke-`` is deleted by its
``ItemRef`` when the minute in its name is more than ``SWEEP_AFTER`` (two
hours) ago. The minute is when the run started, and a run with the engine's
default total timeout lasts 15 minutes, so a run in progress in any
environment keeps its objects. A younger item, or one whose minute is in
the future, is kept, and a prefixed name whose minute does not parse is
counted and never deleted. An item listed twice is swept once. One sweep
deletes at most ``MAX_DELETES`` items of a connector
(``cleanup_failed``/``cap_reached``), and leaves the rest for the next.
After a failed delete, ``find`` lists the area once more: an item missing
from a listing that shows at least one id counts as deleted, as when another
environment's sweeper took it first. A delete that still failed is
``cleanup_failed``/``delete:<detail>``.

The pin, ``find``, each delete and the second listing start only when a
call timeout of the sweep's total timeout is left; otherwise the connector is
``not_reached``/``time_budget``, what was due stays for the next sweep, and
the sweep is not ``complete``.

The result is its own document, ``xagent.connector_smoke.sweep``
(``sweep.schema.json``): per connector, counts of the prefixed items found,
deleted, failed, kept and unparsed. It carries no name, id or value.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import resources
from typing import Any

from ...config import get_connector_smoke_write_apps
from ..services.connector_probe import find_tool, probe_call_timeout_seconds
from .assertions import at_path, resolve_expected, value_at
from .codes import APP_ID_PATTERN, is_code
from .declarations import (
    DECLARATION_FORMAT_VERSION,
    RUN_SCOPED_PREFIX,
    ConnectorChecks,
    ExpectedVerdict,
    ItemRef,
    Sweep,
    ToolStep,
    validate_connector_checks,
)
from .fixtures import FixtureMissing, resolve_fixtures
from .identity import (
    ConnectorTarget,
    IdentityAdapter,
    IdentitySetupError,
    SessionPurpose,
    WritePolicy,
)
from .registry import connector_checks_for
from .results import (
    ENVIRONMENT_PATTERN,
    FAIL_CLASSES,
    IDENTITY_PATTERN,
    SKIP_CLASSES,
    FailureClass,
    Outcome,
    classify_session,
)
from .writes import (
    Step,
    WritePolicyFailed,
    call_step,
    is_id,
    map_leaves,
    pin_tenant,
    raise_if_cancelled,
    step_fault,
)

logger = logging.getLogger(__name__)

SWEEP_SCHEMA_ID = "xagent.connector_smoke.sweep"
# Bumped on any change a consumer of the document must adapt to.
SWEEP_SCHEMA_VERSION = 1
# How old a leftover must be, by the minute in its name, to be deleted.
SWEEP_AFTER = timedelta(hours=2)
# The most items one sweep deletes for one connector.
MAX_DELETES = 50
# The sweep's total bound when the caller gives none.
DEFAULT_SWEEP_TIMEOUT_SECONDS = 600.0

_NAMED = re.compile(
    re.escape(RUN_SCOPED_PREFIX) + r"(?P<minute>[0-9]{12})-[0-9a-f]{6}-"
)
_PIN = "pin"
_FIND = "find"
_DELETE = "delete"

DeclarationLookup = Callable[[str], ConnectorChecks | None]


@dataclass(frozen=True)
class SweepRow:
    """One connector's sweep: its outcome and counts of the items whose name
    carries the run-scoped prefix."""

    connector: str
    outcome: Outcome
    failure_class: FailureClass | None = None
    reason: str | None = None
    found: int = 0
    deleted: int = 0
    failed: int = 0
    kept: int = 0
    unparsed: int = 0
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if not APP_ID_PATTERN.fullmatch(self.connector):
            raise ValueError(f"invalid connector: {self.connector!r}")
        if self.reason is not None and not is_code(self.reason):
            raise ValueError("a reason must be a code")
        valid = {
            Outcome.PASS: self.failure_class is None,
            Outcome.FAIL: self.failure_class in FAIL_CLASSES,
            Outcome.SKIP: self.failure_class in SKIP_CLASSES,
        }.get(self.outcome, False)
        if not valid:
            raise ValueError(f"{self.outcome} cannot carry {self.failure_class}")
        for name in ("found", "deleted", "failed", "kept", "unparsed", "duration_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_json(self) -> dict[str, Any]:
        return {
            "connector": self.connector,
            "outcome": self.outcome.value,
            "failure_class": (
                self.failure_class.value if self.failure_class is not None else None
            ),
            "reason": self.reason,
            "found": self.found,
            "deleted": self.deleted,
            "failed": self.failed,
            "kept": self.kept,
            "unparsed": self.unparsed,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class SweepResult:
    """One sweep under one identity. ``failure`` is a code for a sweep that
    could not be set up; such a sweep has no rows and is never complete."""

    identity: str
    started_at: datetime
    finished_at: datetime
    complete: bool
    rows: tuple[SweepRow, ...]
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
                raise ValueError("sweep times must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("a sweep cannot finish before it starts")
        if self.failure is not None:
            if not is_code(self.failure):
                raise ValueError("a sweep failure must be a code")
            if self.complete or self.rows:
                raise ValueError(
                    "a sweep that failed to set up is incomplete and empty"
                )
        connectors = [row.connector for row in self.rows]
        if len(set(connectors)) != len(connectors):
            raise ValueError("two rows for one connector")

    def to_document(self) -> dict[str, Any]:
        return {
            "schema": SWEEP_SCHEMA_ID,
            "schema_version": SWEEP_SCHEMA_VERSION,
            "declaration_format_version": DECLARATION_FORMAT_VERSION,
            "identity": self.identity,
            "environment": self.environment,
            "started_at": _timestamp(self.started_at),
            "finished_at": _timestamp(self.finished_at),
            "complete": self.complete,
            "failure": self.failure,
            "older_than_seconds": int(SWEEP_AFTER.total_seconds()),
            "max_deletes": MAX_DELETES,
            "rows": [row.to_json() for row in self.rows],
        }


def load_sweep_schema() -> dict[str, Any]:
    """The JSON Schema every sweep document validates against."""
    text = (
        resources.files(__package__)
        .joinpath("sweep.schema.json")
        .read_text(encoding="utf-8")
    )
    schema: dict[str, Any] = json.loads(text)
    return schema


@dataclass(frozen=True)
class _Plan:
    """One connector to sweep, prepared before any connector starts."""

    target: ConnectorTarget
    sweep: Sweep
    enabled: bool
    refusal: tuple[FailureClass, str] | None = None
    steps: Mapping[str, Step] = field(default_factory=dict, repr=False)
    tenant: Any = field(default=None, repr=False)

    @property
    def connector(self) -> str:
        return self.target.connector


@dataclass(frozen=True)
class _Counts:
    """A connector's items whose name carries the run-scoped prefix."""

    found: int = 0
    deleted: int = 0
    failed: int = 0
    kept: int = 0
    unparsed: int = 0


class _ResolverFailed(Exception):
    """The adapter's fixture resolver raised; carries nothing from the error."""


class _Deadline:
    def __init__(self, total_timeout: float) -> None:
        self._started = time.monotonic()
        self._deadline = self._started + total_timeout
        self.cut = False

    def elapsed(self) -> timedelta:
        return timedelta(seconds=time.monotonic() - self._started)

    def remaining(self) -> float:
        return self._deadline - time.monotonic()


async def run_sweep(
    adapter: IdentityAdapter,
    *,
    environment: str | None = None,
    connectors: Collection[str] | None = None,
    call_timeout: float | None = None,
    total_timeout: float = DEFAULT_SWEEP_TIMEOUT_SECONDS,
    declarations: DeclarationLookup = connector_checks_for,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> SweepResult:
    """Sweep ``adapter``'s connectors and return the sweep result.

    ``connectors`` selects connectors by their result label; a call's
    timeout defaults to the L3 probe call's. Raises before any connector
    starts: ``ValueError`` for an invalid label or timeout, or a declaration
    that breaks the format; ``TypeError`` for a selector given as one string
    or a fixture value that is not a JSON scalar. A fixture resolver that
    raises fails the sweep as ``fixtures:exception``, and a write policy that
    raises as ``write_policy:exception``.
    """
    started_at = now()
    for timeout in (total_timeout, call_timeout):
        if timeout is not None and (
            isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("a timeout must be a finite number of seconds above 0")
    if isinstance(connectors, str):
        raise TypeError("a selector is a collection of names, not one string")
    if not IDENTITY_PATTERN.fullmatch(adapter.label):
        raise ValueError("invalid identity label")
    if environment is not None and not ENVIRONMENT_PATTERN.fullmatch(environment):
        raise ValueError("invalid environment label")
    timeout = float(call_timeout or probe_call_timeout_seconds())
    deadline = _Deadline(total_timeout)

    def finish(rows: Sequence[SweepRow], failure: str | None) -> SweepResult:
        return SweepResult(
            identity=adapter.label,
            environment=environment,
            started_at=started_at,
            finished_at=started_at + deadline.elapsed(),
            complete=failure is None and not deadline.cut,
            rows=tuple(rows),
            failure=failure,
        )

    try:
        targets = list(adapter.enumerate_targets())
    except IdentitySetupError as error:
        return finish((), error.code)
    except Exception as error:
        logger.warning("Enumerating connectors raised %s", type(error).__name__)
        return finish((), "enumeration:exception")
    if not targets:
        return finish((), "enumeration:no_connectors")
    if len({target.connector for target in targets}) != len(targets):
        return finish((), "enumeration:duplicate_connector")
    try:
        plans = _plan(adapter, targets, declarations, connectors)
    except _ResolverFailed:
        return finish((), "fixtures:exception")
    except WritePolicyFailed:
        return finish((), "write_policy:exception")
    if connectors is not None and not plans:
        return finish((), "selection:no_targets")

    rows = [
        await _sweep_connector(adapter, plan, deadline, timeout, now())
        for plan in plans
    ]
    return finish(rows, None)


def _plan(
    adapter: IdentityAdapter,
    targets: Sequence[ConnectorTarget],
    declarations: DeclarationLookup,
    connectors: Collection[str] | None,
) -> list[_Plan]:
    write_apps = get_connector_smoke_write_apps()
    selected: list[tuple[ConnectorTarget, ConnectorChecks, Sweep]] = []
    for target in targets:
        if connectors is not None and target.connector not in connectors:
            continue
        declaration = declarations(target.app_id)
        if declaration is None or declaration.sweep is None:
            continue
        if declaration.app_id != target.app_id:
            raise ValueError(
                f"the declaration for {target.connector} names another app"
            )
        validate_connector_checks(declaration)
        selected.append((target, declaration, declaration.sweep))
    # Prepared only once every declaration is valid, as the engine does.
    plans: list[_Plan] = []
    for target, declaration, sweep in selected:
        enabled = target.app_id in write_apps
        if not enabled or target.skip_reason is not None:
            plans.append(_Plan(target, sweep, enabled))
            continue
        plans.append(_prepare(adapter, target, declaration, sweep))
    return plans


def _prepare(
    adapter: IdentityAdapter,
    target: ConnectorTarget,
    declaration: ConnectorChecks,
    sweep: Sweep,
) -> _Plan:
    """The connector's refusal, or its steps with fixtures resolved."""
    tool = sweep.delete.tool
    declared = {
        write.expected_verdicts[tool]
        for write in declaration.writes
        if write.cleanup.tool == tool
    }
    refusal: tuple[FailureClass, str] | None = None
    if not isinstance(adapter, WritePolicy):
        refusal = FailureClass.VERDICT_MISMATCH, "no_write_policy"
    else:
        try:
            answer: ExpectedVerdict | None = adapter.write_policy_verdict(target, tool)
        except Exception as error:
            logger.warning(
                "The write policy for %s raised %s",
                target.connector,
                type(error).__name__,
            )
            raise WritePolicyFailed() from None
        if answer is None:
            refusal = FailureClass.VERDICT_MISMATCH, "no_write_policy"
        elif declared != {answer}:
            refusal = FailureClass.VERDICT_MISMATCH, _DELETE
    if refusal is not None:
        return _Plan(target, sweep, True, refusal)

    resolver = _SafeResolver(adapter)
    try:
        tenant = resolve_fixtures(sweep.tenant_pin.expected, resolver)
        steps = {
            _PIN: _step(sweep.tenant_pin.step, resolver),
            _FIND: _step(sweep.find, resolver),
            _DELETE: _step(sweep.delete, resolver),
        }
    except FixtureMissing as missing:
        return _Plan(target, sweep, True, (FailureClass.TENANT_MISMATCH, missing.key))
    return _Plan(target, sweep, True, None, steps, tenant)


class _SafeResolver:
    """The adapter's resolver; a raise other than ``FixtureMissing`` is
    ``_ResolverFailed``, which carries nothing from the error."""

    def __init__(self, adapter: IdentityAdapter) -> None:
        self._adapter = adapter

    def resolve_fixture(self, key: str) -> Any:
        try:
            return self._adapter.resolve_fixture(key)
        except FixtureMissing:
            raise FixtureMissing(key) from None
        except Exception as error:
            logger.warning("Resolving fixture %s raised %s", key, type(error).__name__)
            raise _ResolverFailed() from None


def _step(step: ToolStep, resolver: _SafeResolver) -> Step:
    return Step(
        step.tool,
        resolve_fixtures(dict(step.arguments), resolver),
        step.payload,
        resolve_expected(step.assertions, resolver),
    )


async def _sweep_connector(
    adapter: IdentityAdapter,
    plan: _Plan,
    deadline: _Deadline,
    timeout: float,
    now: datetime,
) -> SweepRow:
    started = time.monotonic()
    where = f"{plan.connector} sweep"

    def row(
        outcome: Outcome,
        failure_class: FailureClass | None = None,
        reason: str | None = None,
        counts: _Counts = _Counts(),
    ) -> SweepRow:
        elapsed = max(0, int((time.monotonic() - started) * 1000))
        return SweepRow(
            plan.connector,
            outcome,
            failure_class,
            reason,
            counts.found,
            counts.deleted,
            counts.failed,
            counts.kept,
            counts.unparsed,
            elapsed,
        )

    if not plan.enabled:
        return row(Outcome.SKIP, FailureClass.WRITES_DISABLED)
    if plan.target.skip_reason is not None:
        return row(
            Outcome.SKIP, FailureClass.SKIPPED_BY_IDENTITY, plan.target.skip_reason
        )
    if plan.refusal is not None:
        return row(Outcome.FAIL, *plan.refusal)

    def out_of_time() -> bool:
        if deadline.remaining() < timeout:
            deadline.cut = True
            return True
        return False

    if out_of_time():
        return row(Outcome.FAIL, FailureClass.NOT_REACHED, "time_budget")

    bound = asyncio.timeout(deadline.remaining())
    listing = None
    try:
        async with bound:
            listing = await adapter.open_session(
                plan.target, purpose=SessionPurpose.CALL_TOOLS
            )
    except Exception as error:
        logger.warning(
            "Opening a session for %s raised %s", plan.connector, type(error).__name__
        )
    raise_if_cancelled()
    if bound.expired():
        deadline.cut = True
        return row(Outcome.FAIL, FailureClass.TIMED_OUT, "run_deadline")
    if listing is None:
        return row(Outcome.FAIL, FailureClass.FAILED_TO_START, "exception")
    outcome, failure_class, reason = classify_session(listing)
    if outcome is not Outcome.PASS:
        return row(outcome, failure_class, reason)
    tools = listing.load_result.tools if listing.load_result is not None else ()

    steps = plan.steps
    found = {name: find_tool(tools, step.tool) for name, step in steps.items()}
    for name, tool in found.items():
        if tool is None:
            return row(Outcome.FAIL, FailureClass.TOOL_MISSING, name)
    if out_of_time():
        return row(Outcome.FAIL, FailureClass.NOT_REACHED, "time_budget")
    mismatch = await pin_tenant(
        found[_PIN],
        steps[_PIN],
        plan.sweep.tenant_pin.path,
        plan.tenant,
        timeout,
        where,
    )
    if mismatch is not None:
        return row(Outcome.FAIL, FailureClass.TENANT_MISMATCH, mismatch)

    if out_of_time():
        return row(Outcome.FAIL, FailureClass.NOT_REACHED, "time_budget")
    listed = await call_step(found[_FIND], steps[_FIND], timeout)
    raise_if_cancelled()
    fault = step_fault(listed, steps[_FIND], where, _FIND)
    if fault is not None:
        return row(Outcome.FAIL, fault[0], f"{_FIND}:{fault[1]}")
    items = value_at(listed.raw, steps[_FIND].payload, plan.sweep.items)
    if not isinstance(items, list):
        return row(Outcome.FAIL, FailureClass.ASSERTION_FAILED, f"{_FIND}:items")

    prefixed = kept = unparsed = 0
    due: list[Any] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        name = at_path(item, plan.sweep.name)
        if not isinstance(name, str) or not name.startswith(RUN_SCOPED_PREFIX):
            continue
        item_id = _item_id(steps[_DELETE], item)
        if item_id is not None:
            if item_id in seen:
                continue
            seen.add(item_id)
        prefixed += 1
        minute = _minute(name)
        if minute is None:
            unparsed += 1
        elif now - minute <= SWEEP_AFTER:
            kept += 1
        else:
            due.append(item)
    capped = len(due) > MAX_DELETES
    if capped:
        kept += len(due) - MAX_DELETES
        due = due[:MAX_DELETES]

    deleted = 0
    failures: list[tuple[Any, str]] = []
    stopped = False
    for index, item in enumerate(due):
        if out_of_time():
            stopped = True
            kept += len(due) - index
            break
        step = _for_item(steps[_DELETE], item)
        if step is None:
            failures.append((item, "item_id_missing"))
            continue
        called = await call_step(found[_DELETE], step, timeout)
        raise_if_cancelled()
        fault = step_fault(called, step, where, _DELETE)
        if fault is None:
            deleted += 1
        else:
            failures.append((item, fault[1]))

    # A delete that failed because the item is already gone is done. With
    # no time left to list again, the failures stand and the sweep is cut.
    if failures and not out_of_time():
        again = await call_step(found[_FIND], steps[_FIND], timeout)
        raise_if_cancelled()
        listed_again = (
            value_at(again.raw, steps[_FIND].payload, plan.sweep.items)
            if step_fault(again, steps[_FIND], where, _FIND) is None
            else None
        )
        present = (
            {_item_id(steps[_DELETE], item) for item in listed_again} - {None}
            if isinstance(listed_again, list)
            else set()
        )
        # A listing with no readable id proves nothing; nor does an empty
        # one, since search may lag behind a delete.
        if present:
            still = [
                (item, detail)
                for item, detail in failures
                if (item_id := _item_id(steps[_DELETE], item)) is None
                or item_id in present
            ]
            deleted += len(failures) - len(still)
            failures = still

    counts = _Counts(prefixed, deleted, len(failures), kept, unparsed)
    if failures:
        detail = failures[0][1]
        return row(
            Outcome.FAIL,
            FailureClass.CLEANUP_FAILED,
            f"{_DELETE}:{detail}",
            counts=counts,
        )
    if stopped:
        return row(Outcome.FAIL, FailureClass.NOT_REACHED, "time_budget", counts=counts)
    if capped:
        return row(
            Outcome.FAIL, FailureClass.CLEANUP_FAILED, "cap_reached", counts=counts
        )
    return row(Outcome.PASS, counts=counts)


def _minute(name: str) -> datetime | None:
    """The UTC minute in a run-scoped name, or None when it has none."""
    match = _NAMED.match(name)
    if match is None:
        return None
    try:
        minute = datetime.strptime(match["minute"], "%Y%m%d%H%M")
    except ValueError:
        return None
    return minute.replace(tzinfo=timezone.utc)


def _for_item(step: Step, item: Any) -> Step | None:
    """``step`` with its ``ItemRef``s read from ``item``, or None when one
    finds no id."""
    missing = False

    def fill(value: Any) -> Any:
        nonlocal missing
        if not isinstance(value, ItemRef):
            return value
        found = at_path(item, value.path)
        if not is_id(found):
            missing = True
            return value
        return found

    arguments = map_leaves(step.arguments, fill)
    if missing:
        return None
    return Step(step.tool, arguments, step.payload, step.assertions)


def _item_id(step: Step, item: Any) -> tuple[str, ...] | None:
    """What ``step``'s ``ItemRef``s read from ``item``, as text: the item's
    identity when it is listed again, so ``17`` and ``"17"`` are one item;
    None when it has none."""
    found: list[Any] = []

    def collect(value: Any) -> Any:
        if isinstance(value, ItemRef):
            found.append(at_path(item, value.path))
        return value

    map_leaves(step.arguments, collect)
    if not found or not all(is_id(value) for value in found):
        return None
    return tuple(
        str(int(value))
        if isinstance(value, float) and value.is_integer()
        else str(value)
        for value in found
    )


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
