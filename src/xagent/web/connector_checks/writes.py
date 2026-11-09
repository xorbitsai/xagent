"""Running one smoke V2 write check (#3564).

A write check creates, verifies and deletes a real object in a test tenant,
so it runs only for a connector whose app id is listed in
``XAGENT_CONNECTOR_SMOKE_WRITE_APPS`` (``config.get_connector_smoke_write_apps``),
and only once its connector's session has passed. The engine prepares it
before any connector starts (``prepare_write``): it resolves the check's
fixtures and fills in its run-scoped names, ``xagent-smoke-<run_id>-<suffix>``,
and asks the identity adapter's write policy for each write tool's verdict.

``run_write`` then refuses the write, without calling anything, when

1. the identity has no write policy, or its verdict for the create or the
   cleanup tool is not the declared one (``verdict_mismatch``);
2. a fixture has no value (``tenant_mismatch``, the key as its reason);
3. the server does not list a tool the check calls (``tool_missing``);
4. the run's remaining time does not cover a call timeout for each step
   (``not_reached``/``time_budget``; later checks still run);

and otherwise reads the tenant pin (``tenant_mismatch``: ``pin_failed`` or
``mismatch``), creates, verifies, and cleans up. Cleanup runs whenever create
returned a result to clean up, a failed verify included; when it deletes by
a ``CreatedRef``, the cleanup guard first reads the object by that reference
and must confirm its name (``cleanup_failed``/``ownership``). The guard and
the cleanup run under their own call timeouts, not the run's deadline, and
finish even when the run is cancelled, which is then re-raised. A create
that raised, timed out or returned an error, or returned no usable id,
leaves whatever it made to the sweeper (``sweeper.py``), which removes it
from the write area's listing once it is more than two hours old.

A row reports the first of: a refusal, ``cleanup_failed``, a failed create,
a failed verify, ``pass``. A step's failure is ``<step>:<detail>``, the
detail being the reason a read check would report (``create:tool_error``,
``verify:equals``, ``cleanup:check_call``).

Nothing from a result, an exception or a fixture value reaches a row or a
log line.
"""

from __future__ import annotations

import asyncio
import logging
import math
import secrets
import time
from collections.abc import Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Protocol

from ..services.connector_probe import find_tool
from .assertions import MISSING, json_equal, judge_result, resolve_expected, value_at
from .declarations import (
    RUN_SCOPED_PREFIX,
    Assertion,
    CreatedRef,
    Equals,
    ExpectedVerdict,
    PayloadPath,
    PayloadSource,
    RunScopedName,
    ToolStep,
    WriteCheck,
)
from .fixtures import FixtureMissing, FixtureResolver, FixtureValue, resolve_fixtures
from .identity import ConnectorTarget, WritePolicy
from .results import CheckRow, FailureClass, Outcome

logger = logging.getLogger(__name__)

# Step names, which also open a failed step's reason.
_PIN = "pin"
_CREATE = "create"
_VERIFY = "verify"
_GUARD = "guard"
_CLEANUP = "cleanup"


class WritePolicyFailed(Exception):
    """The identity's write policy raised; carries nothing from the error."""


class RunClock(Protocol):
    """The run's deadline, as the engine keeps it."""

    def remaining(self) -> float: ...


def new_run_id(started_at: datetime) -> str:
    """``<yyyymmddhhmm>-<6 hex digits>``, in UTC: the sweeper reads an
    object's age from the time, and the random part keeps two runs apart,
    in any environment."""
    minute = started_at.astimezone(timezone.utc).strftime("%Y%m%d%H%M")
    return f"{minute}-{secrets.token_hex(3)}"


@dataclass(frozen=True)
class Step:
    """One tool call with its fixtures resolved and its names filled in; a
    ``CreatedRef`` stays until create has run. Values stay out of ``repr``."""

    tool: str
    arguments: Mapping[str, Any] = field(repr=False)
    payload: PayloadSource
    assertions: tuple[Assertion, ...] = field(repr=False)


@dataclass(frozen=True)
class PreparedWrite:
    """A write check ready to run, or the fixture key that has no value.

    ``verdicts`` is None when the identity has no write policy.
    """

    check: WriteCheck
    steps: Mapping[str, Step] = field(default_factory=dict, repr=False)
    tenant: FixtureValue | None = field(default=None, repr=False)
    verdicts: Mapping[str, ExpectedVerdict | None] | None = None
    missing_key: str | None = None


def prepare_writes(
    checks: Iterable[Any],
    target: ConnectorTarget,
    adapter: object,
    resolver: FixtureResolver,
    run_id: str,
) -> dict[str, PreparedWrite]:
    """Every write check in ``checks``, prepared to run on ``target``.

    The verdicts come from the adapter's ``identity.WritePolicy``; an adapter
    without one prepares every check with no verdicts.
    """
    policy = adapter if isinstance(adapter, WritePolicy) else None
    return {
        check.id: prepare_write(check, resolver, policy, target, run_id)
        for check in checks
        if isinstance(check, WriteCheck)
    }


def prepare_write(
    check: WriteCheck,
    resolver: FixtureResolver,
    policy: WritePolicy | None,
    target: ConnectorTarget,
    run_id: str,
) -> PreparedWrite:
    """``check`` with its fixtures resolved, its names filled in and its
    write tools' verdicts.

    A resolver error other than ``FixtureMissing`` propagates, as for a read
    check; a write policy that raises is ``WritePolicyFailed``.
    """
    verdicts: dict[str, ExpectedVerdict | None] | None = None
    if policy is not None:
        verdicts = {}
        for tool in (check.create.tool, check.cleanup.tool):
            try:
                verdicts[tool] = policy.write_policy_verdict(target, tool)
            except Exception as error:
                logger.warning(
                    "The write policy for %s raised %s", check.id, type(error).__name__
                )
                raise WritePolicyFailed() from None
    declared = {
        _PIN: check.tenant_pin.step,
        _CREATE: check.create,
        _VERIFY: check.verify,
        _CLEANUP: check.cleanup,
    }
    if check.cleanup_guard is not None:
        declared[_GUARD] = check.cleanup_guard
    try:
        tenant = resolve_fixtures(check.tenant_pin.expected, resolver)
        steps = {
            name: _prepare_step(step, resolver, run_id)
            for name, step in declared.items()
        }
    except FixtureMissing as missing:
        return PreparedWrite(check, verdicts=verdicts, missing_key=missing.key)
    return PreparedWrite(check, steps, tenant, verdicts)


def unavailable_write_row(connector: str, check: WriteCheck) -> CheckRow:
    """The row of a switched-on write check whose session did not pass."""
    return CheckRow(
        connector,
        check.id,
        "write",
        check.create.tool,
        Outcome.SKIP,
        FailureClass.SESSION_UNAVAILABLE,
    )


def _prepare_step(step: ToolStep, resolver: FixtureResolver, run_id: str) -> Step:
    def name(value: Any) -> Any:
        if isinstance(value, RunScopedName):
            return f"{RUN_SCOPED_PREFIX}{run_id}-{value.suffix}"
        return value

    arguments = map_leaves(resolve_fixtures(dict(step.arguments), resolver), name)
    assertions = tuple(
        replace(item, value=name(item.value)) if isinstance(item, Equals) else item
        for item in resolve_expected(step.assertions, resolver)
    )
    return Step(step.tool, arguments, step.payload, assertions)


def map_leaves(value: Any, leaf: Callable[[Any], Any]) -> Any:
    if isinstance(value, Mapping):
        return {key: map_leaves(item, leaf) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [map_leaves(item, leaf) for item in value]
    return leaf(value)


@dataclass(frozen=True)
class Called:
    raw: Any = None
    error: BaseException | None = None
    timed_out: bool = False


# A failed step: its failure class and the detail its reason ends with.
Fault = tuple[FailureClass, str]


async def run_write(
    connector: str,
    prepared: PreparedWrite,
    tools: Any,
    *,
    clock: RunClock,
    check_timeout: float,
) -> CheckRow:
    """Run ``prepared`` on the session's listed ``tools`` and return its row."""
    check = prepared.check
    started = time.monotonic()

    def row(
        outcome: Outcome,
        failure_class: FailureClass | None = None,
        reason: str | None = None,
    ) -> CheckRow:
        elapsed = max(0, int((time.monotonic() - started) * 1000))
        return CheckRow(
            connector,
            check.id,
            "write",
            check.create.tool,
            outcome,
            failure_class,
            reason,
            elapsed,
        )

    refusal = _refusal(prepared)
    if refusal is not None:
        return row(Outcome.FAIL, *refusal)
    steps = prepared.steps
    found = {name: find_tool(tools, step.tool) for name, step in steps.items()}
    for name, tool in found.items():
        if tool is None:
            return row(Outcome.FAIL, FailureClass.TOOL_MISSING, name)
    if clock.remaining() < check_timeout * len(steps):
        # Started now, the write could not be sure to reach its cleanup. The
        # refusal is the write's alone: later checks still run.
        return row(Outcome.FAIL, FailureClass.NOT_REACHED, "time_budget")

    mismatch = await pin_tenant(
        found[_PIN],
        steps[_PIN],
        check.tenant_pin.path,
        prepared.tenant,
        check_timeout,
        check.id,
    )
    if mismatch is not None:
        return row(Outcome.FAIL, FailureClass.TENANT_MISMATCH, mismatch)

    created: Any = None
    create_fault: Fault | None = None
    verify_fault: Fault | None = None
    cancelled = False
    try:
        called = await call_step(found[_CREATE], steps[_CREATE], check_timeout)
        create_fault = step_fault(called, steps[_CREATE], check.id, _CREATE)
        if create_fault is None or create_fault[0] is FailureClass.ASSERTION_FAILED:
            # A result to clean up: the object may exist even when an
            # assertion on it failed.
            created = called.raw
        raise_if_cancelled()
        resolved = _resolve_created(steps, created)
        if created is not None and create_fault is None and _VERIFY not in resolved:
            create_fault = (FailureClass.ASSERTION_FAILED, "created_ref_missing")
        if create_fault is None:
            verified = await call_step(found[_VERIFY], resolved[_VERIFY], check_timeout)
            verify_fault = step_fault(verified, resolved[_VERIFY], check.id, _VERIFY)
            raise_if_cancelled()
    except asyncio.CancelledError:
        cancelled = True

    cleanup_fault: Fault | None = None
    if created is not None:
        resolved = _resolve_created(steps, created)
        if _CLEANUP not in resolved or (_GUARD in steps and _GUARD not in resolved):
            # No usable id to delete by: the sweeper's.
            if create_fault is None:
                create_fault = (FailureClass.ASSERTION_FAILED, "created_ref_missing")
        else:
            cleanup_fault = await _run_through_cancellation(
                _clean(found, resolved, check.id, check_timeout)
            )
    if cancelled:
        # No row will carry the outcome: the run is being cancelled. A failed
        # cleanup, or a create whose object is left to the sweeper, is logged.
        dropped = cleanup_fault
        if dropped is None and create_fault is not None:
            dropped = (create_fault[0], f"{_CREATE}:{create_fault[1]}")
        if dropped is not None:
            logger.warning(
                "Write check %s was cancelled and would have reported %s/%s",
                check.id,
                dropped[0].value,
                dropped[1],
            )
        raise asyncio.CancelledError
    if cleanup_fault is not None:
        return row(Outcome.FAIL, *cleanup_fault)
    for name, fault in ((_CREATE, create_fault), (_VERIFY, verify_fault)):
        if fault is not None:
            return row(Outcome.FAIL, fault[0], f"{name}:{fault[1]}")
    return row(Outcome.PASS)


async def pin_tenant(
    tool: Any,
    step: Step,
    path: PayloadPath,
    expected: Any,
    timeout: float,
    where: str,
) -> str | None:
    """Read the connection's tenant id at ``path`` with ``step``: the
    ``tenant_mismatch`` reason (``pin_failed``, ``mismatch``), or None when
    it is ``expected``. The sweeper pins its tenant the same way."""
    pinned = await call_step(tool, step, timeout)
    raise_if_cancelled()
    if step_fault(pinned, step, where, _PIN) is not None:
        return "pin_failed"
    tenant = value_at(pinned.raw, step.payload, path)
    if tenant is MISSING or not json_equal(tenant, expected):
        return "mismatch"
    return None


def _refusal(prepared: PreparedWrite) -> tuple[FailureClass, str] | None:
    """Why the write may not start, before any call."""
    check = prepared.check
    if prepared.verdicts is None:
        return FailureClass.VERDICT_MISMATCH, "no_write_policy"
    for name, tool in ((_CREATE, check.create.tool), (_CLEANUP, check.cleanup.tool)):
        answer = prepared.verdicts.get(tool)
        if answer is None:
            return FailureClass.VERDICT_MISMATCH, "no_write_policy"
        if answer != check.expected_verdicts[tool]:
            return FailureClass.VERDICT_MISMATCH, name
    if prepared.missing_key is not None:
        return FailureClass.TENANT_MISMATCH, prepared.missing_key
    return None


def _resolve_created(steps: Mapping[str, Step], created: Any) -> dict[str, Step]:
    """Every step after create whose ``CreatedRef``s the create result holds,
    with them filled in; a step it does not hold is left out."""
    create = steps[_CREATE]
    resolved: dict[str, Step] = {}
    for name in (_VERIFY, _GUARD, _CLEANUP):
        step = steps.get(name)
        if step is None:
            continue
        missing = False

        def fill(value: Any) -> Any:
            nonlocal missing
            if not isinstance(value, CreatedRef):
                return value
            found = value_at(created, create.payload, value.path)
            if found is MISSING or not is_id(found):
                missing = True
                return value
            return found

        arguments = map_leaves(step.arguments, fill)
        assertions = tuple(
            replace(item, value=fill(item.value)) if isinstance(item, Equals) else item
            for item in step.assertions
        )
        if not missing:
            resolved[name] = Step(step.tool, arguments, step.payload, assertions)
    return resolved


def is_id(value: Any) -> bool:
    """An object id: a non-empty string or a finite number, never a bool."""
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, int)


async def _clean(
    found: Mapping[str, Any],
    resolved: Mapping[str, Step],
    check_id: str,
    check_timeout: float,
) -> Fault | None:
    """The guard, if any, then the cleanup; the failure, or None."""
    guard = resolved.get(_GUARD)
    if guard is not None:
        guarded = await call_step(found[_GUARD], guard, check_timeout)
        fault = step_fault(guarded, guard, check_id, _GUARD)
        if fault is not None:
            # The object it read is not shown to be this run's: keep it.
            if fault[0] is FailureClass.ASSERTION_FAILED:
                return FailureClass.CLEANUP_FAILED, "ownership"
            return FailureClass.CLEANUP_FAILED, f"{_GUARD}:{fault[1]}"
    cleaned = await call_step(found[_CLEANUP], resolved[_CLEANUP], check_timeout)
    fault = step_fault(cleaned, resolved[_CLEANUP], check_id, _CLEANUP)
    if fault is not None:
        return FailureClass.CLEANUP_FAILED, f"{_CLEANUP}:{fault[1]}"
    return None


async def _run_through_cancellation(
    cleanup: Coroutine[Any, Any, Fault | None],
) -> Fault | None:
    """Run ``cleanup`` to its end, through any cancellation of the run.

    Its calls run under their own timeouts. A cancellation that arrives
    meanwhile is re-raised once it is done.
    """
    task: asyncio.Future[Fault | None] = asyncio.ensure_future(cleanup)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    outcome = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return outcome


async def call_step(tool: Any, step: Step, timeout: float) -> Called:
    bound = asyncio.timeout(timeout)
    try:
        async with bound:
            raw = await tool.run_json_async(dict(step.arguments))
    except Exception as error:
        # A tool that turned its bound's cancellation into another error timed
        # out all the same, as a read check's call does.
        timed_out = isinstance(error, TimeoutError) or bound.expired()
        return Called(error=error, timed_out=timed_out)
    # A tool that swallowed its bound's cancellation and returned.
    return Called(raw=raw, timed_out=bound.expired())


def step_fault(called: Called, step: Step, where: str, name: str) -> Fault | None:
    if called.timed_out:
        return FailureClass.TIMED_OUT, "check_call"
    if called.error is not None:
        logger.warning(
            "%s: %s (%s) raised %s",
            where,
            name,
            step.tool,
            type(called.error).__name__,
        )
        return FailureClass.CALL_FAILED, "exception"
    failure_class, reason = judge_result(called.raw, step.payload, step.assertions)
    if failure_class is None:
        return None
    return failure_class, reason or failure_class.value


def raise_if_cancelled() -> None:
    # A cancellation a tool swallowed is still a cancellation of the run.
    task = asyncio.current_task()
    if task is not None and task.cancelling() > 0:
        raise asyncio.CancelledError
