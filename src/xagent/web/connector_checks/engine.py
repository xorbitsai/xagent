"""The smoke V2 engine: run every declared check of an identity's connectors.

``run_checks`` is a library call. It takes an identity adapter
(``identity.IdentityAdapter``) and returns one ``results.RunResult``; the
caller (a ``python -m`` CLI, or Xagent SaaS's Toby module after it installs
its hooks) turns it into the JSON document.

Before any connector starts, the engine enumerates the targets, validates
every selected declaration (``registry.connector_checks_for`` by default) and
resolves, through the adapter, the fixtures of every selected read check on
a target the adapter does not skip. A check whose key has no value is
``skip``/``fixture_missing`` and is never called. A resolver that raises fails
the run as ``fixtures:exception``: a broken fixture store is a failure of the
run, never a skipped check. A write check of a connector whose writes are
switched on (``XAGENT_CONNECTOR_SMOKE_WRITE_APPS``) is prepared the same way
(``writes.prepare_writes``), with the run's id in its names and its write
tools' verdicts from the adapter's write policy; a policy that raises fails
the run as ``write_policy:exception``.
Then, for each target in the adapter's order:

1. A target the adapter skips is never started.
2. One session: one config build and one tool listing
   (``adapter.open_session``), which is the connector's ``session`` row. It
   opens as ``SessionPurpose.LIST_ONLY`` when the app has no declaration or
   every check it declares is unprobeable, otherwise as ``CALL_TOOLS``. The
   purpose follows the whole declaration, so neither a selector, a fixture
   nor a write switch changes how a connector starts.
3. The checks run in declaration order on the listed tools, each call under
   its own timeout. Each call still opens its own MCP session, so an in-repo
   server spawns one process per check. A write check runs through
   ``writes.run_write`` when its connector's writes are switched on, and is
   ``skip``/``writes_disabled`` otherwise.

When a session does not pass, none of its checks runs: a read check, and a
switched-on write check, is ``skip``/``session_unavailable``, while an
unprobeable check, a switched-off write check and an undeclared app's row
keep their own outcome, which does not depend on the session.

Connectors run in sequence under one total timeout. The step it interrupts
(a session or a call) fails as ``timed_out`` with reason ``run_deadline``.
A connector it keeps from starting has a ``fail``/``not_reached`` session row
and the check rows of any session that did not pass; under a session that
passed, a call it keeps from starting is ``fail``/``not_reached``. The run is
then not ``complete``. A write check refused as ``not_reached``/``time_budget``
(``writes.py``) is the one exception: it is refused before the deadline, and
the run goes on.

The engine is meant for a one-shot CLI process: it does not take the web
process's probe slot (``services/connector_probe.py``), so it must not run
inside the web server.

Nothing from a tool's result, an exception or a fixture value reaches a row:
rows carry codes, check ids and tool names only.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ...config import get_connector_smoke_write_apps
from ..services.connector_probe import find_tool, probe_call_timeout_seconds
from ..services.mcp_connector_session import ConnectorToolsResult
from .assertions import judge_result, resolve_expected
from .declarations import (
    Assertion,
    ConnectorChecks,
    ReadCheck,
    UnprobeableCheck,
    WriteCheck,
    validate_connector_checks,
)
from .fixtures import FixtureMissing, resolve_fixtures
from .identity import (
    ConnectorTarget,
    IdentityAdapter,
    IdentitySetupError,
    SessionPurpose,
)
from .registry import connector_checks_for
from .results import (
    ENVIRONMENT_PATTERN,
    IDENTITY_PATTERN,
    CheckRow,
    FailureClass,
    Outcome,
    RunResult,
    SessionRow,
    classify_session,
)
from .writes import (
    PreparedWrite,
    WritePolicyFailed,
    new_run_id,
    prepare_writes,
    run_write,
    unavailable_write_row,
)

logger = logging.getLogger(__name__)

# The run's total bound when the caller gives none.
DEFAULT_TOTAL_TIMEOUT_SECONDS = 900.0

# The check id an app without a declaration reports, as the derived L3 check
# would be named (``registry.l3_read_check``).
_UNDECLARED_CHECK = "l3_probe"

# Reasons for ``timed_out``: the call's own bound, or the run's deadline.
_CHECK_CALL = "check_call"
_RUN_DEADLINE = "run_deadline"

# Finds a catalog app's declaration, or None.
DeclarationLookup = Callable[[str], ConnectorChecks | None]
_Check = ReadCheck | UnprobeableCheck | WriteCheck


@dataclass(frozen=True)
class _Prepared:
    """A read check with its fixtures resolved, or the key that had none.

    The resolved values are kept out of ``repr``: a fixture value never
    reaches a log or a row.
    """

    check: ReadCheck
    arguments: dict[str, Any] | None = field(default=None, repr=False)
    assertions: tuple[Assertion, ...] = field(default=(), repr=False)
    missing_key: str | None = None


@dataclass(frozen=True)
class _ConnectorPlan:
    """One selected target, its declaration (None for an undeclared app),
    its selected checks and their prepared read checks."""

    target: ConnectorTarget
    declaration: ConnectorChecks | None
    checks: tuple[_Check, ...]
    prepared: Mapping[str, _Prepared]
    # Whether the connector's write checks run, and each one prepared.
    writes_enabled: bool = False
    writes: Mapping[str, PreparedWrite] = field(default_factory=dict)

    @property
    def connector(self) -> str:
        return self.target.connector


class _ResolverFailed(Exception):
    """The adapter's fixture resolver raised; carries nothing from the error."""


class _SafeResolver:
    """Wraps the adapter's resolver so a raise becomes ``_ResolverFailed``."""

    def __init__(self, adapter: IdentityAdapter) -> None:
        self._adapter = adapter

    def resolve_fixture(self, key: str) -> Any:
        try:
            return self._adapter.resolve_fixture(key)
        except FixtureMissing:
            # A missing key all the same; reported under the requested key,
            # which is a validated code, never under one the adapter chose.
            raise FixtureMissing(key) from None
        except Exception as error:
            logger.warning("Resolving fixture %s raised %s", key, type(error).__name__)
            raise _ResolverFailed() from None


class _Clock:
    """The run's deadline, and whether it has cut anything."""

    def __init__(self, total_timeout: float) -> None:
        self._started = time.monotonic()
        self._deadline = self._started + total_timeout
        self.cut = False

    def elapsed(self) -> timedelta:
        return timedelta(seconds=time.monotonic() - self._started)

    def remaining(self) -> float:
        return self._deadline - time.monotonic()

    def expired(self) -> bool:
        if not self.cut and self.remaining() <= 0:
            self.cut = True
        return self.cut


async def run_checks(
    adapter: IdentityAdapter,
    *,
    environment: str | None = None,
    connectors: Collection[str] | None = None,
    checks: Collection[str] | None = None,
    check_timeout: float | None = None,
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT_SECONDS,
    declarations: DeclarationLookup = connector_checks_for,
) -> RunResult:
    """Run ``adapter``'s connectors' checks and return the run result.

    ``connectors`` selects connectors by their result label and ``checks``
    selects check ids; with a check selector, a connector none of whose
    checks it names is left out. A call's timeout defaults to the L3 probe
    call's (``connector_probe.probe_call_timeout_seconds``).

    Raises before any connector starts: ``ValueError`` for an invalid
    identity or environment label, a declaration that breaks the format or
    names another app, or a timeout that is a bool, NaN, infinite or not above
    0; ``TypeError`` for a selector given as one string, a timeout that is not
    a number, or a fixture value that is not a JSON scalar. A resolver that
    raises fails the run as ``fixtures:exception``, and a write policy that
    raises as ``write_policy:exception``.
    """
    started_at = datetime.now(timezone.utc)
    _require_finite_positive(total_timeout)
    if check_timeout is not None:
        _require_finite_positive(check_timeout)
    clock = _Clock(total_timeout)
    if isinstance(connectors, str) or isinstance(checks, str):
        raise TypeError("a selector is a collection of names, not one string")
    if not IDENTITY_PATTERN.fullmatch(adapter.label):
        raise ValueError("invalid identity label")
    if environment is not None and not ENVIRONMENT_PATTERN.fullmatch(environment):
        raise ValueError("invalid environment label")
    if check_timeout is None:
        check_timeout = float(probe_call_timeout_seconds())
    write_apps = get_connector_smoke_write_apps()
    run_id = new_run_id(started_at)

    def finish(rows: Sequence[SessionRow | CheckRow], failure: str | None) -> RunResult:
        return RunResult(
            identity=adapter.label,
            environment=environment,
            started_at=started_at,
            # Measured on the monotonic clock, so a wall-clock step during
            # the run can never put the end before the start.
            finished_at=started_at + clock.elapsed(),
            complete=failure is None and not clock.cut,
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
        plans = _plan(
            adapter,
            targets,
            declarations,
            connectors,
            checks,
            write_apps=write_apps,
            run_id=run_id,
        )
    except _ResolverFailed:
        return finish((), "fixtures:exception")
    except WritePolicyFailed:
        return finish((), "write_policy:exception")
    if not plans:
        return finish((), "selection:no_targets")
    if any(plan.writes for plan in plans):
        # The objects a run creates carry its id; the log ties them to it.
        # WARNING, so it reaches stderr under the CLI's logging setup.
        logger.warning("Write checks run under run id %s", run_id)

    rows: list[SessionRow | CheckRow] = []
    for plan in plans:
        rows.extend(
            await _run_connector(
                adapter, plan, clock=clock, check_timeout=check_timeout
            )
        )
    return finish(rows, None)


def _plan(
    adapter: IdentityAdapter,
    targets: Sequence[ConnectorTarget],
    declarations: DeclarationLookup,
    connectors: Collection[str] | None,
    checks: Collection[str] | None,
    *,
    write_apps: Collection[str],
    run_id: str,
) -> list[_ConnectorPlan]:
    """Every selected target's plan, built before any connector starts."""
    plans: list[_ConnectorPlan] = []
    for target in targets:
        if connectors is not None and target.connector not in connectors:
            continue
        declaration = declarations(target.app_id)
        if declaration is None:
            if checks is None or _undeclared_check_id(target.connector) in checks:
                plans.append(_ConnectorPlan(target, None, (), {}))
            continue
        if declaration.app_id != target.app_id:
            raise ValueError(
                f"the declaration for {target.connector} names another app"
            )
        validate_connector_checks(declaration)
        selected = tuple(
            check
            for check in (*declaration.reads, *declaration.writes)
            if checks is None or check.id in checks
        )
        if not selected and checks is not None:
            continue
        plans.append(
            _ConnectorPlan(
                target,
                declaration,
                selected,
                {},
                writes_enabled=target.app_id in write_apps,
            )
        )
    # Fixtures resolve only after every declaration has been validated, so an
    # invalid declaration raises whatever the order of the targets.
    return [
        plan
        if plan.target.skip_reason is not None
        else dataclasses.replace(
            plan,
            prepared={
                check.id: _prepare(check, adapter)
                for check in plan.checks
                if isinstance(check, ReadCheck)
            },
            writes=prepare_writes(
                plan.checks, plan.target, adapter, _SafeResolver(adapter), run_id
            )
            if plan.writes_enabled
            else {},
        )
        for plan in plans
    ]


def _raise_if_cancelled() -> None:
    # The bound's own cancellation is uncancelled when its block exits, so a
    # count left over is an outer cancel that a tool or adapter swallowed.
    task = asyncio.current_task()
    if task is not None and task.cancelling() > 0:
        raise asyncio.CancelledError


def _require_finite_positive(timeout: float) -> None:
    # A bool is an int; NaN breaks the event loop's timer ordering, and
    # infinity never expires.
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("a timeout must be a finite number of seconds above 0")


def _prepare(check: ReadCheck, adapter: IdentityAdapter) -> _Prepared:
    """``check`` with its fixtures resolved; only an unresolved key is a
    skip, and a resolver that raises is ``_ResolverFailed``."""
    resolver = _SafeResolver(adapter)
    try:
        arguments = resolve_fixtures(dict(check.arguments), resolver)
        assertions = resolve_expected(check.assertions, resolver)
    except FixtureMissing as missing:
        return _Prepared(check, missing_key=missing.key)
    return _Prepared(check, arguments, assertions)


async def _run_connector(
    adapter: IdentityAdapter,
    plan: _ConnectorPlan,
    *,
    clock: _Clock,
    check_timeout: float,
) -> list[SessionRow | CheckRow]:
    connector = plan.connector
    target = plan.target
    if target.skip_reason is not None:
        session = SessionRow(
            connector,
            Outcome.SKIP,
            FailureClass.SKIPPED_BY_IDENTITY,
            target.skip_reason,
        )
        return [session, *_without_session(plan)]
    if clock.expired():
        session = SessionRow(connector, Outcome.FAIL, FailureClass.NOT_REACHED)
        return [session, *_without_session(plan)]

    purpose = _session_purpose(plan.declaration)
    bound = asyncio.timeout(clock.remaining())
    started = time.monotonic()
    listing: ConnectorToolsResult | None = None
    error_type: str | None = None
    try:
        async with bound:
            listing = await adapter.open_session(target, purpose=purpose)
    except Exception as error:
        error_type = type(error).__name__
    _raise_if_cancelled()
    duration_ms = _elapsed_ms(started)
    tools: Any = ()
    if bound.expired():
        # The run's deadline ended the session, whether its cancellation
        # escaped as a TimeoutError or the adapter swallowed it and returned.
        clock.cut = True
        session = SessionRow(
            connector, Outcome.FAIL, FailureClass.TIMED_OUT, _RUN_DEADLINE, duration_ms
        )
    elif listing is None:
        # Anything the adapter raises, a TimeoutError of its own included: an
        # adapter reports a slow listing through the listing's outcome.
        logger.warning("Opening a session for %s raised %s", connector, error_type)
        session = SessionRow(
            connector,
            Outcome.FAIL,
            FailureClass.FAILED_TO_START,
            "exception",
            duration_ms,
        )
    else:
        outcome, failure_class, reason = classify_session(listing)
        session = SessionRow(connector, outcome, failure_class, reason, duration_ms)
        if listing.load_result is not None:
            tools = listing.load_result.tools
    if session.outcome is not Outcome.PASS:
        return [session, *_without_session(plan)]

    rows: list[SessionRow | CheckRow] = [session]
    for check in plan.checks:
        if isinstance(check, ReadCheck):
            rows.append(
                await _run_read(
                    connector,
                    plan.prepared[check.id],
                    tools,
                    clock=clock,
                    check_timeout=check_timeout,
                )
            )
        elif isinstance(check, WriteCheck) and plan.writes_enabled:
            rows.append(
                await run_write(
                    connector,
                    plan.writes[check.id],
                    tools,
                    clock=clock,
                    check_timeout=check_timeout,
                )
            )
        else:
            rows.append(_static_row(connector, check))
    if plan.declaration is None:
        rows.append(_undeclared_row(connector))
    return rows


async def _run_read(
    connector: str,
    prepared: _Prepared,
    tools: Any,
    *,
    clock: _Clock,
    check_timeout: float,
) -> CheckRow:
    check = prepared.check

    def row(
        outcome: Outcome,
        failure_class: FailureClass | None = None,
        reason: str | None = None,
        duration_ms: int = 0,
    ) -> CheckRow:
        return _read_row(connector, check, outcome, failure_class, reason, duration_ms)

    if prepared.missing_key is not None or prepared.arguments is None:
        return row(Outcome.SKIP, FailureClass.FIXTURE_MISSING, prepared.missing_key)
    tool = find_tool(tools, check.tool)
    if check.require_read_only_hint and not _advertises_read_only(tool):
        return row(Outcome.SKIP, FailureClass.READ_ONLY_HINT_MISSING)
    if tool is None:
        return row(Outcome.FAIL, FailureClass.TOOL_MISSING)
    if clock.expired():
        return row(Outcome.FAIL, FailureClass.NOT_REACHED)

    remaining = clock.remaining()
    bound = asyncio.timeout(min(check_timeout, remaining))
    started = time.monotonic()
    raw: Any = None
    call_error: Exception | None = None
    try:
        async with bound:
            raw = await tool.run_json_async(dict(prepared.arguments))
    except Exception as error:
        call_error = error
    _raise_if_cancelled()
    duration_ms = _elapsed_ms(started)
    if isinstance(call_error, TimeoutError) or bound.expired():
        # Any TimeoutError times the call out, as for the L3 probe call, and
        # so does an expired bound whose cancellation the tool swallowed.
        # Only the run's deadline, when it was the tighter bound, cuts the run.
        cut_by_deadline = bound.expired() and remaining < check_timeout
        if cut_by_deadline:
            clock.cut = True
        timeout_reason = _RUN_DEADLINE if cut_by_deadline else _CHECK_CALL
        return row(Outcome.FAIL, FailureClass.TIMED_OUT, timeout_reason, duration_ms)
    if call_error is not None:
        logger.warning(
            "Check %s (%s) raised %s", check.id, check.tool, type(call_error).__name__
        )
        return row(Outcome.FAIL, FailureClass.CALL_FAILED, "exception", duration_ms)
    failure_class, reason = judge_result(raw, check.payload, prepared.assertions)
    if failure_class is None:
        return row(Outcome.PASS, duration_ms=duration_ms)
    return row(Outcome.FAIL, failure_class, reason, duration_ms)


def _session_purpose(declaration: ConnectorChecks | None) -> SessionPurpose:
    """``LIST_ONLY`` when the app has no declaration or every check it
    declares is unprobeable (``identity.SessionPurpose``), else
    ``CALL_TOOLS``.

    Decided from the whole declaration, so neither a selector, a fixture nor
    a write switch changes how a connector starts or its session row.
    """
    if declaration is not None and (
        declaration.writes
        or any(isinstance(check, ReadCheck) for check in declaration.reads)
    ):
        return SessionPurpose.CALL_TOOLS
    return SessionPurpose.LIST_ONLY


def _advertises_read_only(tool: Any) -> bool:
    """Whether the server lists ``tool`` with ``readOnlyHint: true`` (and no
    ``destructiveHint``), read from its raw wire annotations."""
    # Imported here, as the connector session does, so importing the engine
    # does not load the MCP adapter module.
    from ...core.tools.adapters.vibe.mcp_adapter import MCPWriteHint

    return getattr(tool, "write_hint", None) is MCPWriteHint.READ_ONLY


def _without_session(plan: _ConnectorPlan) -> list[CheckRow]:
    """The check rows of a connector whose session did not pass."""
    if plan.declaration is None:
        return [_undeclared_row(plan.connector)]
    return [
        _read_row(plan.connector, check, Outcome.SKIP, FailureClass.SESSION_UNAVAILABLE)
        if isinstance(check, ReadCheck)
        else unavailable_write_row(plan.connector, check)
        if isinstance(check, WriteCheck) and plan.writes_enabled
        else _static_row(plan.connector, check)
        for check in plan.checks
    ]


def _read_row(
    connector: str,
    check: ReadCheck,
    outcome: Outcome,
    failure_class: FailureClass | None = None,
    reason: str | None = None,
    duration_ms: int = 0,
) -> CheckRow:
    return CheckRow(
        connector,
        check.id,
        "read",
        check.tool,
        outcome,
        failure_class,
        reason,
        duration_ms,
    )


def _static_row(connector: str, check: UnprobeableCheck | WriteCheck) -> CheckRow:
    """The row of a check whose outcome never depends on running it."""
    if isinstance(check, UnprobeableCheck):
        return CheckRow(
            connector,
            check.id,
            "read",
            None,
            Outcome.UNPROBEABLE,
            reason=check.reason,
        )
    return CheckRow(
        connector,
        check.id,
        "write",
        check.create.tool,
        Outcome.SKIP,
        FailureClass.WRITES_DISABLED,
    )


def _undeclared_check_id(connector: str) -> str:
    return f"{connector}.{_UNDECLARED_CHECK}"


def _undeclared_row(connector: str) -> CheckRow:
    return CheckRow(
        connector,
        _undeclared_check_id(connector),
        "read",
        None,
        Outcome.SKIP,
        FailureClass.NOT_DECLARED,
    )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
