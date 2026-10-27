"""The presence precheck in front of the dispatcher's candidate scan."""

# Pytest fixture imports are intentionally shadowed by test parameters.
# ruff: noqa: F811

from __future__ import annotations

import asyncio
import itertools
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from tests.shared.async_waits import DB_PROGRESS_TIMEOUT
from tests.shared.postgres_disposable import (
    load_migration_module,
)
from tests.web.services.test_dispatch_batch_line_scan import LIMIT, cap_of, fill
from tests.web.services.test_dispatch_cap import (  # noqa: F401
    CapHost,
    cap_host,
)
from xagent.config import SHARED_TASK_EXECUTION_ENABLED
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import (
    _CLAIMABLE_STATUS_CLAUSE,
    _UNFINISHED_STATUS_CLAUSE,
    CLAIMABLE_COMMAND_STATUSES,
    TERMINAL_COMMAND_STATUSES,
    TaskExecutionCommand,
)
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution_admission
from xagent.web.services.task_command_transport import (
    COMMAND_COMPLETED,
    COMMAND_FAILED,
    COMMAND_PENDING,
    COMMAND_PROCESSING,
    COMMAND_TERMINAL,
    TaskCommandKind,
)

REPO_ROOT = Path(__file__).parent.parent.parent.parent
MIGRATIONS = REPO_ROOT / "src/xagent/migrations/versions"
KIND_INDEX_MIGRATION = MIGRATIONS / "20261014_task_command_claimable_kind_index.py"
ID_INDEX_MIGRATION = MIGRATIONS / "20261015_task_command_claimable_id_index.py"
FAIR_INDEX_MIGRATION = MIGRATIONS / "20261022_task_command_claimable_fair_index.py"
UNFINISHED_INDEX_MIGRATION = (
    MIGRATIONS / "20261026_task_command_unfinished_task_index.py"
)

NOW = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)


def set_shared(monkeypatch: pytest.MonkeyPatch, shared: bool) -> None:
    monkeypatch.setenv(SHARED_TASK_EXECUTION_ENABLED, "true" if shared else "false")


def is_precheck(statement: str) -> bool:
    return statement.lstrip().startswith("SELECT EXISTS")


@contextmanager
def command_selects(host: CapHost) -> Iterator[list[str]]:
    """The SELECT statements that read the command table while the block runs.

    Recording stops at the claim's UPDATE: what follows reloads the claimed row
    and is not part of the scan.
    """
    engine = host.factory().get_bind()
    statements: list[str] = []
    claimed = False

    def record(_conn, _cursor, statement, *_args) -> None:
        nonlocal claimed
        text = statement.lstrip().upper()
        if text.startswith("UPDATE TASK_EXECUTION_COMMANDS"):
            claimed = True
        elif (
            not claimed
            and text.startswith("SELECT")
            and "FROM task_execution_commands" in statement
        ):
            statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


# ---------------------------------------------------------- the status set


def _migration_predicate(path: Path, name: str) -> str:
    # The variable is the partial-index WHERE clause every claimable-index
    # migration builds.
    return str(load_migration_module(path, name).PREDICATE.text)


def test_claimable_statuses_single_source() -> None:
    assert str(_CLAIMABLE_STATUS_CLAUSE) == "status IN ('pending', 'processing')"
    assert str(_CLAIMABLE_STATUS_CLAUSE) == _migration_predicate(
        KIND_INDEX_MIGRATION, "presence_kind_index_migration"
    )
    assert str(_CLAIMABLE_STATUS_CLAUSE) == _migration_predicate(
        ID_INDEX_MIGRATION, "presence_id_index_migration"
    )
    assert str(_CLAIMABLE_STATUS_CLAUSE) == _migration_predicate(
        FAIR_INDEX_MIGRATION, "presence_fair_index_migration"
    )
    assert set(CLAIMABLE_COMMAND_STATUSES) == {COMMAND_PENDING, COMMAND_PROCESSING}


def test_terminal_statuses_single_source() -> None:
    assert str(_UNFINISHED_STATUS_CLAUSE) == "status NOT IN ('completed', 'failed')"
    assert str(_UNFINISHED_STATUS_CLAUSE) == _migration_predicate(
        UNFINISHED_INDEX_MIGRATION, "presence_unfinished_index_migration"
    )
    assert set(TERMINAL_COMMAND_STATUSES) == set(COMMAND_TERMINAL)
    assert set(TERMINAL_COMMAND_STATUSES) == set(task_execution_admission._TERMINAL)


def test_claimable_and_terminal_statuses_cover_every_written_status() -> None:
    # The unfinished-task index holds every status that is not terminal, and the
    # claimable indexes hold the claimable ones: a status in neither set would be
    # indexed for one reader and not for the other.
    assert not set(CLAIMABLE_COMMAND_STATUSES) & set(TERMINAL_COMMAND_STATUSES)
    assert set(CLAIMABLE_COMMAND_STATUSES) | set(TERMINAL_COMMAND_STATUSES) == {
        COMMAND_PENDING,
        COMMAND_PROCESSING,
        COMMAND_COMPLETED,
        COMMAND_FAILED,
    }


_STATUS_COMPARISON = re.compile(
    r"task_execution_commands\.status\s*(?:=\s*'([^']*)'|IN\s*\(([^)]*)\))"
)


def status_literals(sql: str) -> set[str]:
    """Every status literal the compiled SQL compares ``status`` against."""
    found: set[str] = set()
    for equal, members in _STATUS_COMPARISON.findall(sql):
        if equal:
            found.add(equal)
        for member in re.findall(r"'([^']*)'", members):
            found.add(member)
    return found


@pytest.mark.parametrize("dialect", [sqlite.dialect(), postgresql.dialect()])
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "non-shared"])
def test_availability_predicate_statuses_subset_of_precheck(
    monkeypatch: pytest.MonkeyPatch, shared: bool, dialect: Any
) -> None:
    set_shared(monkeypatch, shared)

    sql = str(
        transport._claim_availability_predicate(NOW).compile(
            dialect=dialect, compile_kwargs={"literal_binds": True}
        )
    )

    statuses = status_literals(sql)
    assert statuses and COMMAND_PENDING in statuses, sql
    assert statuses <= set(CLAIMABLE_COMMAND_STATUSES), sql


# ------------------------------------------------------ necessary condition


@pytest.fixture
def plain_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Session, int]]:
    """One session over an in-memory queue holding a single task."""
    engine = sa.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        user = User(username="precheck-user", password_hash="unused")
        db.add(user)
        db.commit()
        task = Task(user_id=user.id, title="task", description="task")
        db.add(task)
        db.commit()
        yield db, int(task.id)
    engine.dispose()


HOUR = timedelta(hours=1)
TIMES = {"none": None, "past": NOW - HOUR, "future": NOW + HOUR}
PRECHECK_STATUSES = CLAIMABLE_COMMAND_STATUSES + COMMAND_TERMINAL


@pytest.mark.parametrize("kind", [kind.value for kind in TaskCommandKind])
@pytest.mark.parametrize("status", PRECHECK_STATUSES)
@pytest.mark.parametrize("shared", [True, False], ids=["shared", "non-shared"])
def test_presence_precheck_is_necessary_for_every_claimable_case(
    plain_db: tuple[Session, int],
    monkeypatch: pytest.MonkeyPatch,
    shared: bool,
    status: str,
    kind: str,
) -> None:
    set_shared(monkeypatch, shared)
    db, task_id = plain_db

    def lookup() -> Any:
        return transport._dispatcher_candidate_query(
            db, runner_id="worker", command_db_id=None, shared=shared, now=NOW
        )

    for (retry_name, retry), (expires_name, expires) in itertools.product(
        TIMES.items(), TIMES.items()
    ):
        case = f"retry={retry_name} expires={expires_name}"
        row = TaskExecutionCommand(
            task_id=task_id,
            command_id="c",
            kind=kind,
            payload={},
            status=status,
            retry_available_at=retry,
            claim_expires_at=expires,
        )
        db.add(row)
        db.commit()
        try:
            candidate = lookup().first() is not None
            reply = (
                lookup()
                .filter(TaskExecutionCommand.kind == TaskCommandKind.RESUME_INPUT.value)
                .first()
                is not None
            )
            any_open, any_reply = db.execute(
                transport._OPEN_COMMAND_AND_REPLY_PRESENCE
            ).one()
            any_open_alone = db.execute(transport._OPEN_COMMAND_PRESENCE).scalar_one()

            assert any_open == any_open_alone, case
            if candidate:
                assert any_open, case
            if reply:
                assert any_reply, case
            if status in COMMAND_TERMINAL:
                assert not candidate and not any_open, case
        finally:
            db.delete(row)
            db.commit()


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "non-shared"])
def test_necessary_condition_matrix_is_not_vacuous(
    plain_db: tuple[Session, int], monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    """A plain claimable command is a candidate, and the precheck sees it."""
    set_shared(monkeypatch, shared)
    db, task_id = plain_db
    db.add(
        TaskExecutionCommand(
            task_id=task_id, command_id="c", kind="pause", payload={}, status="pending"
        )
    )
    db.commit()

    candidate = transport._dispatcher_candidate_query(
        db, runner_id="worker", command_db_id=None, shared=shared, now=NOW
    ).first()

    assert candidate is not None
    assert db.execute(transport._OPEN_COMMAND_AND_REPLY_PRESENCE).one() == (True, False)


# ------------------------------------------------------------ idle behavior


class Spy:
    """Counts the calls a function receives and forwards them unchanged."""

    def __init__(self, target: Callable[..., Any]) -> None:
        self.target = target
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.target(*args, **kwargs)


def settle_history(host: CapHost) -> None:
    """Settled commands only: the table is not empty, nothing is claimable."""
    task = host.new_task()
    for status in (COMMAND_COMPLETED, COMMAND_FAILED):
        command = host.enqueue(task, "message")
        with host.factory() as db:
            db.get(TaskExecutionCommand, command).status = status
            db.commit()


@pytest.mark.parametrize("history", [False, True], ids=["empty", "settled-history"])
async def test_idle_scan_builds_no_candidate_query(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch, history: bool
) -> None:
    host = cap_host
    if history:
        settle_history(host)
    shared_builds = Spy(transport._dispatcher_candidate_query)
    plain_builds = Spy(transport._claimable_query)
    monkeypatch.setattr(transport, "_dispatcher_candidate_query", shared_builds)
    monkeypatch.setattr(transport, "_claimable_query", plain_builds)

    assert transport._find_command_candidate(None, (), cap_of(host)) is None
    assert shared_builds.calls == 0

    set_shared(monkeypatch, False)
    with host.factory() as db:
        assert transport.claim_task_command(db, runner_id="worker") is None
    assert plain_builds.calls == 0
    assert shared_builds.calls == 0


# --------------------------------------------------------- statement counts


@dataclass
class Row:
    """One row of the statement-count table.

    ``shared`` picks the execution mode, ``setup`` enqueues the commands (a
    label for each) and ``act`` runs the call under test and returns the id it
    selected, or None. A shared-mode claim without an id finds its candidate but
    holds no task lease in this test, so it selects nothing.
    """

    shared: bool
    setup: Callable[[CapHost], dict[str, int]]
    act: Callable[[CapHost, dict[str, int]], int | None]
    statements: int
    picked: str | None
    consumes: bool = True


def _nothing(host: CapHost) -> dict[str, int]:
    return {}


def _one_start(host: CapHost) -> dict[str, int]:
    fill(host, 1)
    return {"start": host.enqueue(host.new_task(), "start")}


def _one_pause(host: CapHost) -> dict[str, int]:
    return {"pause": host.enqueue(host.new_task(), "pause")}


def _claimable_reply(host: CapHost) -> dict[str, int]:
    fill(host, 1)
    host.enqueue(host.new_task(), "start")
    answer = host.enqueue(
        host.new_task(status=TaskStatus.WAITING_FOR_USER), "resume_input"
    )
    return {"answer": answer}


def _unclaimable_reply(host: CapHost) -> dict[str, int]:
    fill(host, 1)
    start = host.enqueue(host.new_task(), "start")
    answer = host.enqueue(
        host.new_task(status=TaskStatus.WAITING_FOR_USER), "resume_input"
    )
    with host.factory() as db:
        db.get(TaskExecutionCommand, answer).retry_available_at = datetime.now(
            timezone.utc
        ) + timedelta(days=1)
        db.commit()
    return {"start": start, "answer": answer}


def _closed_with_pause(host: CapHost) -> dict[str, int]:
    fill(host, LIMIT)
    return {"pause": host.enqueue(host.new_task(), "pause")}


def _scan(host: CapHost, _ids: dict[str, int]) -> int | None:
    candidate = transport._find_command_candidate(None, (), cap_of(host))
    return candidate.id if candidate else None


def _scan_closed(host: CapHost, _ids: dict[str, int]) -> int | None:
    cap = cap_of(host)
    assert cap.closed
    candidate = transport._find_command_candidate(None, (), cap)
    return candidate.id if candidate else None


def _scan_by_id(label: str) -> Callable[[CapHost, dict[str, int]], int | None]:
    def act(host: CapHost, ids: dict[str, int]) -> int | None:
        candidate = transport._find_command_candidate(ids[label], (), cap_of(host))
        return candidate.id if candidate else None

    return act


def _claim_without_id(host: CapHost, _ids: dict[str, int]) -> int | None:
    with host.factory() as db:
        claimed = transport.claim_task_command(db, runner_id="worker")
        return claimed.id if claimed else None


def _claim_by_id(label: str) -> Callable[[CapHost, dict[str, int]], int | None]:
    def act(host: CapHost, ids: dict[str, int]) -> int | None:
        with host.factory() as db:
            claimed = transport.claim_task_command(
                db, runner_id="worker", command_db_id=ids[label]
            )
            return claimed.id if claimed else None

    return act


MATRIX = [
    pytest.param(Row(True, _nothing, _scan, 1, None), id="empty-table"),
    pytest.param(Row(True, _one_start, _scan, 2, "start"), id="command-without-reply"),
    pytest.param(Row(True, _claimable_reply, _scan, 2, "answer"), id="claimable-reply"),
    pytest.param(
        Row(True, _unclaimable_reply, _scan, 3, "start"), id="unclaimable-reply"
    ),
    pytest.param(
        Row(True, _closed_with_pause, _scan_closed, 1, "pause"), id="cap-closed"
    ),
    pytest.param(
        Row(False, _nothing, _claim_without_id, 1, None),
        id="claim-no-id-non-shared-empty",
    ),
    pytest.param(
        Row(True, _nothing, _claim_without_id, 1, None),
        id="claim-no-id-shared-empty",
    ),
    pytest.param(
        Row(False, _one_pause, _claim_without_id, 2, "pause"),
        id="claim-no-id-non-shared-with-command",
    ),
    pytest.param(
        Row(True, _one_pause, _claim_without_id, 2, None),
        id="claim-no-id-shared-with-command",
    ),
    pytest.param(
        Row(True, _one_pause, _scan_by_id("pause"), 1, "pause"),
        id="scan-by-id-shared",
    ),
    pytest.param(
        Row(False, _one_pause, _claim_by_id("pause"), 1, "pause"),
        id="claim-by-id-non-shared",
    ),
    pytest.param(
        Row(False, _nothing, _scan, 1, None), id="non-shared-scan-empty-table"
    ),
    pytest.param(
        Row(False, _one_pause, _claim_without_id, 0, None, consumes=False),
        id="does-not-consume",
    ),
]


@pytest.mark.parametrize("row", MATRIX)
async def test_scan_statement_count_matrix(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch, row: Row
) -> None:
    host = cap_host
    set_shared(monkeypatch, row.shared)
    ids = row.setup(host)
    monkeypatch.setattr(transport, "consumes_task_commands", lambda: row.consumes)

    with command_selects(host) as statements:
        picked = row.act(host, ids)

    assert len(statements) == row.statements, statements
    assert picked == (ids[row.picked] if row.picked else None)


# ---------------------------------------------------------------- autoflush


def test_precheck_sees_unflushed_command_in_autoflush_session(
    plain_db: tuple[Session, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    set_shared(monkeypatch, False)
    db, task_id = plain_db
    assert db.autoflush

    def unflushed() -> TaskExecutionCommand:
        row = TaskExecutionCommand(
            task_id=task_id,
            command_id="unflushed",
            kind="resume_input",
            payload={},
            status="pending",
        )
        db.add(row)
        assert row in db.new
        return row

    unflushed()
    assert db.execute(transport._OPEN_COMMAND_PRESENCE).scalar_one() is True
    db.rollback()

    unflushed()
    assert db.execute(transport._OPEN_COMMAND_AND_REPLY_PRESENCE).one() == (
        True,
        True,
    )
    db.rollback()


# ------------------------------------------------------------ failure paths


def fail_precheck(engine: Any, failures: int) -> Callable[[], None]:
    """Make the next ``failures`` precheck statements raise; return the undo."""
    left = {"count": failures}

    def raise_unreachable(_conn, _cursor, statement, *_args) -> None:
        if is_precheck(statement) and left["count"]:
            left["count"] -= 1
            raise OperationalError(statement, None, Exception("database unreachable"))

    event.listen(engine, "before_cursor_execute", raise_unreachable)
    return lambda: event.remove(engine, "before_cursor_execute", raise_unreachable)


async def test_precheck_failure_propagates_to_backoff(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    host.enqueue(host.new_task(), "pause")
    builds = Spy(transport._dispatcher_candidate_query)
    monkeypatch.setattr(transport, "_dispatcher_candidate_query", builds)
    backoff_requests: list[int] = []

    builds_at_failure: list[int] = []

    def record_backoff(consecutive_failures: int) -> float:
        backoff_requests.append(consecutive_failures)
        builds_at_failure.append(builds.calls)
        return 0.001

    monkeypatch.setattr(
        transport, "_dispatcher_failure_backoff_seconds", record_backoff
    )
    monkeypatch.setattr(transport, "_dispatcher_wakeup", asyncio.Event())
    monkeypatch.setattr(transport, "DISPATCHER_IDLE_SECONDS", 0.001)
    undo = fail_precheck(host.factory().get_bind(), failures=2)
    worker = asyncio.create_task(
        transport._run_task_command_dispatcher_worker(host.handler)
    )
    try:
        async with asyncio.timeout(DB_PROGRESS_TIMEOUT):
            while not host.claimed:
                await asyncio.sleep(0.01)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        undo()

    # Each failed precheck counted as a failed dispatch and waited out its
    # backoff; the scans that failed built no candidate query.
    assert backoff_requests == [1, 2]
    assert builds_at_failure == [0, 0]
    assert [kind for _, kind in host.claimed] == [TaskCommandKind.PAUSE]


async def test_non_shared_precheck_failure_propagates_unchanged(
    cap_host: CapHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = cap_host
    set_shared(monkeypatch, False)
    host.enqueue(host.new_task(), "pause")
    plain_builds = Spy(transport._claimable_query)
    monkeypatch.setattr(transport, "_claimable_query", plain_builds)
    undo = fail_precheck(host.factory().get_bind(), failures=1)
    try:
        with host.factory() as db, pytest.raises(OperationalError) as raised:
            transport.claim_task_command(db, runner_id="worker")
    finally:
        undo()

    assert "database unreachable" in str(raised.value.orig)
    assert plain_builds.calls == 0
    # Nothing was claimed on the way out.
    with host.factory() as db:
        assert transport.claim_task_command(db, runner_id="worker") is not None


# ----------------------------------------------------- commit after precheck


async def test_command_committed_after_precheck_found_by_next_poll(
    cap_host: CapHost,
) -> None:
    host = cap_host
    assert transport._find_command_candidate(None, (), cap_of(host)) is None

    # The empty answer is not remembered: a command committed after the
    # precheck is found by the next scan.
    pause = host.enqueue(host.new_task(), "pause")

    found = transport._find_command_candidate(None, (), cap_of(host))
    assert found is not None and found.id == pause
