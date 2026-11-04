"""The public claimable count agrees with the dispatcher's own selection."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.orm import sessionmaker

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.models import database
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy

engine = engine_fixture
task_id = task_id_fixture

RUNNER = "worker"


@pytest.fixture
def factory(engine, monkeypatch):
    result = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "_SessionLocal", result)
    monkeypatch.setattr(transport, "get_runner_id", lambda: RUNNER)
    return result


def seed(factory, owner_task_id):
    """Persist one command per readiness case; return {case: command id}."""

    now = datetime.now(timezone.utc)
    expired, live = now - timedelta(seconds=5), now + timedelta(minutes=5)
    with factory() as db, db.begin():
        user_id = db.get(Task, owner_task_id).user_id

        def task(**fields):
            row = Task(user_id=user_id, title="case", **fields)
            db.add(row)
            db.flush()
            return row.id

        def command(tid, name, kind="message", **fields):
            row = TaskExecutionCommand(
                task_id=tid, command_id=name, kind=kind, payload={}, **fields
            )
            db.add(row)
            db.flush()
            return row.id

        idle = task()
        foreign_expired = task(
            status=TaskStatus.RUNNING,
            runner_id="retired",
            lease_attempt_id="retired-attempt",
            lease_expires_at=expired,
        )
        foreign_live = task(
            status=TaskStatus.RUNNING,
            runner_id="other",
            lease_attempt_id="other-attempt",
            lease_expires_at=live,
        )
        own_expired = task(
            status=TaskStatus.RUNNING,
            runner_id=RUNNER,
            lease_attempt_id="own-attempt",
            lease_expires_at=expired,
        )
        ids = {
            "idle_head": command(idle, "idle-head"),
            "idle_second": command(idle, "idle-second"),
            "foreign_expired_owner": command(foreign_expired, "foreign-expired"),
            "foreign_live_owner": command(foreign_live, "foreign-live"),
            "own_expired_owner": command(own_expired, "own-expired"),
            "start": command(task(), "start", kind="start"),
            "retry_later": command(task(), "retry-later", retry_available_at=live),
            "completed": command(task(), "completed", status="completed"),
        }
        db.add(TaskAdmissionBucket(key="team", capacity=1, max_pending=10))
        for case in ("admission_head", "admission_waiter"):
            tid = task()
            ids[case] = command(tid, case)
            db.add(
                TaskAdmissionTicket(
                    command_id=ids[case], bucket_key="team", task_id=tid
                )
            )
    return ids


def dispatcher_selects(factory, command_db_id, shared):
    if shared:
        return transport._find_command_candidate(command_db_id) is not None
    # Without shared execution the dispatcher claims through
    # claim_task_command, whose candidate lookup is the bare readiness query.
    with factory() as db:
        query = transport._claimable_query(
            db, runner_id=RUNNER, command_db_id=command_db_id
        )
        return query.first() is not None


@pytest.mark.parametrize(
    ("shared", "expected"),
    [
        (
            True,
            {
                "idle_head",
                "own_expired_owner",
                "start",
                "admission_head",
            },
        ),
        (
            False,
            {
                "idle_head",
                "foreign_expired_owner",
                "foreign_live_owner",
                "own_expired_owner",
                "retry_later",
                "admission_head",
            },
        ),
    ],
    ids=["shared", "local"],
)
def test_count_matches_dispatcher_selection_command_by_command(
    factory, task_id, monkeypatch, shared, expected
):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", str(shared).lower())
    ids = seed(factory, task_id)

    verdicts = {
        case: dispatcher_selects(factory, cid, shared) for case, cid in ids.items()
    }
    # Pin the fixture so a drift shared by both sides cannot pass unnoticed.
    assert {case for case, selected in verdicts.items() if selected} == expected

    # Compare one command at a time: hide every command, then restore them
    # in id order. The gates that read other commands' status (the
    # earlier-unfinished-command gate and the older-waiter check) only look
    # at smaller ids, and the test changes no task, ticket or pacing row. So
    # each restore must grow the count by exactly that command's verdict.
    with factory() as db:
        rows = {case: db.get(TaskExecutionCommand, cid) for case, cid in ids.items()}
        statuses = {case: row.status for case, row in rows.items()}
        for row in rows.values():
            row.status = "completed"
        db.flush()
        count = transport.count_claimable_task_commands(db, runner_id=RUNNER)
        assert count == 0
        for case in sorted(ids, key=ids.get):
            rows[case].status = statuses[case]
            db.flush()
            previous = count
            count = transport.count_claimable_task_commands(db, runner_id=RUNNER)
            assert (case, count - previous) == (case, int(verdicts[case]))
        db.rollback()


def test_count_ignores_process_local_busy_tasks(factory, task_id, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    ids = seed(factory, task_id)
    with factory() as db:
        busy = db.get(TaskExecutionCommand, ids["idle_head"]).task_id

    class BusyRegistry:
        def busy_command_tasks(self):
            return (busy,)

    # The dispatcher's busy list comes from this process's coordinator
    # registry. Make it report the idle task as busy through both ways of
    # reaching the registry, so a count that consulted it would drop it.
    registry = BusyRegistry()
    monkeypatch.setattr(runtime, "get_task_coordinator_registry", lambda: registry)
    monkeypatch.setattr(runtime, "_registry", registry)

    # The dispatcher skips its own busy tasks.
    assert transport._find_command_candidate(ids["idle_head"], (busy,)) is None
    # The count still reports every shared-mode selection in the seed,
    # idle_head included: idle_head, own_expired_owner, start, admission_head.
    with factory() as db:
        assert transport.count_claimable_task_commands(db, runner_id=RUNNER) == 4


def test_count_ignores_the_process_dispatch_cap(factory, task_id, monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    ids = seed(factory, task_id)
    # A closed cap: this process already runs as many executions as it may,
    # none of them for a seeded task.
    closed = DispatchCapSnapshot(
        policy=DispatchPolicy(limit=1, interactive_reserve=0),
        active_ids=frozenset({-1}),
    )
    assert closed.closed
    monkeypatch.setattr(transport, "build_dispatch_cap_snapshot", lambda: closed)

    # The dispatcher's scan skips a command that would open an execution.
    assert transport._find_command_candidate(ids["idle_head"], (), closed) is None
    # The count still reports every shared-mode selection in the seed.
    with factory() as db:
        assert transport.count_claimable_task_commands(db, runner_id=RUNNER) == 4
