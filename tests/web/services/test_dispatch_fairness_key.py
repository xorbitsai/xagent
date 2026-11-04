"""The tenant key written to each task command row and used to account for dispatch slots."""

import asyncio
import contextlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from tests.web.services.coordinator_command_shared import claim_for_owner
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_coordinator_runtime as runtime
from xagent.web.services import task_coordinator_service as ownership
from xagent.web.services import task_execution
from xagent.web.services import task_execution_admission as admission
from xagent.web.services.task_dispatch_cap import (
    DISPATCH_CLASS_INTERACTIVE,
    DispatchCapSnapshot,
    DispatchPolicy,
)
from xagent.web.services.task_execution import BackgroundTaskManager

KINDS = transport.TaskCommandKind


@pytest.fixture
def host(tmp_path, monkeypatch):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'fairness.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(database_module, "_SessionLocal", sessions)
    with sessions() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.commit()
        user_id = user.id

    class Host:
        def __init__(self):
            self.sessions = sessions
            self.engine = engine
            self.user = user_id
            self.counter = itertools.count(1)

        def task(self):
            with sessions() as db:
                task = Task(user_id=user_id, title="Work", status=TaskStatus.PENDING)
                db.add(task)
                db.commit()
                return task.id

        def stage(self, kind, *, task_id, command_id=None):
            command_id = command_id or f"{kind.value}-{next(self.counter)}"
            with sessions() as db:
                staged = transport.stage_task_command(
                    db,
                    task_id=task_id,
                    actor_user_id=user_id,
                    command_id=command_id,
                    kind=kind,
                    payload={},
                )
                db.commit()
                return staged

        def key(self, db_id):
            with sessions() as db:
                return db.execute(
                    sa.select(TaskExecutionCommand.fairness_key).where(
                        TaskExecutionCommand.id == db_id
                    )
                ).scalar_one()

        def seed(self, task_id, kind, fairness_key):
            with sessions() as db:
                db.add(
                    TaskExecutionCommand(
                        task_id=task_id,
                        command_id=f"seed-{next(self.counter)}",
                        kind=kind,
                        payload={},
                        status="completed",
                        fairness_key=fairness_key,
                    )
                )
                db.commit()

        @contextlib.contextmanager
        def statements(self):
            seen: list[str] = []

            def record(_conn, _cursor, statement, *_args):
                seen.append(" ".join(statement.split()))

            sa.event.listen(engine, "before_cursor_execute", record)
            try:
                yield seen
            finally:
                sa.event.remove(engine, "before_cursor_execute", record)

    try:
        yield Host()
    finally:
        engine.dispose()


def classifier_returning(**fields):
    def classify(db, command, view):
        return admission.CommandClassification(**fields)

    return classify


def test_a_new_execution_command_defaults_to_the_owner_key(host):
    task_id = host.task()
    staged = host.stage(KINDS.START, task_id=task_id)
    assert host.key(staged.staged_db_id) == f"user:{host.user}"


@pytest.mark.parametrize(
    "result,updates",
    [
        ({"fairness_key": "team:7"}, 1),
        (
            {"dispatch_class": DISPATCH_CLASS_INTERACTIVE, "fairness_key": "team:7"},
            1,
        ),
        ({"fairness_key": "owner"}, 0),
    ],
    ids=["key_only", "key_and_class", "key_equals_default"],
)
def test_a_classifier_key_is_written_in_the_flush_that_writes_the_class(
    host, result, updates
):
    result = dict(result)
    expected = result["fairness_key"]
    if expected == "owner":
        expected = result["fairness_key"] = f"user:{host.user}"
    admission.set_command_classifier(
        classifier_returning(**result), admission_enabled=False
    )
    task_id = host.task()
    with host.sessions() as db, db.no_autoflush, host.statements() as seen:
        staged = transport.stage_task_command(
            db,
            task_id=task_id,
            actor_user_id=host.user,
            command_id="start-1",
            kind=KINDS.START,
            payload={},
        )
        # Read before the commit and without autoflush: only a row the
        # acceptance step flushed shows the key.
        written = db.execute(
            sa.select(TaskExecutionCommand.fairness_key).where(
                TaskExecutionCommand.id == staged.staged_db_id
            )
        ).scalar_one()
        db.commit()
    assert written == expected
    changes = [s for s in seen if s.startswith("UPDATE task_execution_commands")]
    assert len(changes) == updates
    if "dispatch_class" in result:
        assert "fairness_key" in changes[0] and "dispatch_class" in changes[0]


@pytest.mark.parametrize(
    "key,valid",
    [("x", True), ("x" * 64, True), ("", False), ("x" * 65, False), (5, False)],
    ids=["one", "longest", "empty", "too_long", "not_text"],
)
def test_only_a_text_key_of_one_to_sixty_four_characters_is_accepted(host, key, valid):
    if valid:
        assert admission.CommandClassification(fairness_key=key).fairness_key == key
        return
    with pytest.raises(ValueError, match="Invalid fairness key"):
        admission.CommandClassification(fairness_key=key)

    # An enqueue whose classifier returns it leaves no row behind.
    def classify(db, command, view):
        return admission.CommandClassification(fairness_key=key)

    admission.set_command_classifier(classify, admission_enabled=False)
    task_id = host.task()
    with host.sessions() as db:
        with pytest.raises(ValueError, match="Invalid fairness key"):
            transport.stage_task_command(
                db,
                task_id=task_id,
                actor_user_id=host.user,
                command_id="start-1",
                kind=KINDS.START,
                payload={},
            )
        db.rollback()
        assert db.query(TaskExecutionCommand).count() == 0


# Earlier commands as (task, key), oldest first; task 1 is another task.
@pytest.mark.parametrize(
    "earlier,expected",
    [
        ([], "owner"),
        ([(0, "")], ""),
        ([(0, "team:7")], "team:7"),
        ([(0, "team:1"), (0, "team:7")], "team:7"),
        ([(0, "team:7"), (1, "team:9")], "team:7"),
    ],
    ids=["none", "pre_migration", "one", "latest_of_two", "other_task_newer"],
)
def test_a_control_command_takes_the_key_of_the_latest_command_of_its_task(
    host, earlier, expected
):
    tasks = [host.task(), host.task()]
    for task_index, key in earlier:
        host.seed(tasks[task_index], "start", key)
    cancel = host.stage(KINDS.CANCEL, task_id=tasks[0])
    expected = f"user:{host.user}" if expected == "owner" else expected
    assert host.key(cancel.staged_db_id) == expected


def test_a_repeated_command_id_changes_nothing(host):
    task_id = host.task()
    first = host.stage(KINDS.START, task_id=task_id, command_id="same")
    with host.statements() as seen:
        replay = host.stage(KINDS.START, task_id=task_id, command_id="same")
    assert replay.created is False
    assert replay.staged_db_id == first.staged_db_id
    assert host.key(first.staged_db_id) == f"user:{host.user}"
    assert not [s for s in seen if s.startswith("UPDATE task_execution_commands")]


# --- dispatch candidate and claim statements ----------------------------------

CLAIM_STATEMENTS = 4  # statements one claim issues
SELECTED_KEY = (
    ", task_execution_commands.fairness_key AS task_execution_commands_fairness_key"
)
# The statements a shared claim of a START issued before the key column existed
# (SQLite, whitespace collapsed). A test compares the claim with them, with the
# key column removed once from each statement. If the claim, admission or either
# table's columns change on purpose, re-record the file from the current tree
# (the list the failing assertion prints) or drop the comparison.
CLAIM_BASELINE_HINT = (
    "The shared claim no longer matches the statements recorded before the key "
    "column existed. If that is intended, re-record "
    "fixtures/shared_claim_statements_before_fairness_key.json from the current "
    "tree with SELECTED_KEY removed once from each statement (the list printed "
    "below), or drop this comparison."
)
SHARED_CLAIM_BEFORE = json.loads(
    (
        Path(__file__).parent
        / "fixtures"
        / "shared_claim_statements_before_fairness_key.json"
    ).read_text()
)
POLICY = DispatchPolicy(limit=2, interactive_reserve=1)
# (name, other active tasks, candidate's task active too, statements, found)
CAP_STATES = [
    ("open", frozenset(), False, 2, True),
    ("batch_line", frozenset({9001}), False, 2, True),
    ("full", frozenset({9001, 9002}), False, 1, False),
    ("full_task_active", frozenset({9001}), True, 1, True),
]


def captured_statements(host, action):
    with host.statements() as seen:
        result = action()
    return result, seen


COUNT_PREFIX = "SELECT count(*) AS count_1 FROM ("


def assert_key_only_in_the_count_select_list(statement):
    # The count wraps the candidate query: the key belongs to the inner select
    # list only, not to the outer select list and not after the inner FROM.
    assert statement.startswith(COUNT_PREFIX), statement
    select_list, _, rest = statement[len(COUNT_PREFIX) :].partition(" FROM ")
    assert "fairness_key" in select_list, statement
    assert "fairness_key" not in rest, statement


def assert_key_only_in_the_select_list(statements):
    # A SELECT may name the key before its first FROM only; any other statement
    # (an UPDATE's SET or WHERE included) may not name it at all.
    for statement in statements:
        if statement.startswith(COUNT_PREFIX):
            assert_key_only_in_the_count_select_list(statement)
        elif statement.startswith("SELECT "):
            _, _, rest = statement.partition(" FROM ")
            assert "fairness_key" not in rest, statement
        else:
            assert "fairness_key" not in statement, statement


def staged_interactive_start(host, kind=KINDS.START):
    admission.set_command_classifier(
        classifier_returning(
            dispatch_class=DISPATCH_CLASS_INTERACTIVE, fairness_key="team:7"
        ),
        admission_enabled=False,
    )
    task_id = host.task()
    return task_id, host.stage(kind, task_id=task_id)


@pytest.mark.parametrize(
    "state,others,task_active,expected_statements,found", CAP_STATES
)
def test_the_candidate_scan_carries_the_row_key_and_adds_no_statement(
    host, state, others, task_active, expected_statements, found
):
    task_id, staged = staged_interactive_start(host)
    active = others | ({task_id} if task_active else frozenset())
    cap = DispatchCapSnapshot(policy=POLICY, active_ids=frozenset(active))

    candidate, statements = captured_statements(
        host, lambda: transport._find_command_candidate(None, (), cap)
    )

    assert len(statements) == expected_statements
    assert_key_only_in_the_select_list(statements)
    if found:
        assert candidate is not None
        assert candidate.id == staged.staged_db_id
        assert candidate.fairness_key == "team:7"
    else:
        assert candidate is None


def test_the_claim_lookup_adds_no_statement_and_keeps_the_key_in_the_select_list(
    host, monkeypatch
):
    # Without shared execution the claim is a plain lookup and update, with no
    # task lease to hold; START stays dormant there, so claim a message.
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    task_id, staged = staged_interactive_start(host, KINDS.MESSAGE)
    cap = DispatchCapSnapshot(policy=POLICY, active_ids=frozenset())

    def claim():
        with host.sessions() as db:
            claimed = transport.claim_task_command(
                db, runner_id="worker", command_db_id=staged.staged_db_id, cap=cap
            )
            db.commit()
            return claimed

    claimed, statements = captured_statements(host, claim)

    assert claimed is not None
    assert len(statements) == CLAIM_STATEMENTS
    assert_key_only_in_the_select_list(statements)


@pytest.mark.parametrize("shared,expected", [("true", 2), ("false", 1)])
def test_the_count_keeps_the_key_in_its_inner_select_list(
    host, monkeypatch, shared, expected
):
    # One START and one MESSAGE: without shared execution START stays dormant,
    # so the count also shows which mode the statement was built for.
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", shared)
    staged_interactive_start(host, KINDS.START)
    staged_interactive_start(host, KINDS.MESSAGE)

    def count():
        with host.sessions() as db:
            return transport.count_claimable_task_commands(db, runner_id="worker")

    counted, statements = captured_statements(host, count)

    assert counted == expected
    assert len(statements) == 1
    assert_key_only_in_the_count_select_list(statements[0])


def test_the_shared_claim_of_a_start_matches_the_statements_before_the_key(host):
    task_id, staged = staged_interactive_start(host)
    with host.sessions() as db:
        lease = ownership.acquire_task_lease_no_commit(db, task_id, runner_id="worker")
        db.commit()
    assert lease is not None

    def claim():
        with host.sessions() as db:
            return claim_for_owner(db, lease, staged.staged_db_id)

    claimed, statements = captured_statements(host, claim)

    assert claimed is not None
    assert claimed.id == staged.staged_db_id
    assert claimed.kind == KINDS.START
    # The claim lookup and the refetch select the row; nothing else names the key.
    recorded = json.dumps(
        [s.replace(SELECTED_KEY, "", 1) for s in statements], indent=1
    )
    message = f"{CLAIM_BASELINE_HINT}\n{recorded}"
    assert [s.count(SELECTED_KEY) for s in statements] == [1, 0, 0, 0, 1, 0], message
    assert_key_only_in_the_select_list(statements)
    assert [s.replace(SELECTED_KEY, "", 1) for s in statements] == (
        SHARED_CLAIM_BEFORE
    ), message


def test_a_dispatch_candidate_cannot_be_built_without_its_key():
    with pytest.raises(TypeError, match="fairness_key"):
        transport.CommandCandidate(11, 5, KINDS.START, True)


async def test_dispatch_passes_the_candidate_key_to_the_slot_reservation(
    host, monkeypatch
):
    manager = BackgroundTaskManager()
    monkeypatch.setattr(task_execution, "background_task_manager", manager)
    monkeypatch.setattr(
        runtime,
        "get_task_coordinator_registry",
        lambda: SimpleNamespace(busy_command_tasks=lambda: set()),
    )
    monkeypatch.setattr(
        transport,
        "_find_command_candidate",
        lambda *_args, **_kwargs: transport.CommandCandidate(
            11, 5, KINDS.START, True, "team:3"
        ),
    )
    seen: list[dict] = []

    def reserve(task_id, **kwargs):
        seen.append({"task_id": task_id, **kwargs})
        return None

    monkeypatch.setattr(manager, "reserve_dispatch_slot", reserve)

    async def executor(command):
        raise AssertionError("no command should be applied")

    assert await transport.dispatch_one_task_command(executor) is False
    assert len(seen) == 1
    assert seen[0]["task_id"] == 5
    assert seen[0]["interactive"] is True
    assert seen[0]["fairness_key"] == "team:3"


# --- slot accounting per tenant ------------------------------------------------

LARGE = DispatchPolicy(limit=10, interactive_reserve=0)


def reserve(manager, task_id, key):
    token = manager.reserve_dispatch_slot(
        task_id, policy=LARGE, interactive=False, fairness_key=key
    )
    assert token is not None
    return token


async def test_a_task_already_active_keeps_its_key():
    manager = BackgroundTaskManager()
    reserve(manager, 7, "team:1")
    reserve(manager, 8, "team:1")

    reserve(manager, 7, "team:2")

    assert manager.slot_tenant_counts() == (1, 2)
    assert manager._slot_keys == {7: "team:1", 8: "team:1"}


@pytest.mark.parametrize(
    "reservations,expected",
    [
        ([(7, "team:1"), (8, "team:1")], (1, 2)),
        ([(1, "a"), (2, "a"), (3, "a"), (4, "b"), (5, "c")], (3, 3)),
        # The empty key of a pre-migration row is an ordinary tenant.
        ([(3, "")], (1, 1)),
        # A reservation without a key records nothing.
        ([(3, None)], (0, 0)),
    ],
    ids=["one_tenant", "largest_of_three", "empty_key", "no_key"],
)
async def test_slots_are_counted_per_tenant(reservations, expected):
    manager = BackgroundTaskManager()
    for task_id, key in reservations:
        reserve(manager, task_id, key)
    assert manager.slot_tenant_counts() == expected
    assert manager._slot_keys == {t: k for t, k in reservations if k is not None}


async def test_an_execution_registered_outside_the_dispatcher_counts_for_no_tenant():
    manager = BackgroundTaskManager()
    finish = asyncio.Event()
    handle = asyncio.get_running_loop().create_task(finish.wait())
    try:
        manager.register_task(21, handle)
        assert 21 in manager._active_task_ids()
        assert manager.slot_tenant_counts() == (0, 0)
    finally:
        finish.set()
        await handle


async def test_the_key_map_is_cleaned_only_by_the_cap_snapshot():
    manager = BackgroundTaskManager()

    # A reservation that ends without an execution.
    token = reserve(manager, 7, "team:1")
    manager.release_dispatch_slot(token)
    assert manager._slot_keys == {7: "team:1"}
    assert manager.slot_tenant_counts() == (0, 0)
    assert manager._slot_keys == {7: "team:1"}
    manager.dispatch_cap_snapshot(LARGE)
    assert manager._slot_keys == {}

    # A reservation whose execution is registered before the token is released.
    finish = asyncio.Event()
    handle = asyncio.get_running_loop().create_task(finish.wait())
    token = reserve(manager, 8, "team:2")
    manager.register_task(8, handle)
    manager.release_dispatch_slot(token)
    manager.dispatch_cap_snapshot(LARGE)
    assert manager._slot_keys == {8: "team:2"}
    assert manager.slot_tenant_counts() == (1, 1)

    finish.set()
    await handle
    await asyncio.sleep(0)
    manager.dispatch_cap_snapshot(LARGE)
    assert manager._slot_keys == {}


@pytest.mark.parametrize("ending", ["shutdown", "start_accepting"])
async def test_the_key_map_is_cleared_with_the_other_tables(ending):
    manager = BackgroundTaskManager()
    manager._slot_keys[7] = "team:1"
    if ending == "shutdown":
        await manager.shutdown()
    else:
        # Leftover keys are derived accounting, not background work: no refusal.
        manager.start_accepting()
    assert manager._slot_keys == {}
