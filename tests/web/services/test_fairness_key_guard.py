"""The test-time guard that refuses any rewrite of a command row's tenant key."""

import itertools

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from tests.shared import fairness_key_guard as guard
from xagent.web.models import database as database_module
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as transport
from xagent.web.services import task_execution_admission as admission

KINDS = transport.TaskCommandKind


@pytest.fixture
def host(tmp_path, monkeypatch):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'guard.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(database_module, "_SessionLocal", sessions)
    with sessions() as db:
        user = User(username="owner", password_hash="unused")
        db.add(user)
        db.commit()
        user_id = user.id
    counter = itertools.count(1)

    class Host:
        sessions_ = sessions
        user = user_id

        def task(self, db):
            task = Task(user_id=user_id, title="Work", status=TaskStatus.PENDING)
            db.add(task)
            db.flush()
            return task.id

        def command(self, db, key="user:1"):
            command = TaskExecutionCommand(
                task_id=self.task(db),
                command_id=f"c{next(counter)}",
                kind="start",
                payload={},
                status="pending",
                fairness_key=key,
            )
            db.add(command)
            db.flush()
            return command

        def stage(self, db, *, task_id=None):
            return transport.stage_task_command(
                db,
                task_id=task_id or self.task(db),
                actor_user_id=user_id,
                command_id=f"s{next(counter)}",
                kind=KINDS.START,
                payload={},
            )

    try:
        yield Host()
    finally:
        engine.dispose()


def refused_once() -> str:
    """Consume the single refusal the guard recorded and return its statement."""
    (refusal,) = guard.take_violations()
    return refusal[1]


def classify_with(action):
    """Install a classifier that runs ``action(db, command)`` during acceptance."""

    def classify(db, command, view):
        action(db, command)
        return admission.CommandClassification()

    admission.set_command_classifier(classify, admission_enabled=False)


# --- statements the guard lets through -----------------------------------------


def test_a_command_accepted_with_a_classifier_key_is_let_through(host) -> None:
    admission.set_command_classifier(
        lambda db, command, view: admission.CommandClassification(
            fairness_key="team:7"
        ),
        admission_enabled=False,
    )
    with host.sessions_() as db:
        staged = host.stage(db)
        db.commit()
        key = db.get(TaskExecutionCommand, staged.staged_db_id).fairness_key
    assert key == "team:7"
    assert guard.take_violations() == []


def test_other_statements_are_let_through(host) -> None:
    command_table = TaskExecutionCommand.__table__
    with host.sessions_() as db:
        command = host.command(db)
        db.execute(sa.select(command_table.c.fairness_key))
        db.execute(
            sa.update(Task).where(Task.id == command.task_id).values(title="renamed")
        )
        command.status = "completed"
        db.flush()
        db.execute(
            sa.update(command_table)
            .where(command_table.c.id == command.id)
            .values(error="done")
        )
        # An upsert that leaves the key alone.
        db.execute(
            sqlite_insert(command_table)
            .values(
                id=command.id,
                task_id=command.task_id,
                command_id="again",
                kind="start",
                payload={},
            )
            .on_conflict_do_update(index_elements=["id"], set_={"error": "again"})
        )
        # Reads the key without writing it.
        db.execute(
            sa.update(command_table)
            .where(command_table.c.id == command.id)
            .values(error=sa.select(command_table.c.fairness_key).scalar_subquery())
        )
    assert guard.take_violations() == []


# --- statements the guard refuses ----------------------------------------------


def test_assigning_the_key_of_a_stored_command_is_refused(host) -> None:
    with host.sessions_() as db:
        command = host.command(db)
        command.fairness_key = "team:9"
        with pytest.raises(AssertionError, match="fairness_key"):
            db.flush()
    assert "UPDATE task_execution_commands" in refused_once()


def aliased_update(table, command):
    alias = table.alias("a")
    return sa.update(alias).where(alias.c.id == command.id).values(fairness_key="x")


@pytest.mark.parametrize(
    "write",
    [
        lambda t, c: sa.update(t).where(t.c.id == c.id).values(fairness_key="x"),
        lambda t, c: sa.update(t).where(t.c.id == c.id).values(**{"fairness_key": "x"}),
        lambda t, c: sa.update(t).where(t.c.id == c.id).values({"fairness_key": "x"}),
        lambda t, c: sa.text(
            "UPDATE \"task_execution_commands\" SET status = status, fairness_key = 'x' "
            f"WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            f"update task_execution_commands set fairness_key='x' where id = {c.id}"
        ),
        lambda t, c: aliased_update(t, c),
        lambda t, c: (
            sqlite_insert(t)
            .values(
                id=c.id,
                task_id=c.task_id,
                command_id="again",
                kind="start",
                payload={},
            )
            .on_conflict_do_update(index_elements=["id"], set_={"fairness_key": "x"})
        ),
        lambda t, c: sa.text(
            "/* a note */ UPDATE task_execution_commands SET fairness_key = 'x' "
            f"WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            "-- a note\nUPDATE task_execution_commands SET fairness_key = 'x' "
            f"WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            "WITH q AS (SELECT 1) UPDATE task_execution_commands "
            f"SET fairness_key = 'x' WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            "UPDATE OR REPLACE task_execution_commands SET fairness_key = 'x' "
            f"WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            f"UPDATE task_execution_commands SET FAIRNESS_KEY = 'x' WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            f"UPDATE task_execution_commands SET Fairness_Key = 'x' WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            "UPDATE task_execution_commands SET \"FAIRNESS_KEY\" = 'x' "
            f"WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            "UPDATE task_execution_commands SET error = 'WHERE', fairness_key = 'x' "
            f"WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            "UPDATE task_execution_commands SET error = 'from', fairness_key = 'x' "
            f"WHERE id = {c.id}"
        ),
        lambda t, c: sa.text(
            "UPDATE task_execution_commands SET error = "
            "(SELECT 'a' FROM task_execution_commands LIMIT 1), fairness_key = 'x' "
            f"WHERE id = {c.id}"
        ),
    ],
    ids=[
        "keyword",
        "unpacked",
        "mapping",
        "quoted_raw",
        "lowercase_keywords",
        "alias",
        "upsert",
        "block_comment",
        "line_comment",
        "with_clause",
        "or_replace",
        "uppercase_key",
        "mixed_case_key",
        "quoted_uppercase_key",
        "where_in_a_value",
        "from_in_a_value",
        "subquery_before_the_key",
    ],
)
def test_other_ways_to_write_the_key_are_refused(host, write) -> None:
    with host.sessions_() as db:
        command = host.command(db)
        with pytest.raises(AssertionError, match="fairness_key"):
            db.execute(write(TaskExecutionCommand.__table__, command))
    refused_once()


@pytest.mark.parametrize(
    "write",
    [
        lambda c: sa.text(
            "UPDATE task_execution_commands SET error = 'x' "
            f"WHERE fairness_key = 'user:1' AND id = {c.id}"
        ),
        lambda c: sa.text(
            f"UPDATE task_execution_commands SET error = 'fairness_key' WHERE id = {c.id}"
        ),
        lambda c: sa.text(
            "UPDATE task_execution_commands SET error = "
            "(SELECT fairness_key FROM task_execution_commands LIMIT 1) "
            f"WHERE id = {c.id}"
        ),
    ],
    ids=["key_only_in_where", "key_only_in_a_value", "key_only_read_in_set"],
)
def test_raw_sql_that_updates_the_table_and_mentions_the_key_anywhere_is_refused(
    host, write
) -> None:
    """Raw SQL is not parsed, so any mention of the key in a raw UPDATE of the table
    is refused, written or not."""
    with host.sessions_() as db:
        command = host.command(db)
        with pytest.raises(AssertionError, match="fairness_key"):
            db.execute(write(command))
    refused_once()


def test_raw_sql_that_does_not_update_the_table_is_let_through(host) -> None:
    with host.sessions_() as db:
        command = host.command(db)
        db.execute(sa.text("SELECT FAIRNESS_KEY FROM task_execution_commands"))
        db.execute(
            sa.text(
                "UPDATE tasks SET title = "
                "(SELECT fairness_key FROM task_execution_commands LIMIT 1) "
                f"WHERE id = {command.task_id}"
            )
        )
        db.execute(
            sa.text(
                f"UPDATE task_execution_commands SET error = 'x' WHERE id = {command.id}"
            )
        )
    assert guard.take_violations() == []


def test_an_upsert_that_names_the_key_in_another_case_is_refused(host) -> None:
    table = TaskExecutionCommand.__table__
    with host.sessions_() as db:
        command = host.command(db)
        upsert = (
            sqlite_insert(table)
            .values(
                id=command.id,
                task_id=command.task_id,
                command_id="again",
                kind="start",
                payload={},
            )
            .on_conflict_do_update(index_elements=["id"], set_={"FAIRNESS_KEY": "x"})
        )
        # SQLAlchemy only warns about a name that is not a column key; SQLite
        # reads it as the key column.
        with pytest.warns(sa.exc.SAWarning, match="Additional column names"):
            with pytest.raises(AssertionError, match="fairness_key"):
                db.execute(upsert)
    refused_once()


class AppendsAfterTheCopy(list):
    """A record into which another thread appends just after a copy is taken."""

    def __init__(self, items, late):
        super().__init__(items)
        self.late = late
        self.appended = False

    def _append_once(self):
        if not self.appended:
            self.appended = True
            super().append(self.late)

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(index, slice):
            self._append_once()
        return value

    def __iter__(self):
        items = list(super().__iter__())
        self._append_once()
        return iter(items)


def test_a_record_added_while_the_records_are_taken_is_kept_for_the_next_take(
    monkeypatch,
) -> None:
    early = ("t::early", "UPDATE early", None)
    late = ("t::late", "UPDATE late", None)
    monkeypatch.setattr(guard, "violations", AppendsAfterTheCopy([early], late))

    assert guard.take_violations() == [early]
    assert guard.take_violations() == [late]
    assert guard.take_violations() == []


def test_a_bulk_update_through_the_query_is_refused(host) -> None:
    with host.sessions_() as db:
        command = host.command(db)
        with pytest.raises(AssertionError):
            db.query(TaskExecutionCommand).filter(
                TaskExecutionCommand.id == command.id
            ).update(
                {TaskExecutionCommand.fairness_key: "x"}, synchronize_session=False
            )
    refused_once()


def stage_task_command_that_rewrites(db, command):
    """A function of the same name as the real one, which is not the real one."""
    command.fairness_key = "team:9"
    db.flush()


def test_a_function_called_stage_task_command_is_not_let_through(host) -> None:
    stage_task_command_that_rewrites.__code__ = (
        stage_task_command_that_rewrites.__code__.replace(co_name="stage_task_command")
    )
    with host.sessions_() as db:
        command = host.command(db)
        with pytest.raises(AssertionError):
            stage_task_command_that_rewrites(db, command)
    refused_once()


def test_the_real_acceptance_may_not_write_the_key_of_another_command(host) -> None:
    with host.sessions_() as db:
        other = host.command(db)

        def rewrite_another(db, command):
            other.fairness_key = "team:9"
            db.flush()

        classify_with(rewrite_another)
        with pytest.raises(AssertionError):
            host.stage(db)
    refused_once()


def test_a_batch_that_mixes_in_another_command_is_refused(host) -> None:
    with host.sessions_() as db:

        def rewrite_both(db, command):
            other = host.command(db)  # inserted after, so its id is larger
            command.fairness_key = "team:1"
            other.fairness_key = "team:2"
            db.flush()

        classify_with(rewrite_both)
        with pytest.raises(AssertionError):
            host.stage(db)
    assert "UPDATE task_execution_commands" in refused_once()


def test_raw_sql_is_refused_inside_the_acceptance(host) -> None:
    with host.sessions_() as db:

        def raw(db, command):
            db.execute(
                sa.text(
                    "UPDATE task_execution_commands SET fairness_key = 'x' "
                    "WHERE id = :id"
                ),
                {"id": command.id},
            )

        classify_with(raw)
        with pytest.raises(AssertionError):
            host.stage(db)
    refused_once()


# --- the decision, on its own ---------------------------------------------------


@pytest.mark.parametrize(
    "parameter_sets,id_parameter,command_id,allowed",
    [
        ([{"id": 5}], "id", 5, True),
        ([{"id": 5}, {"id": 5}], "id", 5, True),
        ([{"id": 5}], "id", 6, False),
        # The first set is the command's own, a later one is another row's.
        ([{"id": 5}, {"id": 6}], "id", 5, False),
        ([{"id": 6}, {"id": 5}], "id", 5, False),
        ([], "id", 5, False),
        ([{"id": 5}], None, 5, False),
        ([{"id": 5}], "id", None, False),
        ([{"other": 5}], "id", 5, False),
    ],
    ids=[
        "own",
        "own_twice",
        "other",
        "own_then_other",
        "other_then_own",
        "no_parameters",
        "no_id_parameter",
        "no_command_id",
        "id_missing",
    ],
)
def test_a_write_is_allowed_only_when_every_parameter_set_names_the_command(
    parameter_sets, id_parameter, command_id, allowed
) -> None:
    assert (
        guard.every_parameter_set_targets(parameter_sets, id_parameter, command_id)
        is allowed
    )


def test_outside_the_acceptance_no_command_is_being_accepted() -> None:
    assert guard._command_being_accepted() is None


def test_a_refusal_that_the_code_under_test_swallows_still_fails_the_test(
    pytester,
) -> None:
    pytester.makeconftest('pytest_plugins = ["tests.shared.fairness_key_guard_plugin"]')
    pytester.makepyfile(
        """
        import sqlalchemy as sa

        def test_swallows_the_refusal():
            engine = sa.create_engine("sqlite://")
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE task_execution_commands (id INTEGER, fairness_key TEXT)"
                )
            try:
                with engine.begin() as connection:
                    connection.exec_driver_sql(
                        "UPDATE task_execution_commands SET fairness_key = 'x'"
                    )
            except Exception:
                pass  # what a broad handler in production code would do
        """
    )

    result = pytester.runpytest_inprocess("-p", "no:cacheprovider")

    # The test body passes; the check when the test ends reports the refusal.
    result.assert_outcomes(passed=1, errors=1)
    assert any("rewrote" in line or "tenant key" in line for line in result.outlines)
