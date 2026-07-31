"""Admin user deletion must not run its ORM work on the event loop."""

from __future__ import annotations

import asyncio
import threading

import pytest
from sqlalchemy import event, text

from xagent.core.task_runtime import TaskRuntimeContribution
from xagent.web.api import admin_users as admin_users_module
from xagent.web.api.admin_users import delete_user
from xagent.web.models.database import get_engine
from xagent.web.models.task import DAGExecution, DAGExecutionPhase, Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User
from xagent.web.services.task_runtime import (
    agent_config_with_task_extension_bindings,
    register_task_extension,
    unregister_task_extension,
)

from .conftest import _admin_headers, _direct_db_session, _register_second_user

pytestmark = pytest.mark.usefixtures("_test_db")


def _seed_user_with_tasks(db, username: str, task_count: int) -> tuple[int, list[int]]:
    target = db.query(User).filter(User.username == username).one()
    tasks = [
        Task(user_id=int(target.id), title=f"task {index}", description="")
        for index in range(task_count)
    ]
    db.add_all(tasks)
    db.flush()
    for task in tasks:
        db.add(DAGExecution(task_id=int(task.id), phase=DAGExecutionPhase.COMPLETED))
        db.add(
            UploadedFile(
                user_id=int(target.id),
                task_id=int(task.id),
                filename=f"f{task.id}.txt",
                storage_path=f"/tmp/f{task.id}.txt",
                file_size=1,
            )
        )
    db.commit()
    return int(target.id), [int(task.id) for task in tasks]


@pytest.mark.asyncio
async def test_admin_user_delete_runs_task_purge_off_the_event_loop() -> None:
    """The per-user purge is synchronous ORM work; it must run in a thread.

    Rather than patching ``asyncio.to_thread``, this watches the SQLAlchemy
    cursor events and asserts the ``DELETE FROM tasks`` statement is issued
    from a thread other than the one running the event loop.
    """

    _admin_headers()
    _register_second_user("offload-user", "offloadpass1")
    db = _direct_db_session()
    engine = get_engine()
    delete_threads: list[int] = []

    def _record(conn, cursor, statement, parameters, context, executemany) -> None:
        normalized = " ".join(statement.split()).lower()
        if normalized.startswith("delete from tasks"):
            delete_threads.append(threading.get_ident())

    event.listen(engine, "before_cursor_execute", _record)
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target_id, task_ids = _seed_user_with_tasks(db, "offload-user", 3)

        loop_thread_ident = threading.get_ident()
        assert asyncio.get_running_loop() is not None

        response = await delete_user(target_id, admin, db)

        assert response == {"message": "User deleted successfully"}
        assert delete_threads, "no DELETE FROM tasks statement was observed"
        assert loop_thread_ident not in delete_threads, (
            "user deletion issued its task DELETE on the event loop thread"
        )
        assert db.query(Task).filter(Task.user_id == target_id).count() == 0
        assert db.query(User).filter(User.id == target_id).count() == 0
        assert (
            db.query(DAGExecution).filter(DAGExecution.task_id.in_(task_ids)).count()
            == 0
        )
    finally:
        event.remove(engine, "before_cursor_execute", _record)
        db.close()


@pytest.mark.asyncio
async def test_admin_user_delete_batches_child_table_deletes() -> None:
    """Child rows are cleared with one statement per table, not one per task."""

    _admin_headers()
    _register_second_user("batch-user", "batchpass1")
    db = _direct_db_session()
    engine = get_engine()
    statements: list[str] = []

    def _record(conn, cursor, statement, parameters, context, executemany) -> None:
        statements.append(" ".join(statement.split()).lower())

    admin = db.query(User).filter(User.username == "admin").one()
    target_id, _task_ids = _seed_user_with_tasks(db, "batch-user", 5)

    event.listen(engine, "before_cursor_execute", _record)
    try:
        await delete_user(target_id, admin, db)
    finally:
        event.remove(engine, "before_cursor_execute", _record)

    try:
        dag_deletes = [
            s for s in statements if s.startswith("delete from dag_executions")
        ]
        task_deletes = [s for s in statements if s.startswith("delete from tasks")]
        assert len(dag_deletes) == 1, dag_deletes
        assert len(task_deletes) == 1, task_deletes
        assert db.query(Task).filter(Task.user_id == target_id).count() == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_admin_user_delete_is_fk_safe_under_enforced_foreign_keys() -> None:
    """The batched purge must still satisfy strict foreign keys (M2 guard)."""

    _admin_headers()
    _register_second_user("fk-user", "fkpass1")
    engine = get_engine()

    def _enable_fk(dbapi_connection, connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    event.listen(engine, "connect", _enable_fk)
    # The purge runs in a worker thread on its own pooled connection, so a
    # PRAGMA issued on the request session would not reach it. Recycle the pool
    # so every connection handed out from here on enforces foreign keys.
    engine.dispose()
    db = _direct_db_session()
    try:
        assert db.execute(text("PRAGMA foreign_keys")).scalar() == 1
        admin = db.query(User).filter(User.username == "admin").one()
        target_id, task_ids = _seed_user_with_tasks(db, "fk-user", 3)

        response = await delete_user(target_id, admin, db)

        assert response == {"message": "User deleted successfully"}
        assert db.query(User).filter(User.id == target_id).count() == 0
        assert db.query(Task).filter(Task.id.in_(task_ids)).count() == 0
        assert (
            db.query(DAGExecution).filter(DAGExecution.task_id.in_(task_ids)).count()
            == 0
        )
    finally:
        event.remove(engine, "connect", _enable_fk)
        db.close()


class _NoopDeleteProvider:
    async def on_task_created(self, context, configuration) -> None:
        return None

    async def build_runtime(self, context) -> TaskRuntimeContribution:
        return TaskRuntimeContribution()

    async def public_metadata(self, context) -> dict:
        return {}

    async def on_task_deleted(self, context) -> None:
        return None


def _is_keyset_task_page_select(normalized: str) -> bool:
    """Match only the runtime-cleanup keyset page SELECT, nothing else."""

    return (
        normalized.startswith("select tasks.id as tasks_id")
        and "tasks.agent_config as tasks_agent_config" in normalized
        and " from tasks " in normalized
        and "tasks.id > " in normalized
        and "order by tasks.id" in normalized
        and " limit " in normalized
    )


@pytest.mark.asyncio
async def test_admin_user_delete_runs_keyset_pages_off_the_event_loop(
    monkeypatch,
) -> None:
    """The runtime-cleanup keyset pagination must not query on the loop thread.

    With a registered task extension the deletion walks ``tasks`` in keyset
    pages. That loop is unbounded in page count, so every page SELECT has to
    run in a worker thread just like the purge does.
    """

    _admin_headers()
    _register_second_user("keyset-user", "keysetpass1")
    register_task_extension("keyset_delete_observer", _NoopDeleteProvider())
    monkeypatch.setattr(admin_users_module, "_TASK_RUNTIME_DELETE_PAGE_SIZE", 2)
    db = _direct_db_session()
    engine = get_engine()
    loop_thread_ident = threading.get_ident()
    page_threads: list[int] = []

    def _record(conn, cursor, statement, parameters, context, executemany) -> None:
        normalized = " ".join(statement.split()).lower()
        if _is_keyset_task_page_select(normalized):
            page_threads.append(threading.get_ident())

    event.listen(engine, "before_cursor_execute", _record)
    try:
        admin = db.query(User).filter(User.username == "admin").one()
        target = db.query(User).filter(User.username == "keyset-user").one()
        target_id = int(target.id)
        bound = agent_config_with_task_extension_bindings(
            {}, ["keyset_delete_observer"]
        )
        db.add_all(
            [
                Task(
                    user_id=target_id,
                    title=f"keyset task {index}",
                    description="",
                    agent_config=dict(bound),
                )
                for index in range(5)
            ]
        )
        db.commit()

        assert asyncio.get_running_loop() is not None

        response = await delete_user(target_id, admin, db)

        assert response == {"message": "User deleted successfully"}
        # 5 tasks / page size 2 -> 3 full pages plus the terminating empty page.
        assert len(page_threads) == 4, page_threads
        on_loop = [ident for ident in page_threads if ident == loop_thread_ident]
        assert not on_loop, (
            f"{len(on_loop)} of {len(page_threads)} keyset page SELECTs ran on "
            "the event loop thread"
        )
        assert db.query(User).filter(User.id == target_id).count() == 0
    finally:
        event.remove(engine, "before_cursor_execute", _record)
        unregister_task_extension("keyset_delete_observer")
        db.close()
