import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import delete, text, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Query, sessionmaker

from tests.web.services.test_task_execution_event_store import engine as engine_fixture
from tests.web.services.test_task_execution_event_store import (
    task_id as task_id_fixture,
)
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE, CHECKPOINT_TYPE
from xagent.core.agent.trace import TraceEvent
from xagent.web.models.chat_message import TaskChatMessage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.services import task_execution as execution
from xagent.web.services import task_lease_service as leases
from xagent.web.services import task_orchestrator as orch
from xagent.web.services import trace_handlers as traces
from xagent.web.services import workforce_runtime as workforce
from xagent.web.services.uploaded_file_store import UploadedFileStore

engine = engine_fixture
task_id = task_id_fixture


def make_call(route, factory, tid, uid, lease, output):
    empty = execution._PreparedTaskFileOutputs((), (), ())
    if route == "finish":
        fn = orch.finish_turn

        def call():
            with factory() as db:
                return fn(db, tid, task_lease=lease)
    elif route == "result":
        fn = execution._finalize_task_execution_result_isolated

        def call():
            return fn(
                task_id=tid,
                task_user_id=uid,
                pre_run_status=TaskStatus.RUNNING,
                result={"status": "completed", "success": True, "output": output},
                expected_run_id=lease.run_id,
                task_lease=lease,
                resolved_scope_segments=(),
                prepared_outputs=empty,
            )
    elif route == "resume":
        fn = execution._finalize_resumed_task

        def call():
            return fn(
                tid,
                status="completed",
                success=True,
                output=output,
                task_owner_user_id=uid,
                result={"output": output},
                task_lease=lease,
                prepared_outputs=empty,
            )
    else:
        fn = workforce._sync_workforce_run_status_for_task_id

        def call():
            with factory() as db:
                return fn(db, tid, TaskStatus.RUNNING, task_lease=lease)

    return call


@pytest.mark.parametrize("route", ["finish", "result", "resume", "workforce"])
def test_actual_entrypoint_lock_and_child_visibility(
    engine, task_id, monkeypatch, route
):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row and FK locks")
    factory = sessionmaker(engine)
    monkeypatch.setattr(execution, "get_session_local", lambda: factory)
    with factory() as db:
        lease = leases.acquire_task_lease(db, task_id, runner_id="review", new_run=True)
        task = db.get(Task, task_id)
        uid = task.user_id
        if route == "finish":
            task.status = TaskStatus.COMPLETED
            db.add(
                TaskChatMessage(
                    task_id=task_id,
                    user_id=uid,
                    role="assistant",
                    message_type="assistant_response",
                    content="old answer",
                )
            )
            db.commit()
    ready, proceed = Event(), Event()
    first = Query.first

    def pause_first(query):
        result = first(query)
        if (
            result is not None
            and isinstance(result, Task)
            and not query.session.info.get("paused")
        ):
            query.session.info["paused"] = True
            ready.set()
            assert proceed.wait(8)
        return result

    monkeypatch.setattr(Query, "first", pause_first)
    output = "[report.txt](file:00000000-0000-0000-0000-000000000001)"
    call = make_call(route, factory, task_id, uid, lease, output)
    file_id = str(uuid4())
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(call)
        try:
            assert ready.wait(8)
            with factory() as contender:
                contender.execute(text("SET LOCAL lock_timeout='150ms'"))
                with pytest.raises(OperationalError, match="lock timeout"):
                    contender.execute(
                        update(Task).where(Task.id == task_id).values(runner_id="other")
                    )
                contender.rollback()
                contender.execute(text("SET LOCAL lock_timeout='150ms'"))
                with pytest.raises(OperationalError, match="lock timeout"):
                    contender.execute(delete(Task).where(Task.id == task_id))
                contender.rollback()
            with factory() as child:
                child.execute(text("SET LOCAL lock_timeout='150ms'"))
                if route == "finish":
                    child.add(
                        TaskChatMessage(
                            task_id=task_id,
                            user_id=uid,
                            role="assistant",
                            message_type="assistant_response",
                            content="new answer",
                        )
                    )

                def insert_and_commit():
                    if route != "finish":
                        UploadedFileStore(child).add_already_durable(
                            UploadedFile(
                                task_id=task_id,
                                user_id=uid,
                                file_id=file_id,
                                filename="report.txt",
                                storage_path="/review/" + file_id,
                                storage_status="available",
                                storage_key="review/" + file_id,
                                checksum="review-checksum",
                            )
                        )
                    child.commit()

                with pytest.raises(OperationalError, match="lock timeout"):
                    insert_and_commit()
                child.rollback()
        finally:
            proceed.set()
        result = future.result(timeout=8)
    with factory() as db:
        task = db.get(Task, task_id)
        if route == "finish":
            assert task.output == "old answer"
        elif route == "resume":
            assert file_id not in result["output"]
        elif route == "result":
            assert file_id not in result.ai_response


def test_checkpoint_and_settlement_complete_without_lock_upgrade(
    engine, task_id, monkeypatch
):
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL row and FK locks")
    factory = sessionmaker(engine)
    with factory() as db:
        lease = leases.acquire_task_lease(db, task_id, runner_id="review", new_run=True)
    inserted, attempting, proceed = Event(), Event(), Event()
    pids = {}
    stage = traces.stage_trace_event_row

    def paused_stage(db, *args, **kwargs):
        result = stage(db, *args, **kwargs)
        pids["checkpoint"] = db.scalar(text("select pg_backend_pid()"))
        inserted.set()
        assert proceed.wait(8)
        return result

    monkeypatch.setattr(traces, "stage_trace_event_row", paused_stage)
    lock = orch.lock_task_lease_for_settlement_no_commit

    def noticed_lock(db, lease):
        pids["finalizer"] = db.scalar(text("select pg_backend_pid()"))
        attempting.set()
        return lock(db, lease)

    monkeypatch.setattr(orch, "lock_task_lease_for_settlement_no_commit", noticed_lock)
    event = TraceEvent(
        CHECKPOINT_EVENT_TYPE,
        task_id=str(task_id),
        step_id="step",
        data={
            "checkpoint_type": CHECKPOINT_TYPE,
            "execution_id": "review",
            "snapshot": {},
        },
        require_persisted=True,
    )

    def checkpoint():
        with factory() as db, leases.bind_task_lease_context(lease):
            db.execute(text("SET LOCAL statement_timeout='5s'"))
            traces.DatabaseTraceHandler(task_id)._save_trace_event(db, event)

    def finalize():
        with factory() as db:
            db.execute(text("SET LOCAL statement_timeout='5s'"))
            return orch.finish_turn(db, task_id, task_lease=lease)

    with ThreadPoolExecutor(2) as pool:
        writer = pool.submit(checkpoint)
        try:
            assert inserted.wait(8)
            finisher = pool.submit(finalize)
            assert attempting.wait(8)
            # Release only after PostgreSQL confirms the finalizer is waiting
            # for the checkpoint transaction, rather than guessing timing.
            with factory() as observer:
                deadline = time.monotonic() + 3
                while True:
                    blockers = observer.scalar(
                        text("select pg_blocking_pids(:pid)"),
                        {"pid": pids["finalizer"]},
                    )
                    if pids["checkpoint"] in blockers:
                        break
                    assert time.monotonic() < deadline
                    time.sleep(0.005)
        finally:
            proceed.set()
        writer.result(timeout=8)
        assert finisher.result(timeout=8)
    with factory() as db:
        task = db.get(Task, task_id)
        assert task.last_checkpoint_event_id == str(event.id)
        assert task.last_checkpoint_trace_event_id is not None
        assert task.status == TaskStatus.FAILED


@pytest.mark.parametrize("route", ["finish", "result", "resume", "workforce"])
def test_stale_owner_still_rejected(engine, task_id, monkeypatch, route):
    factory = sessionmaker(engine)
    monkeypatch.setattr(execution, "get_session_local", lambda: factory)
    with factory() as db:
        old = leases.acquire_task_lease(db, task_id, runner_id="review", new_run=True)
        leases.acquire_task_lease(
            db, task_id, runner_id="review", expected_run_id=old.run_id
        )
        task = db.get(Task, task_id)
        uid = task.user_id
        before = (
            task.status,
            task.output,
            task.runner_id,
            task.run_id,
            task.lease_attempt_id,
        )
    result = make_call(route, factory, task_id, uid, old, "stale result")()
    if route == "result":
        assert result.late_result
    elif route == "resume":
        assert result["late_result"]
    else:
        assert result is False
    with factory() as db:
        task = db.get(Task, task_id)
        assert (
            task.status,
            task.output,
            task.runner_id,
            task.run_id,
            task.lease_attempt_id,
        ) == before
        assert db.query(TaskChatMessage).count() == 0
