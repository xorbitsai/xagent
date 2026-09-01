"""Tests for the workforce external-deployment foundation (#946).

Covers: run snapshot config fingerprint + turn-entry drift/archive guards,
``create_workforce_run`` source threading and idempotency, and archive
terminating in-flight runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.models.workforce import Workforce, WorkforceAgent, WorkforceRun
from xagent.web.services import workforce_runs as workforce_runs_service
from xagent.web.services.task_orchestrator import (
    TaskTurnError,
    TaskTurnOrchestrator,
    TaskTurnPayload,
    TurnKind,
)
from xagent.web.services.workforce_runtime import (
    WorkforceRunPauseTarget,
    WorkforceTurnRejectedError,
    ensure_workforce_turn_allowed,
    pause_workforce_tasks_after_archive,
    sync_workforce_run_status,
)
from xagent.web.services.workforce_snapshot import (
    build_workforce_snapshot,
    compute_live_workforce_config_fingerprint,
)

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")


def _user_id(username: str = "admin") -> int:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == username).one()
        return int(user.id)
    finally:
        db.close()


def _create_published_agent(user_id: int, name: str) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _create_active_workforce(name: str = "Deploy Workforce") -> int:
    headers = _admin_headers()
    owner_id = _user_id()
    manager_agent_id = _create_published_agent(owner_id, f"{name} Manager")
    worker_agent_id = _create_published_agent(owner_id, f"{name} Worker")
    response = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "Coordinates deployment tests",
            "manager_agent_id": manager_agent_id,
            "workers": [
                {
                    "source_type": "existing",
                    "agent_id": worker_agent_id,
                    "alias": "worker-1",
                    "assignment_instructions": "Handle everything",
                    "enabled": True,
                    "sort_order": 1,
                }
            ],
        },
    )
    assert response.status_code == 200, response.text
    workforce_id = int(response.json()["id"])
    published = client.post(f"/api/workforces/{workforce_id}/publish", headers=headers)
    assert published.status_code == 200, published.text
    return workforce_id


def _load_workforce(db: Any, workforce_id: int) -> Workforce:
    workforce = (
        db.query(Workforce)
        .options(
            selectinload(Workforce.manager_agent),
            selectinload(Workforce.workers).selectinload(WorkforceAgent.agent),
        )
        .filter(Workforce.id == workforce_id)
        .one()
    )
    return workforce


def _build_snapshot(workforce_id: int) -> dict[str, Any]:
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        workforce = _load_workforce(db, workforce_id)
        return build_workforce_snapshot(db, user, workforce)
    finally:
        db.close()


def _create_workforce_task_and_run(
    workforce_id: int,
    *,
    snapshot: dict[str, Any],
    task_status: TaskStatus = TaskStatus.COMPLETED,
    is_preview: bool = False,
    # "completed" is the realistic default: it's a workforce conversation's
    # resting state between turns (see ensure_workforce_turn_allowed's
    # "cancelled" check, PR #1060 review F2), not a rejection. Callers
    # testing something status-specific (e.g. an actually-cancelled run)
    # override this explicitly.
    run_status: str = "completed",
) -> tuple[int, int]:
    db = _direct_db_session()
    try:
        user_id = int(db.query(User).filter(User.username == "admin").one().id)
        workforce = db.query(Workforce).filter(Workforce.id == workforce_id).one()
        task = Task(
            user_id=user_id,
            title="Workforce session",
            description="hello",
            status=task_status,
            agent_id=int(workforce.manager_agent_id),
            source="internal",
        )
        db.add(task)
        db.flush()
        run = WorkforceRun(
            workforce_id=workforce_id,
            task_id=int(task.id),
            user_id=user_id,
            status=run_status,
            is_preview=is_preview,
            snapshot=snapshot,
        )
        db.add(run)
        db.flush()
        task.agent_config = {"workforce_run_id": int(run.id)}
        db.commit()
        return int(task.id), int(run.id)
    finally:
        db.close()


def _change_worker_agent_instructions(workforce_id: int) -> None:
    db = _direct_db_session()
    try:
        workforce = _load_workforce(db, workforce_id)
        worker_agent = workforce.workers[0].agent
        worker_agent.instructions = "Completely different live instructions"
        db.commit()
    finally:
        db.close()


# ===== Fingerprint =====


def test_snapshot_pins_fingerprint_matching_live_state() -> None:
    workforce_id = _create_active_workforce("Fingerprint Workforce")
    snapshot = _build_snapshot(workforce_id)

    fingerprint = snapshot.get("config_fingerprint")
    assert isinstance(fingerprint, str) and len(fingerprint) == 64

    db = _direct_db_session()
    try:
        workforce = _load_workforce(db, workforce_id)
        assert compute_live_workforce_config_fingerprint(workforce) == fingerprint
    finally:
        db.close()


def test_live_fingerprint_changes_when_worker_agent_drifts() -> None:
    workforce_id = _create_active_workforce("Drift Workforce")
    snapshot = _build_snapshot(workforce_id)

    _change_worker_agent_instructions(workforce_id)

    db = _direct_db_session()
    try:
        workforce = _load_workforce(db, workforce_id)
        live = compute_live_workforce_config_fingerprint(workforce)
        assert live != snapshot["config_fingerprint"]
    finally:
        db.close()


# ===== Turn-entry guard =====


def test_turn_guard_is_noop_for_non_workforce_tasks() -> None:
    _admin_headers()  # ensure the admin user exists
    db = _direct_db_session()
    try:
        user_id = int(db.query(User).filter(User.username == "admin").one().id)
        task = Task(
            user_id=user_id,
            title="Plain task",
            description="hello",
            status=TaskStatus.COMPLETED,
            source="internal",
        )
        db.add(task)
        db.commit()
        ensure_workforce_turn_allowed(
            db, task_id=int(task.id), task_owner_user_id=user_id
        )
    finally:
        db.close()


def test_turn_guard_rejects_missing_run_row() -> None:
    workforce_id = _create_active_workforce("Missing Run Workforce")
    snapshot = _build_snapshot(workforce_id)
    task_id, run_id = _create_workforce_task_and_run(workforce_id, snapshot=snapshot)

    db = _direct_db_session()
    try:
        db.query(WorkforceRun).filter(WorkforceRun.id == run_id).delete()
        db.commit()
        with pytest.raises(WorkforceTurnRejectedError) as excinfo:
            ensure_workforce_turn_allowed(
                db, task_id=task_id, task_owner_user_id=_user_id()
            )
        assert excinfo.value.reason == "workforce_run_not_found"
    finally:
        db.close()


def test_turn_guard_rejects_cancelled_run() -> None:
    """PR #1060 review, F2: a run cancelled by the archive path or the
    stale-preview-run reaper flips WorkforceRun.status to "cancelled" but
    that alone enforces nothing -- before this fix, a RESUME (or any new
    turn) on an already-cancelled run would proceed indefinitely since no
    turn-entry check inspected the run's own status."""
    workforce_id = _create_active_workforce("Cancelled Run Guard Workforce")
    snapshot = _build_snapshot(workforce_id)
    task_id, run_id = _create_workforce_task_and_run(workforce_id, snapshot=snapshot)

    db = _direct_db_session()
    try:
        db.query(WorkforceRun).filter(WorkforceRun.id == run_id).update(
            {"status": "cancelled"}
        )
        db.commit()
        with pytest.raises(WorkforceTurnRejectedError) as excinfo:
            ensure_workforce_turn_allowed(
                db, task_id=task_id, task_owner_user_id=_user_id()
            )
        assert excinfo.value.reason == "workforce_run_not_active"
    finally:
        db.close()


def test_turn_guard_allows_a_normal_completed_run_to_continue() -> None:
    """Self-review follow-up on F2: the "cancelled" check must NOT reject
    "completed"/"failed" -- that's the run's ordinary resting state between
    turns (every turn ends by projecting WorkforceRun.status to "completed"
    or "failed", see sync_workforce_run_status / task_orchestrator.py's
    _APPENDABLE_STATUSES), not evidence of a cancellation. An earlier,
    broader version of this check (rejecting anything not in
    ACTIVE_WORKFORCE_RUN_STATUSES) would have broken every second-and-later
    message in every workforce conversation."""
    workforce_id = _create_active_workforce("Completed Run Continues Workforce")
    snapshot = _build_snapshot(workforce_id)
    task_id, run_id = _create_workforce_task_and_run(
        workforce_id, snapshot=snapshot, run_status="completed"
    )

    db = _direct_db_session()
    try:
        # No exception: a completed run must be allowed to continue.
        ensure_workforce_turn_allowed(
            db, task_id=task_id, task_owner_user_id=_user_id()
        )

        db.query(WorkforceRun).filter(WorkforceRun.id == run_id).update(
            {"status": "failed"}
        )
        db.commit()
        ensure_workforce_turn_allowed(
            db, task_id=task_id, task_owner_user_id=_user_id()
        )
    finally:
        db.close()


def test_turn_guard_prefers_archived_reason_over_cancelled_when_both_apply() -> None:
    """Self-review follow-up on F2: the real archive endpoint cancels every
    in-flight run of a workforce in the SAME transaction it flips
    Workforce.status to "archived" (see cancel_active_workforce_runs), so
    this exact combination -- a cancelled run whose workforce is also
    archived -- is the primary real-world case the "cancelled" check must
    handle. The more specific, already-established "workforce_archived"
    reason (its own stable v1 error code/WS message) must still win, not
    the newer, less specific "workforce_run_not_active"."""
    workforce_id = _create_active_workforce("Archived And Cancelled Workforce")
    snapshot = _build_snapshot(workforce_id)
    task_id, run_id = _create_workforce_task_and_run(workforce_id, snapshot=snapshot)

    db = _direct_db_session()
    try:
        db.query(Workforce).filter(Workforce.id == workforce_id).update(
            {"status": "archived"}
        )
        db.query(WorkforceRun).filter(WorkforceRun.id == run_id).update(
            {"status": "cancelled"}
        )
        db.commit()
        with pytest.raises(WorkforceTurnRejectedError) as excinfo:
            ensure_workforce_turn_allowed(
                db, task_id=task_id, task_owner_user_id=_user_id()
            )
        assert excinfo.value.reason == "workforce_archived"
    finally:
        db.close()


def test_turn_guard_rejects_archived_workforce() -> None:
    workforce_id = _create_active_workforce("Archived Guard Workforce")
    snapshot = _build_snapshot(workforce_id)
    task_id, _run_id = _create_workforce_task_and_run(workforce_id, snapshot=snapshot)

    db = _direct_db_session()
    try:
        db.query(Workforce).filter(Workforce.id == workforce_id).update(
            {"status": "archived"}
        )
        db.commit()
        with pytest.raises(WorkforceTurnRejectedError) as excinfo:
            ensure_workforce_turn_allowed(
                db, task_id=task_id, task_owner_user_id=_user_id()
            )
        assert excinfo.value.reason == "workforce_archived"
    finally:
        db.close()


def test_turn_guard_rejects_config_drift_but_allows_preview_and_legacy() -> None:
    workforce_id = _create_active_workforce("Drift Guard Workforce")
    snapshot = _build_snapshot(workforce_id)
    task_id, _ = _create_workforce_task_and_run(workforce_id, snapshot=snapshot)
    preview_task_id, _ = _create_workforce_task_and_run(
        workforce_id, snapshot=snapshot, is_preview=True
    )
    legacy_snapshot = {
        key: value for key, value in snapshot.items() if key != "config_fingerprint"
    }
    legacy_task_id, _ = _create_workforce_task_and_run(
        workforce_id, snapshot=legacy_snapshot
    )

    _change_worker_agent_instructions(workforce_id)

    db = _direct_db_session()
    try:
        user_id = _user_id()
        with pytest.raises(WorkforceTurnRejectedError) as excinfo:
            ensure_workforce_turn_allowed(
                db, task_id=task_id, task_owner_user_id=user_id
            )
        assert excinfo.value.reason == "workforce_config_changed"

        # Preview sessions follow the builder's live edits by design.
        ensure_workforce_turn_allowed(
            db, task_id=preview_task_id, task_owner_user_id=user_id
        )
        # Runs created before fingerprints existed stay appendable.
        ensure_workforce_turn_allowed(
            db, task_id=legacy_task_id, task_owner_user_id=user_id
        )
    finally:
        db.close()


async def test_append_turn_maps_guard_rejection_to_task_turn_error() -> None:
    workforce_id = _create_active_workforce("Append Guard Workforce")
    snapshot = _build_snapshot(workforce_id)
    task_id, _ = _create_workforce_task_and_run(workforce_id, snapshot=snapshot)

    db = _direct_db_session()
    try:
        db.query(Workforce).filter(Workforce.id == workforce_id).update(
            {"status": "archived"}
        )
        db.commit()
    finally:
        db.close()

    with pytest.raises(TaskTurnError) as excinfo:
        await TaskTurnOrchestrator.begin_turn(
            task_id=task_id,
            task_owner_user_id=_user_id(),
            payload=TaskTurnPayload(transcript_message="follow-up"),
            kind=TurnKind.APPEND,
        )
    assert excinfo.value.reason == "workforce_archived"

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        # The rejection fired before the atomic claim: the task never left
        # its appendable status.
        assert task.status == TaskStatus.COMPLETED
    finally:
        db.close()


# ===== create_workforce_run: source + idempotency =====


def _stub_turn_started() -> SimpleNamespace:
    return SimpleNamespace(background_task=None)


async def test_create_workforce_run_threads_source_and_idempotency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_active_workforce("Source Workforce")

    async def _stub_schedule_claimed_turn(**_kwargs: Any) -> SimpleNamespace:
        return _stub_turn_started()

    monkeypatch.setattr(
        workforce_runs_service.TaskTurnOrchestrator,
        "schedule_claimed_create_turn",
        _stub_schedule_claimed_turn,
    )

    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        workforce = _load_workforce(db, workforce_id)
        result = await workforce_runs_service.create_workforce_run(
            db,
            user,
            workforce,
            message="hello from the widget",
            source="widget",
            idempotency_key="retry-abc",
        )
        task = db.get(Task, int(result.task.id))
        run = db.get(WorkforceRun, int(result.workforce_run.id))
        assert task is not None
        assert run is not None
        assert result.created is True
        assert task.source == "widget"
        assert run.idempotency_key == "retry-abc"
        first_run_id = int(result.workforce_run.id)

        replay = await workforce_runs_service.create_workforce_run(
            db,
            user,
            workforce,
            message="hello again (retried)",
            source="widget",
            idempotency_key="retry-abc",
        )
        assert replay.created is False
        assert replay.background_task is None
        assert int(replay.workforce_run.id) == first_run_id
    finally:
        db.close()


async def test_idempotency_replay_with_deleted_task_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    workforce_id = _create_active_workforce("Deleted Task Workforce")

    async def _stub_schedule_claimed_turn(**_kwargs: Any) -> SimpleNamespace:
        return _stub_turn_started()

    monkeypatch.setattr(
        workforce_runs_service.TaskTurnOrchestrator,
        "schedule_claimed_create_turn",
        _stub_schedule_claimed_turn,
    )

    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        workforce = _load_workforce(db, workforce_id)
        result = await workforce_runs_service.create_workforce_run(
            db,
            user,
            workforce,
            message="hello",
            idempotency_key="orphan-key",
        )
        run_id = int(result.workforce_run.id)
        db.query(WorkforceRun).filter(WorkforceRun.id == run_id).update(
            {"task_id": None}
        )
        db.commit()

        with pytest.raises(HTTPException) as excinfo:
            await workforce_runs_service.create_workforce_run(
                db,
                user,
                workforce,
                message="hello again",
                idempotency_key="orphan-key",
            )
        assert excinfo.value.status_code == 409
    finally:
        db.close()


async def test_create_workforce_run_defaults_to_internal_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workforce_id = _create_active_workforce("Default Source Workforce")

    async def _stub_schedule_claimed_turn(**_kwargs: Any) -> SimpleNamespace:
        return _stub_turn_started()

    monkeypatch.setattr(
        workforce_runs_service.TaskTurnOrchestrator,
        "schedule_claimed_create_turn",
        _stub_schedule_claimed_turn,
    )

    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        workforce = _load_workforce(db, workforce_id)
        result = await workforce_runs_service.create_workforce_run(
            db, user, workforce, message="hello"
        )
        task = db.get(Task, int(result.task.id))
        run = db.get(WorkforceRun, int(result.workforce_run.id))
        assert task is not None
        assert run is not None
        assert task.source == "internal"
        assert run.idempotency_key is None
    finally:
        db.close()


def test_runs_history_serializes_task_source() -> None:
    workforce_id = _create_active_workforce("Source History Workforce")
    snapshot = _build_snapshot(workforce_id)
    _create_workforce_task_and_run(workforce_id, snapshot=snapshot)

    listed = client.get(
        f"/api/workforces/{workforce_id}/runs", headers=_admin_headers()
    )
    assert listed.status_code == 200, listed.text
    items = listed.json()["items"]
    assert items and items[0]["source"] == "internal"


# ===== Archive terminates active runs =====


def test_archive_cancels_active_runs_and_keeps_terminal_ones() -> None:
    workforce_id = _create_active_workforce("Archive Workforce")
    snapshot = _build_snapshot(workforce_id)
    running_task_id, running_run_id = _create_workforce_task_and_run(
        workforce_id,
        snapshot=snapshot,
        task_status=TaskStatus.RUNNING,
        run_status="running",
    )
    _completed_task_id, completed_run_id = _create_workforce_task_and_run(
        workforce_id, snapshot=snapshot, run_status="completed"
    )

    archived = client.delete(
        f"/api/workforces/{workforce_id}", headers=_admin_headers()
    )
    assert archived.status_code == 200, archived.text
    assert archived.json()["status"] == "archived"

    db = _direct_db_session()
    try:
        running_run = (
            db.query(WorkforceRun).filter(WorkforceRun.id == running_run_id).one()
        )
        assert running_run.status == "cancelled"
        assert running_run.completed_at is not None

        completed_run = (
            db.query(WorkforceRun).filter(WorkforceRun.id == completed_run_id).one()
        )
        assert completed_run.status == "completed"

        # The live task received a durable PAUSE command. Keyed on run_id
        # alone (not the "archive" reason) so that a concurrent reap-sweep
        # dispatch for the same run would collide with this one instead of
        # creating a second, independent PAUSE command -- see
        # pause_workforce_tasks_after_archive's command_id docstring.
        command = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.task_id == running_task_id,
                TaskExecutionCommand.command_id == f"workforce-pause-{running_run_id}",
            )
            .first()
        )
        assert command is not None
        assert str(command.kind) == "pause"

        # A late task-status projection (the PAUSE landing) must not
        # resurrect the cancelled run.
        task = db.query(Task).filter(Task.id == running_task_id).one()
        assert sync_workforce_run_status(db, task, TaskStatus.PAUSED) is False
        db.refresh(running_run)
        assert running_run.status == "cancelled"
    finally:
        db.close()


def test_permanent_delete_cancels_active_runs_and_dispatches_pause() -> None:
    """Permanent-delete counterpart of
    test_archive_cancels_active_runs_and_keeps_terminal_ones -- every
    existing permanent-delete test in test_workforces_api.py only ever
    creates a "completed" run, which ACTIVE_WORKFORCE_RUN_STATUSES excludes,
    so cancel_active_workforce_runs and the PAUSE dispatch had zero coverage
    on this path before this test."""
    workforce_id = _create_active_workforce("Permanent Delete Workforce")
    snapshot = _build_snapshot(workforce_id)
    running_task_id, running_run_id = _create_workforce_task_and_run(
        workforce_id,
        snapshot=snapshot,
        task_status=TaskStatus.RUNNING,
        run_status="running",
    )
    completed_task_id, completed_run_id = _create_workforce_task_and_run(
        workforce_id, snapshot=snapshot, run_status="completed"
    )

    deleted = client.delete(
        f"/api/workforces/{workforce_id}?permanent=true", headers=_admin_headers()
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"id": workforce_id, "status": "deleted"}

    db = _direct_db_session()
    try:
        assert db.get(Workforce, workforce_id) is None
        # Unlike archive, the run rows themselves are cascade-deleted --
        # nothing left to assert "cancelled" status on.
        assert db.query(WorkforceRun).filter_by(id=running_run_id).first() is None
        assert db.query(WorkforceRun).filter_by(id=completed_run_id).first() is None

        # The live task still received a durable PAUSE command, keyed on
        # run_id exactly like archive's dispatch.
        command = (
            db.query(TaskExecutionCommand)
            .filter(
                TaskExecutionCommand.task_id == running_task_id,
                TaskExecutionCommand.command_id == f"workforce-pause-{running_run_id}",
            )
            .first()
        )
        assert command is not None
        assert str(command.kind) == "pause"

        # Tasks are never hard-deleted (other subsystems key off the task
        # id), but a "permanently deleted" workforce's conversations must
        # stop surfacing in history/search.
        running_task = db.query(Task).filter(Task.id == running_task_id).one()
        assert running_task.is_visible is False
        completed_task = db.query(Task).filter(Task.id == completed_task_id).one()
        assert completed_task.is_visible is False

        # A late task-status projection (the PAUSE landing) has nothing left
        # to resurrect -- the run row is gone, not just cancelled.
        assert sync_workforce_run_status(db, running_task, TaskStatus.PAUSED) is False
    finally:
        db.close()


async def test_pause_dispatch_dedupes_when_archive_and_reap_race_the_same_run() -> None:
    """Self-review follow-up on F5: widening the stale-preview-run reaper's
    filter to ``is_preview IS TRUE`` means it can now select an edit-mode
    preview run of an already-saved, currently-being-archived workforce --
    the same row the archive endpoint's own cancel-and-pause path just
    handled. Neither path locks the row, so both can independently build a
    PAUSE dispatch for it. The command_id is deliberately keyed on run_id
    alone (not the "archive"/"preview-reap" reason label) so that
    enqueue_task_command's own (task_id, command_id) idempotency collapses
    the second dispatch into a no-op instead of creating a second,
    independent PAUSE command."""
    workforce_id = _create_active_workforce("Racing Pause Dispatch Workforce")
    snapshot = _build_snapshot(workforce_id)
    running_task_id, running_run_id = _create_workforce_task_and_run(
        workforce_id,
        snapshot=snapshot,
        task_status=TaskStatus.RUNNING,
        run_status="running",
    )

    archived = client.delete(
        f"/api/workforces/{workforce_id}", headers=_admin_headers()
    )
    assert archived.status_code == 200, archived.text

    # Simulate the reap sweep independently building its own pause target
    # for the SAME run, as if it raced the archive above (its own selection
    # query would find the run before the archive's cancellation commits).
    await pause_workforce_tasks_after_archive(
        [
            WorkforceRunPauseTarget(
                run_id=running_run_id,
                task_id=running_task_id,
                actor_user_id=_user_id(),
            )
        ],
        reason="preview-reap",
    )

    db = _direct_db_session()
    try:
        commands = (
            db.query(TaskExecutionCommand)
            .filter(TaskExecutionCommand.task_id == running_task_id)
            .all()
        )
        pause_commands = [c for c in commands if str(c.kind) == "pause"]
        assert len(pause_commands) == 1
        assert pause_commands[0].command_id == f"workforce-pause-{running_run_id}"
    finally:
        db.close()


# ===== Review follow-ups (#952 re-review) =====


def test_fingerprint_is_insensitive_to_list_order() -> None:
    workforce_id = _create_active_workforce("List Order Workforce")

    db = _direct_db_session()
    try:
        workforce = _load_workforce(db, workforce_id)
        worker_agent = workforce.workers[0].agent
        worker_agent.knowledge_bases = ["kb-b", "kb-a"]
        worker_agent.skills = ["skill-2", "skill-1"]
        worker_agent.tool_categories = ["web", "files"]
        db.commit()
        before = compute_live_workforce_config_fingerprint(workforce)

        # Re-saving the same sets in a different click order must not change
        # the fingerprint (would otherwise force-reject in-flight sessions).
        worker_agent.knowledge_bases = ["kb-a", "kb-b"]
        worker_agent.skills = ["skill-1", "skill-2"]
        worker_agent.tool_categories = ["files", "web"]
        db.commit()
        db.expire_all()
        workforce = _load_workforce(db, workforce_id)
        assert compute_live_workforce_config_fingerprint(workforce) == before
    finally:
        db.close()


async def test_idempotency_concurrent_insert_falls_back_to_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the ``except IntegrityError`` fallback: the pre-insert lookup
    misses (simulating a concurrent inserter that has not committed yet), the
    insert then trips the unique index, and the loser returns the winner's
    run instead of surfacing a 500."""
    workforce_id = _create_active_workforce("Race Workforce")

    async def _stub_schedule_claimed_turn(**_kwargs: Any) -> SimpleNamespace:
        return _stub_turn_started()

    monkeypatch.setattr(
        workforce_runs_service.TaskTurnOrchestrator,
        "schedule_claimed_create_turn",
        _stub_schedule_claimed_turn,
    )

    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        workforce = _load_workforce(db, workforce_id)
        winner = await workforce_runs_service.create_workforce_run(
            db,
            user,
            workforce,
            message="winner",
            idempotency_key="race-key",
        )
        winner_run_id = int(winner.workforce_run.id)

        real_replay = workforce_runs_service._replay_existing_run_by_idempotency_key
        calls = {"n": 0}

        def _miss_first_then_real(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                return None  # pre-insert lookup misses: race window
            return real_replay(*args, **kwargs)

        monkeypatch.setattr(
            workforce_runs_service,
            "_replay_existing_run_by_idempotency_key",
            _miss_first_then_real,
        )

        loser = await workforce_runs_service.create_workforce_run(
            db,
            user,
            workforce,
            message="loser (concurrent retry)",
            idempotency_key="race-key",
        )
        assert calls["n"] >= 2  # IntegrityError fallback re-queried
        assert loser.created is False
        assert loser.background_task is None
        assert int(loser.workforce_run.id) == winner_run_id
    finally:
        db.close()


def test_deployment_model_defaults_and_owner_uniqueness() -> None:
    from xagent.web.models.deployment import Deployment, DeploymentOwnerType

    _admin_headers()  # initialize the test DB/app
    db = _direct_db_session()
    try:
        agent_row = Deployment(owner_type=DeploymentOwnerType.AGENT.value, owner_id=1)
        # Same owner_id under a different owner_type must coexist.
        workforce_row = Deployment(
            owner_type=DeploymentOwnerType.WORKFORCE.value, owner_id=1
        )
        db.add_all([agent_row, workforce_row])
        db.commit()

        assert agent_row.widget_enabled is False
        assert agent_row.share_enabled is False
        assert agent_row.widget_key is None
        assert agent_row.share_token is None
        assert agent_row.created_at is not None

        duplicate = Deployment(
            owner_type=DeploymentOwnerType.WORKFORCE.value, owner_id=1
        )
        db.add(duplicate)
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
    finally:
        db.close()


async def test_ws_append_rejection_surfaces_workforce_code() -> None:
    """Transport-boundary pin: the websocket path must surface stable
    non-retryable workforce guidance instead of the generic (and misleading)
    "task busy" message — an archived workforce can never be retried into
    success."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from xagent.web.api.websocket import handle_chat_message

    workforce_id = _create_active_workforce("WS Reason Workforce")
    snapshot = _build_snapshot(workforce_id)
    task_id, _ = _create_workforce_task_and_run(workforce_id, snapshot=snapshot)

    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        db.query(Workforce).filter(Workforce.id == workforce_id).update(
            {"status": "archived"}
        )
        db.commit()

        ws_manager = MagicMock(
            broadcast_to_task=AsyncMock(), send_personal_message=AsyncMock()
        )
        errors: list[dict[str, Any]] = []
        with patch("xagent.web.api.websocket.manager", ws_manager):
            await handle_chat_message(
                MagicMock(),
                task_id,
                {"message": "follow-up", "user": user, "files": []},
            )
            # The durable command may detach past the 50ms dispatch window;
            # wait for the agent_error broadcast like the owner-actor tests.
            for _ in range(200):
                errors = [
                    call.args[0]
                    for call in ws_manager.broadcast_to_task.await_args_list
                    if call.args and call.args[0].get("type") == "agent_error"
                ]
                if errors:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("agent_error broadcast did not arrive in time")

        assert errors[0]["message"] == (
            "This workforce has been archived. Unarchive and publish it before "
            "starting a new conversation, or select an active workforce."
        )
        assert errors[0]["error_code"] == "workforce_archived"
        assert "busy" not in errors[0]["message"]
    finally:
        db.close()


async def test_create_turn_guard_rejects_archived_workforce() -> None:
    """The last-line guard: an archive committing between
    ``create_workforce_run``'s commit and ``begin_turn``'s claim (the sweep
    cancels the run but cannot stop a turn that never started) must reject
    the CREATE turn instead of executing on the archived workforce."""
    workforce_id = _create_active_workforce("Create Guard Workforce")
    snapshot = _build_snapshot(workforce_id)
    task_id, _ = _create_workforce_task_and_run(
        workforce_id,
        snapshot=snapshot,
        task_status=TaskStatus.PENDING,
        run_status="pending",
    )

    db = _direct_db_session()
    try:
        db.query(Workforce).filter(Workforce.id == workforce_id).update(
            {"status": "archived"}
        )
        db.commit()
    finally:
        db.close()

    with pytest.raises(TaskTurnError) as excinfo:
        await TaskTurnOrchestrator.begin_turn(
            task_id=task_id,
            task_owner_user_id=_user_id(),
            payload=TaskTurnPayload(transcript_message="first message"),
            kind=TurnKind.CREATE,
        )
    assert excinfo.value.reason == "workforce_archived"

    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        assert task.status == TaskStatus.PENDING  # claim rolled back
    finally:
        db.close()


def test_fingerprint_version_mismatch_exempts_old_pins() -> None:
    """A fingerprint algorithm change must not spuriously reject runs pinned
    under the previous algorithm: only same-version pins are compared."""
    workforce_id = _create_active_workforce("Version Exempt Workforce")
    snapshot = _build_snapshot(workforce_id)

    # Simulate a run pinned by an older algorithm: wrong version + a value
    # the current algorithm would never produce.
    old_snapshot = dict(snapshot)
    old_snapshot["config_fingerprint"] = "0" * 64
    old_snapshot["config_fingerprint_version"] = 1
    task_id, _ = _create_workforce_task_and_run(workforce_id, snapshot=old_snapshot)

    db = _direct_db_session()
    try:
        # Exempt despite the mismatched value.
        ensure_workforce_turn_allowed(
            db, task_id=task_id, task_owner_user_id=_user_id()
        )
    finally:
        db.close()
