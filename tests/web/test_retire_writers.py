"""``retire_writers``: the writers stop once, the process keeps serving, shutdown skips it."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from xagent.web import app as app_module


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    calls: list[str] = []

    def stopper(name: str):
        async def stop(*_args, **_kwargs) -> None:
            calls.append(name)

        return stop

    for name in (
        "stop_orphan_upload_gc_task",
        "stop_composio_reconcile_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
        "stop_task_auto_resume_task",
        "stop_task_lease_recovery_task",
    ):
        monkeypatch.setattr(app_module, name, stopper(name))
    monkeypatch.setattr(app_module, "_writers_retired", False)
    monkeypatch.setattr(app_module, "_task_command_dispatcher_task", None)
    monkeypatch.setattr(app_module, "_sandbox_idle_sweep_task", None)
    monkeypatch.setattr(app_module, "_file_storage_startup_sync_task", None)
    monkeypatch.setattr(app_module, "_trigger_dispatcher_task", None)
    monkeypatch.setattr(app_module, "_migration_task", None)
    monkeypatch.setattr(app_module, "_retire_writers_lock", None)
    yield calls
    app_module.release_hold.reset_writers_retired()


async def test_retire_writers_stops_every_writer_once_and_is_idempotent(steps):
    assert not app_module.writers_retired()

    await app_module.retire_writers()
    first = list(steps)
    await app_module.retire_writers()

    assert app_module.writers_retired()
    # Socket writes that remain are refused from here on.
    assert app_module.release_hold.writers_retired
    assert first == [
        "stop_orphan_upload_gc_task",
        "stop_composio_reconcile_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
        "stop_task_auto_resume_task",
        "stop_task_lease_recovery_task",
    ]
    assert steps == first


async def test_a_step_that_fails_leaves_the_retire_unfinished_so_it_can_continue(
    steps, monkeypatch
):
    async def boom(*_a, **_k) -> None:
        raise RuntimeError("stop failed")

    monkeypatch.setattr(app_module, "stop_task_auto_resume_task", boom)
    with pytest.raises(RuntimeError):
        await app_module.retire_writers()
    assert not app_module.writers_retired()

    async def fine(*_a, **_k) -> None:
        steps.append("auto_resume")

    monkeypatch.setattr(app_module, "stop_task_auto_resume_task", fine)
    await app_module.retire_writers()

    assert app_module.writers_retired()
    assert steps[-2:] == ["auto_resume", "stop_task_lease_recovery_task"]


# -- socket writes are refused once the writers are retired -------------------


@pytest.fixture
def retired_hold():
    from xagent.web.services.release_hold import release_hold

    release_hold.mark_writers_retired()
    yield release_hold
    release_hold.reset_writers_retired()


def test_the_retired_flag_is_one_way_until_a_new_lifespan():
    from xagent.web.services.release_hold import release_hold

    assert not release_hold.writers_retired
    release_hold.mark_writers_retired()
    try:
        release_hold.reopen()
        assert release_hold.writers_retired
    finally:
        release_hold.reset_writers_retired()
    assert not release_hold.writers_retired


async def test_a_socket_command_is_refused_with_the_retryable_paused_error(
    retired_hold,
):
    from xagent.web.api import websocket as ws
    from xagent.web.services.client_error_messages import ClientErrorCode
    from xagent.web.services.task_command_transport import TaskCommandKind

    class _User:
        id = 1
        is_admin = False

    with pytest.raises(ws.ClientVisibleValidationError) as caught:
        await ws._enqueue_websocket_task_command(
            task_id=1, message_data={"user": _User()}, kind=TaskCommandKind.PAUSE
        )

    assert caught.value.error_code is ClientErrorCode.EXECUTION_PAUSED


async def test_execute_task_is_refused_before_it_acknowledges_or_stages(retired_hold):
    from xagent.web.api import websocket as ws

    sent: list[dict] = []

    class _User:
        id = 1
        is_admin = False

    async def record(message, _websocket):
        sent.append(message)

    original = ws.manager.send_personal_message
    ws.manager.send_personal_message = record  # type: ignore[assignment]
    try:
        await ws.handle_execute_task(object(), 1, {"user": _User()})  # type: ignore[arg-type]
    finally:
        ws.manager.send_personal_message = original  # type: ignore[assignment]

    assert all(m.get("type") != "execution_started" for m in sent)
    assert any(m.get("error_code") == "execution_paused" for m in sent)
