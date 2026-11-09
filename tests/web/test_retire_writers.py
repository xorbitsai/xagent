"""``retire_writers``: the writers stop once, the process keeps serving, shutdown skips it."""

from __future__ import annotations

import pytest

from xagent.web import app as app_module


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> list[str]:
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
    return calls


async def test_retire_writers_stops_every_writer_once_and_is_idempotent(steps):
    assert not app_module.writers_retired()

    await app_module.retire_writers()
    first = list(steps)
    await app_module.retire_writers()

    assert app_module.writers_retired()
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
