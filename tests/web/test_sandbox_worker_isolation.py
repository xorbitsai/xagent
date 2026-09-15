"""Workers sharing SQL must not overwrite each other's sandbox handles."""

import pytest

from xagent import config
from xagent.sandbox import SandboxConfig, SandboxInfo, SandboxSnapshot, SandboxTemplate
from xagent.web.models.database import get_session_local, init_db
from xagent.web.sandbox_store import DBDockerStore


def test_worker_namespace_is_stable_and_required(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_SANDBOX_NAMESPACE", "deployment")
    monkeypatch.delenv("XAGENT_SANDBOX_WORKER_ID", raising=False)
    with pytest.raises(ValueError, match="XAGENT_SANDBOX_WORKER_ID"):
        config.get_sandbox_worker_namespace()
    monkeypatch.setenv("XAGENT_SANDBOX_WORKER_ID", "worker-1")
    first = config.get_sandbox_worker_namespace()
    assert first == config.get_sandbox_worker_namespace()
    monkeypatch.setenv("XAGENT_SANDBOX_WORKER_ID", "worker-2")
    assert config.get_sandbox_worker_namespace() != first
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")
    assert config.get_sandbox_worker_namespace() == "deployment"


def test_workers_isolate_same_logical_sandbox_and_snapshot(tmp_path, monkeypatch):
    init_db(db_url=f"sqlite:///{tmp_path / 'sandbox.db'}")
    first, second = (
        DBDockerStore(namespace="worker-1"),
        DBDockerStore(namespace="worker-2"),
    )
    for store in (first, second):
        monkeypatch.setattr(store, "_get_db_session", get_session_local())
    for store, state in ((first, "running"), (second, "stopped")):
        store.add_info(
            "task-1",
            SandboxInfo(
                name="task-1",
                state=state,
                template=SandboxTemplate(type="image", image="test"),
                config=SandboxConfig(),
            ),
        )
        store.add_snapshot(
            SandboxSnapshot(snapshot_id="snapshot-1", metadata={"state": state})
        )
    assert first.get_info("task-1").state == "running"
    assert second.get_info("task-1").state == "stopped"
    assert first.list_snapshots()[0].snapshot_id == "snapshot-1"
    assert second.list_snapshots()[0].metadata == {"state": "stopped"}
    first.delete_info("task-1")
    first.delete_snapshot("snapshot-1")
    assert first.get_info("task-1") is None
    assert first.get_snapshot("snapshot-1") is None
    assert second.get_info("task-1").name == "task-1"
    assert second.get_snapshot("snapshot-1").metadata == {"state": "stopped"}


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_runtime_secret_ttl_must_be_positive(monkeypatch, value):
    monkeypatch.setenv("XAGENT_TASK_RUNTIME_SECRETS_TTL_SECONDS", value)
    with pytest.raises(ValueError):
        config.get_task_runtime_secrets_ttl_seconds()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cleanup", "idle_sweep"])
async def test_worker_lifecycle_preserves_other_workers_live_sandbox(
    tmp_path, monkeypatch, action
):
    from unittest.mock import Mock

    import xagent.sandbox.docker_sandbox as docker_module
    from tests.sandbox.test_docker_sandbox import (
        _ClientWithCollection,
        _labeled_container,
        _LabelFilteredCollection,
    )
    from xagent.web.sandbox_manager import SandboxManager, _create_docker_service

    init_db(db_url=f"sqlite:///{tmp_path / 'workers.db'}")
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setenv("XAGENT_SANDBOX_NAMESPACE", "deployment")
    collection = _LabelFilteredCollection()
    client = _ClientWithCollection(collection)
    monkeypatch.setattr(docker_module, "_create_docker_client", lambda: client)
    managers = []
    containers = []
    stopped, deleted = [], []
    for worker in ("worker-a", "worker-b"):
        monkeypatch.setenv("XAGENT_SANDBOX_WORKER_ID", worker)
        namespace = config.get_sandbox_worker_namespace()
        service = _create_docker_service()
        assert service is not None
        managers.append(SandboxManager(service))
        container = _labeled_container(
            {
                "xagent.managed": "v2",
                "xagent.sandbox.namespace": namespace,
                "xagent.sandbox.name": "user::1",
            },
            on_stop=lambda worker=worker: stopped.append(worker),
            on_remove=lambda worker=worker: deleted.append(worker),
        )
        containers.append(container)
    collection._containers = containers
    first, second = managers
    first._lease_providers["user::1"] = Mock()
    assert await first.attach("user", "1")

    if action == "cleanup":
        # Startup and shutdown both use this public lifecycle entry point.
        await second.cleanup()
        await second.cleanup()
        assert stopped == ["worker-b", "worker-b"]
        assert deleted == []
    else:
        assert await second.sweep_idle_sandboxes(idle_ttl=0) == ["user::1"]
        assert deleted == ["worker-b"]
        assert stopped == []
    assert containers[0].attrs["State"]["Status"] == "running"
    assert first.ref_count("user", "1") == 1
