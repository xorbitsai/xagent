from unittest.mock import AsyncMock

import pytest

from xagent.sandbox.base import (
    SandboxConfig,
    SandboxContractError,
    SandboxInfo,
    SandboxNotFoundError,
    SandboxTemplate,
)
from xagent.web.sandbox_manager import SandboxCapacityError, SandboxManager

DIGEST = "d" * 64
NAME = f"chrome-execution::{DIGEST}"


def _info(*, state: str = "running") -> SandboxInfo:
    return SandboxInfo(
        name=NAME,
        state=state,
        template=SandboxTemplate(type="image", image="sandbox:latest"),
        config=SandboxConfig(),
    )


def _service(*, reconciles: bool = True):
    service = AsyncMock()
    service.supports_runtime_spec.return_value = reconciles
    service.list_sandboxes.return_value = [_info()]
    return service


@pytest.mark.asyncio
async def test_capacity_never_selects_durable_chrome_as_a_victim() -> None:
    service = _service()
    manager = SandboxManager(service)
    with pytest.raises(SandboxCapacityError):
        await manager._ensure_capacity_for("user::2", cap=1)
    service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_idle_sweep_excludes_durable_chrome() -> None:
    service = _service()
    manager = SandboxManager(service)
    assert await manager.sweep_idle_sandboxes(idle_ttl=0) == []
    service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_quiesce_excludes_durable_chrome() -> None:
    service = _service(reconciles=True)
    await SandboxManager(service).cleanup()
    service.stop_existing.assert_not_awaited()
    service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_cleanup_excludes_durable_chrome() -> None:
    service = _service(reconciles=False)
    await SandboxManager(service).cleanup()
    service.get_or_create.assert_not_awaited()
    service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_explicit_delete_cannot_bypass_durable_protocol() -> None:
    service = _service()
    manager = SandboxManager(service)
    with pytest.raises(SandboxContractError, match="durable Chrome"):
        await manager.delete_sandbox("chrome-execution", DIGEST)
    service.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_strict_delete_treats_not_found_as_success() -> None:
    service = _service()
    service.delete.side_effect = SandboxNotFoundError("gone")
    await SandboxManager(service).delete_durable_sandbox_strict(DIGEST)
    service.delete.assert_awaited_once_with(NAME)


@pytest.mark.asyncio
async def test_strict_delete_propagates_backend_failure() -> None:
    service = _service()
    service.delete.side_effect = RuntimeError("backend unavailable")
    with pytest.raises(RuntimeError, match="backend unavailable"):
        await SandboxManager(service).delete_durable_sandbox_strict(DIGEST)


@pytest.mark.asyncio
async def test_strict_delete_propagates_discovery_failure() -> None:
    service = _service()
    service.list_sandboxes.side_effect = RuntimeError("list unavailable")
    with pytest.raises(RuntimeError, match="list unavailable"):
        await SandboxManager(service).delete_durable_sandbox_strict(DIGEST)


@pytest.mark.asyncio
async def test_strict_delete_removes_workers_before_primary() -> None:
    service = _service()
    worker = _info(state="stopped").model_copy(update={"name": f"{NAME}::worker::1"})
    service.list_sandboxes.return_value = [_info(), worker]
    await SandboxManager(service).delete_durable_sandbox_strict(DIGEST)
    assert [call.args[0] for call in service.delete.await_args_list] == [
        worker.name,
        NAME,
    ]
