"""Test SandboxManager.cleanup — delete sandbox if config changed."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from xagent.sandbox.base import SandboxConfig, SandboxInfo, SandboxTemplate
from xagent.web.sandbox_manager import SandboxManager


def _make_sb_info(
    name: str,
    *,
    image: str = "img:v1",
    cpus: int = 1,
    memory: int = 512,
    volumes: list[tuple[str, str, str]] | None = None,
    state: str = "running",
) -> SandboxInfo:
    """Helper to build a SandboxInfo for testing."""
    return SandboxInfo(
        name=name,
        state=state,
        template=SandboxTemplate(type="image", image=image),
        config=SandboxConfig(cpus=cpus, memory=memory, volumes=volumes),
    )


@pytest.fixture
def service() -> AsyncMock:
    svc = AsyncMock()
    svc.delete = AsyncMock()
    svc.get_or_create = AsyncMock()
    return svc


@pytest.fixture
def manager(service: AsyncMock) -> SandboxManager:
    return SandboxManager(service)


@pytest.mark.asyncio
async def test_cleanup_deletes_on_image_change(
    manager: SandboxManager, service: AsyncMock
):
    """Sandbox with stale image should be deleted."""
    sb = _make_sb_info("user::1", image="old:v0")

    service.list_sandboxes.return_value = [sb]

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "new:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "512"},
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::1")


@pytest.mark.asyncio
async def test_cleanup_deletes_on_cpus_change(
    manager: SandboxManager, service: AsyncMock
):
    """Sandbox with different cpus should be deleted."""
    sb = _make_sb_info("user::2", image="img:v1", cpus=1)

    service.list_sandboxes.return_value = [sb]

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "4", "SANDBOX_MEMORY": "512"},
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::2")


@pytest.mark.asyncio
async def test_cleanup_deletes_on_memory_change(
    manager: SandboxManager, service: AsyncMock
):
    """Sandbox with different memory should be deleted."""
    sb = _make_sb_info("user::3", image="img:v1", memory=512)

    service.list_sandboxes.return_value = [sb]

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "1024"},
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::3")


@pytest.mark.asyncio
async def test_cleanup_deletes_on_volumes_change(
    manager: SandboxManager, service: AsyncMock, tmp_path: Path
):
    """Sandbox with stale volume mount should be deleted."""
    old_path = "/old/uploads/user_5"
    sb = _make_sb_info(
        "user::5",
        image="img:v1",
        volumes=[(old_path, old_path, "rw")],
    )

    service.list_sandboxes.return_value = [sb]

    new_uploads = tmp_path / "uploads"
    new_uploads.mkdir()

    with (
        patch.dict(
            "os.environ",
            {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "512"},
        ),
        patch("xagent.web.sandbox_manager.UPLOADS_DIR", new_uploads),
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::5")


@pytest.mark.asyncio
async def test_cleanup_stops_when_config_matches(
    manager: SandboxManager, service: AsyncMock, tmp_path: Path
):
    """Sandbox whose config matches should be stopped, not deleted."""
    uploads = tmp_path / "uploads"
    user_dir = uploads / "user_6"
    user_dir.mkdir(parents=True)
    resolved = str(user_dir.resolve())

    sb = _make_sb_info(
        "user::6",
        image="img:v1",
        cpus=1,
        memory=512,
        volumes=[(resolved, resolved, "rw")],
    )

    mock_box = AsyncMock()
    service.list_sandboxes.return_value = [sb]
    service.get_or_create.return_value = mock_box

    with (
        patch.dict(
            "os.environ",
            {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "512"},
        ),
        patch("xagent.web.sandbox_manager.UPLOADS_DIR", uploads),
    ):
        await manager.cleanup()

    service.delete.assert_not_awaited()
    mock_box.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_deletes_on_multiple_changes(
    manager: SandboxManager, service: AsyncMock
):
    """Sandbox with image AND cpus changed should be deleted once."""
    sb = _make_sb_info("user::7", image="old:v0", cpus=1, memory=256)

    service.list_sandboxes.return_value = [sb]

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "new:v2", "SANDBOX_CPUS": "8", "SANDBOX_MEMORY": "2048"},
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::7")


@pytest.mark.asyncio
async def test_cleanup_handles_non_managed_sandbox(
    manager: SandboxManager, service: AsyncMock
):
    """Sandbox with non-standard name should not crash cleanup."""
    sb = _make_sb_info("__warmup__", image="img:v1")

    mock_box = AsyncMock()
    service.list_sandboxes.return_value = [sb]
    service.get_or_create.return_value = mock_box

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "512"},
    ):
        await manager.cleanup()

    # Config matches (except volumes which is skipped), so just stop
    service.delete.assert_not_awaited()
    mock_box.stop.assert_awaited_once()
