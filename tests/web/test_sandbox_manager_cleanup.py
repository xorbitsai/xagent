"""Test SandboxManager.cleanup — delete sandbox if config changed."""

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# WHY: importing app runs its module-level setup_logging(), which resets the
# root handlers via dictConfig. Import it at collection so that reset happens
# before caplog attaches per-test, not inside a caplog context.
from xagent.web.app import _startup_phase

from tests.web.sandbox_fakes import FakeSandboxService, _FakeReconcileContainer
from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
    build_code_mount_volumes,
)
from xagent.sandbox.base import (
    ResolvedSandboxRuntimeSpec,
    SandboxConfig,
    SandboxInfo,
    SandboxMountIntent,
    SandboxTemplate,
)
from xagent.web.sandbox_manager import _SANDBOX_STOP_TIMEOUT_SECONDS, SandboxManager


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
def service() -> FakeSandboxService:
    return FakeSandboxService()


@pytest.fixture
def manager(service: FakeSandboxService) -> SandboxManager:
    return SandboxManager(service)


def test_build_code_mount_volumes_uses_host_project_root(tmp_path: Path):
    """Docker sibling mode should mount source paths from the Docker host."""
    with patch.dict(
        "os.environ",
        {"XAGENT_SANDBOX_HOST_PROJECT_ROOT": str(tmp_path)},
        clear=True,
    ):
        volumes = build_code_mount_volumes()

    assert volumes == [
        (str(tmp_path / "src"), "/app/src", "ro"),
        (str(tmp_path / "tests"), "/app/tests", "ro"),
    ]


def test_default_volumes_map_user_workspace_to_host_storage(
    manager: SandboxManager, tmp_path: Path
):
    """Docker sibling mode should translate backend storage paths to host paths."""
    backend_storage_root = tmp_path / "backend" / ".xagent"
    host_storage_root = tmp_path / "host" / ".xagent"
    backend_user_dir = backend_storage_root / "uploads" / "user_42"

    with (
        patch.dict(
            "os.environ",
            {
                "XAGENT_STORAGE_ROOT": str(backend_storage_root),
                "XAGENT_UPLOADS_DIR": str(backend_storage_root / "uploads"),
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": str(host_storage_root),
            },
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/repo/src", "/app/src", "ro")],
        ),
    ):
        volumes = manager._make_default_volumes(
            "user",
            "42",
            mount_intent=SandboxMountIntent(
                mount_root=str(backend_user_dir),
                extra_mounts=(str(backend_user_dir),),
            ),
        )

    assert volumes == [
        ("/repo/src", "/app/src", "ro"),
        (
            str(host_storage_root / "uploads" / "user_42"),
            str(backend_user_dir),
            "rw",
        ),
    ]


def test_default_volumes_include_build_preview_and_user_dirs(
    manager: SandboxManager, tmp_path: Path
):
    """Preview sandboxes need the preview base plus the user's upload dir."""
    backend_storage_root = tmp_path / "backend" / ".xagent"
    host_storage_root = tmp_path / "host" / ".xagent"
    build_preview_dir = backend_storage_root / "uploads" / "build_preview"
    user_dir = backend_storage_root / "uploads" / "user_7"

    with (
        patch.dict(
            "os.environ",
            {
                "XAGENT_STORAGE_ROOT": str(backend_storage_root),
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": str(host_storage_root),
            },
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/repo/src", "/app/src", "ro")],
        ),
    ):
        volumes = manager._make_default_volumes(
            "user",
            "7",
            mount_intent=SandboxMountIntent(
                mount_root=str(build_preview_dir),
                extra_mounts=(str(user_dir),),
            ),
        )

    assert volumes == [
        ("/repo/src", "/app/src", "ro"),
        (
            str(host_storage_root / "uploads" / "build_preview"),
            str(build_preview_dir),
            "rw",
        ),
        (str(host_storage_root / "uploads" / "user_7"), str(user_dir), "rw"),
    ]


def test_default_volumes_keep_external_dirs_outside_storage(
    manager: SandboxManager, tmp_path: Path
):
    """Only storage-root paths are translated; other allowed dirs stay explicit."""
    backend_storage_root = tmp_path / "backend" / ".xagent"
    host_storage_root = tmp_path / "host" / ".xagent"
    base_dir = backend_storage_root / "uploads" / "user_5"
    external_dir = tmp_path / "shared" / "kb"

    with (
        patch.dict(
            "os.environ",
            {
                "XAGENT_STORAGE_ROOT": str(backend_storage_root),
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": str(host_storage_root),
            },
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/repo/src", "/app/src", "ro")],
        ),
    ):
        volumes = manager._make_default_volumes(
            "user",
            "5",
            mount_intent=SandboxMountIntent(
                mount_root=str(base_dir),
                extra_mounts=(str(external_dir),),
            ),
        )

    assert (str(external_dir), str(external_dir), "rw") in volumes


def test_default_volumes_mount_workspace_owner_not_current_user(
    manager: SandboxManager, tmp_path: Path
):
    """Admin/current-user sandboxes should use the task owner's workspace path."""
    backend_storage_root = tmp_path / "backend" / ".xagent"
    host_storage_root = tmp_path / "host" / ".xagent"
    owner_dir = backend_storage_root / "uploads" / "user_99"

    with (
        patch.dict(
            "os.environ",
            {
                "XAGENT_STORAGE_ROOT": str(backend_storage_root),
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": str(host_storage_root),
            },
            clear=True,
        ),
        patch(
            "xagent.web.sandbox_manager.build_code_mount_volumes",
            return_value=[("/repo/src", "/app/src", "ro")],
        ),
    ):
        volumes = manager._make_default_volumes(
            "user",
            "1",
            mount_intent=SandboxMountIntent(
                mount_root=str(owner_dir),
                extra_mounts=(str(owner_dir),),
            ),
        )

    assert (
        str(host_storage_root / "uploads" / "user_99"),
        str(owner_dir),
        "rw",
    ) in volumes
    assert (
        str(host_storage_root / "uploads" / "user_1"),
        str(backend_storage_root / "uploads" / "user_1"),
        "rw",
    ) not in volumes


@pytest.mark.asyncio
async def test_cleanup_deletes_on_image_change(
    manager: SandboxManager, service: FakeSandboxService
):
    """Sandbox with stale image should be deleted."""
    sb = _make_sb_info("user::1", image="old:v0")

    service.list_sandboxes.return_value = [sb]

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "new:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "512"},
        clear=True,
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::1")


@pytest.mark.asyncio
async def test_cleanup_deletes_on_cpus_change(
    manager: SandboxManager, service: FakeSandboxService
):
    """Sandbox with different cpus should be deleted."""
    sb = _make_sb_info("user::2", image="img:v1", cpus=1)

    service.list_sandboxes.return_value = [sb]

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "4", "SANDBOX_MEMORY": "512"},
        clear=True,
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::2")


@pytest.mark.asyncio
async def test_cleanup_deletes_on_memory_change(
    manager: SandboxManager, service: FakeSandboxService
):
    """Sandbox with different memory should be deleted."""
    sb = _make_sb_info("user::3", image="img:v1", memory=512)

    service.list_sandboxes.return_value = [sb]

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "1024"},
        clear=True,
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::3")


@pytest.mark.asyncio
async def test_cleanup_deletes_on_volumes_change(
    manager: SandboxManager, service: FakeSandboxService, tmp_path: Path
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
            clear=True,
        ),
        patch("xagent.web.sandbox_manager.get_uploads_dir", return_value=new_uploads),
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::5")


@pytest.mark.asyncio
async def test_cleanup_stops_when_config_matches(
    manager: SandboxManager, service: FakeSandboxService, tmp_path: Path
):
    """Sandbox whose config matches should be stopped, not deleted."""
    uploads = tmp_path / "uploads"
    user_dir = uploads / "user_6"
    user_dir.mkdir(parents=True)
    resolved = str(user_dir.resolve())

    # Build expected volumes: code mounts (ro) + user workspace (rw)
    code_volumes = build_code_mount_volumes()
    sb = _make_sb_info(
        "user::6",
        image="img:v1",
        cpus=1,
        memory=512,
        volumes=code_volumes + [(resolved, resolved, "rw")],
    )

    mock_box = AsyncMock()
    service.list_sandboxes.return_value = [sb]
    service.get_or_create.return_value = mock_box

    with (
        patch.dict(
            "os.environ",
            {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "512"},
            clear=True,
        ),
        patch("xagent.web.sandbox_manager.get_uploads_dir", return_value=uploads),
    ):
        await manager.cleanup()

    service.delete.assert_not_awaited()
    mock_box.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_stops_worker_when_base_user_config_matches(
    manager: SandboxManager, service: FakeSandboxService, tmp_path: Path
):
    """Worker sandbox cleanup should compare against the owner workspace."""
    uploads = tmp_path / "uploads"
    user_dir = uploads / "user_6"
    user_dir.mkdir(parents=True)
    resolved = str(user_dir.resolve())

    code_volumes = build_code_mount_volumes()
    sb = _make_sb_info(
        "user::6::worker::0",
        image="img:v1",
        cpus=1,
        memory=512,
        volumes=code_volumes + [(resolved, resolved, "rw")],
    )

    mock_box = AsyncMock()
    service.list_sandboxes.return_value = [sb]
    service.get_or_create.return_value = mock_box

    with (
        patch.dict(
            "os.environ",
            {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "512"},
            clear=True,
        ),
        patch("xagent.web.sandbox_manager.get_uploads_dir", return_value=uploads),
    ):
        await manager.cleanup()

    service.delete.assert_not_awaited()
    service.get_or_create.assert_awaited_once_with(
        "user::6::worker::0", template=sb.template, config=sb.config
    )
    mock_box.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_deletes_on_multiple_changes(
    manager: SandboxManager, service: FakeSandboxService
):
    """Sandbox with image AND cpus changed should be deleted once."""
    sb = _make_sb_info("user::7", image="old:v0", cpus=1, memory=256)

    service.list_sandboxes.return_value = [sb]

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "new:v2", "SANDBOX_CPUS": "8", "SANDBOX_MEMORY": "2048"},
        clear=True,
    ):
        await manager.cleanup()

    service.delete.assert_awaited_once_with("user::7")


@pytest.mark.asyncio
async def test_cleanup_handles_non_managed_sandbox(
    manager: SandboxManager, service: FakeSandboxService
):
    """Sandbox with non-standard name should not crash cleanup."""
    sb = _make_sb_info("__warmup__", image="img:v1")

    mock_box = AsyncMock()
    service.list_sandboxes.return_value = [sb]
    service.get_or_create.return_value = mock_box

    with patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE": "img:v1", "SANDBOX_CPUS": "1", "SANDBOX_MEMORY": "512"},
        clear=True,
    ):
        await manager.cleanup()

    # Config matches (except volumes which is skipped), so just stop
    service.delete.assert_not_awaited()
    mock_box.stop.assert_awaited_once()


class TestQuiesceReconcilingBackend:
    """``cleanup()`` on a backend that supports spec reconciliation routes to
    ``_quiesce`` instead of ``_legacy_cleanup``: stop every running managed
    container, never delete or inspect any container's configuration (that
    convergence decision belongs to the reconciliation matrix on next use,
    not to cleanup)."""

    @pytest.mark.asyncio
    async def test_quiesce_stops_running_and_never_deletes(self) -> None:
        service = FakeSandboxService(runtime_spec_supported=True)
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="img:v1"
        )
        service._containers["user::1"] = _FakeReconcileContainer(
            state="running",
            spec=spec,
            fingerprint_label=spec.fingerprint(),
            version_label="1",
        )
        service._containers["user::2"] = _FakeReconcileContainer(
            state="stopped",
            spec=spec,
            fingerprint_label=spec.fingerprint(),
            version_label="1",
        )
        service.containers = {"user::1", "user::2"}

        manager = SandboxManager(service)

        await manager.cleanup()

        service.delete.assert_not_awaited()
        service.stop_existing.assert_awaited_once_with(
            "user::1", timeout=_SANDBOX_STOP_TIMEOUT_SECONDS
        )

    @pytest.mark.asyncio
    async def test_quiesce_logs_diagnostic_summary(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Quiesce emits one summary line with seen/running/stopped counts so a
        slow startup is diagnosable from logs alone (issue #231)."""
        service = FakeSandboxService(runtime_spec_supported=True)
        spec = ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image", image="img:v1"
        )
        for name, state in (
            ("user::1", "running"),
            ("user::2", "running"),
            ("user::3", "stopped"),
        ):
            service._containers[name] = _FakeReconcileContainer(
                state=state,
                spec=spec,
                fingerprint_label=spec.fingerprint(),
                version_label="1",
            )
        service.containers = {"user::1", "user::2", "user::3"}

        manager = SandboxManager(service)

        with caplog.at_level(logging.INFO, logger="xagent.web.sandbox_manager"):
            await manager.cleanup()

        summaries = [
            r.getMessage()
            for r in caplog.records
            if "Sandbox quiesce completed" in r.getMessage()
        ]
        assert len(summaries) == 1, "expected exactly one quiesce summary line"
        msg = summaries[0]
        assert "seen=3" in msg
        assert "running=2" in msg
        assert "stopped=2" in msg
        assert "failed=0" in msg
        # Lock the full schema so a dropped/renamed duration field is caught.
        assert "stop_time=" in msg
        assert "total=" in msg
        assert "status=ok" in msg

    @pytest.mark.asyncio
    async def test_quiesce_list_failure_marks_status(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A ``list_sandboxes`` failure must not read as clean empty work: the
        summary reports ``status=list_failed`` rather than a zero-count success."""
        service = FakeSandboxService(runtime_spec_supported=True)
        service.list_sandboxes.side_effect = RuntimeError("boom")

        manager = SandboxManager(service)

        with caplog.at_level(logging.INFO, logger="xagent.web.sandbox_manager"):
            await manager.cleanup()

        summaries = [
            r.getMessage()
            for r in caplog.records
            if "Sandbox quiesce completed" in r.getMessage()
        ]
        assert len(summaries) == 1
        assert "seen=0" in summaries[0]
        assert "status=list_failed" in summaries[0]


class TestLegacyCleanupSummary:
    """``_legacy_cleanup`` (Boxlite route) emits one structured summary on every
    path — empty, stop, and error — with the same ``running``/``stop_time``
    fields the quiesce route carries, so cleanup telemetry stays consistent
    across supported backends (issue #231)."""

    @pytest.mark.asyncio
    async def test_empty_listing_still_emits_summary(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeSandboxService()  # legacy route (no runtime spec)
        service.list_sandboxes.return_value = []

        manager = SandboxManager(service)

        with caplog.at_level(logging.INFO, logger="xagent.web.sandbox_manager"):
            await manager.cleanup()

        summaries = [
            r.getMessage()
            for r in caplog.records
            if "Sandbox cleanup completed" in r.getMessage()
        ]
        assert len(summaries) == 1
        assert "seen=0" in summaries[0]
        assert "status=ok" in summaries[0]

    @pytest.mark.asyncio
    async def test_running_sandbox_reports_running_and_stop_time(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        service = FakeSandboxService()
        service.list_sandboxes.return_value = [
            _make_sb_info("__warmup__", image="img:v1", state="running")
        ]
        service.get_or_create.return_value = AsyncMock()

        manager = SandboxManager(service)

        with (
            patch.dict(
                "os.environ",
                {
                    "SANDBOX_IMAGE": "img:v1",
                    "SANDBOX_CPUS": "1",
                    "SANDBOX_MEMORY": "512",
                },
                clear=True,
            ),
            caplog.at_level(logging.INFO, logger="xagent.web.sandbox_manager"),
        ):
            await manager.cleanup()

        summaries = [
            r.getMessage()
            for r in caplog.records
            if "Sandbox cleanup completed" in r.getMessage()
        ]
        assert len(summaries) == 1
        assert "running=1" in summaries[0]
        assert "stopped=1" in summaries[0]
        assert "stop_time=" in summaries[0]
        assert "status=ok" in summaries[0]


class TestStartupPhaseLogging:
    """``_startup_phase`` emits a terminal line on every exit — success, error,
    and cancellation — so a stalled/aborted startup is never left showing only
    its begin line (issue #231)."""

    def test_success_logs_done(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="xagent.web.app"):
            with _startup_phase("demo"):
                pass

        messages = [r.getMessage() for r in caplog.records]
        assert any("startup phase done: demo" in m for m in messages)

    def test_error_logs_failed_and_reraises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="xagent.web.app"):
            with pytest.raises(ValueError):
                with _startup_phase("demo"):
                    raise ValueError("boom")

        messages = [r.getMessage() for r in caplog.records]
        assert any("startup phase failed: demo" in m for m in messages)

    def test_cancellation_logs_terminal_and_reraises(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="xagent.web.app"):
            with pytest.raises(asyncio.CancelledError):
                with _startup_phase("demo"):
                    raise asyncio.CancelledError()

        messages = [r.getMessage() for r in caplog.records]
        assert any("startup phase cancelled: demo" in m for m in messages)
