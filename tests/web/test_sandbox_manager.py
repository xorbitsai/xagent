"""Test sandbox manager functionality."""

import asyncio
import logging
import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.web.sandbox_fakes import FakeSandboxService
from xagent.sandbox.base import (
    SandboxConfig,
    SandboxInfo,
    SandboxMountIntent,
    SandboxRuntimeConflictError,
    SandboxTemplate,
)
from xagent.web.sandbox_manager import (
    SandboxManager,
    _create_boxlite_service,
    _create_docker_service,
    _create_sandbox_service,
    get_sandbox_manager,
)


def _sandbox_info(name: str, state: str = "running") -> SandboxInfo:
    return SandboxInfo(
        name=name,
        state=state,
        template=SandboxTemplate(type="image", image="img:v1"),
        config=SandboxConfig(),
    )


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global singleton state before each test."""
    import xagent.web.sandbox_manager as mod

    mod._sandbox_manager = None
    mod._sandbox_manager_initialized = False
    yield
    mod._sandbox_manager = None
    mod._sandbox_manager_initialized = False


class TestCreateSandboxService:
    """Test _create_sandbox_service function."""

    def test_disabled_returns_none(self):
        """Test sandbox disabled via env returns None."""
        with patch.dict("os.environ", {"SANDBOX_ENABLED": ""}):
            result = _create_sandbox_service()
        assert result is None

    def test_docker_default(self):
        """Test default implementation is docker."""
        with (
            patch.dict("os.environ", {"SANDBOX_ENABLED": "true"}, clear=False),
            patch("xagent.web.sandbox_manager._create_docker_service") as mock_create,
        ):
            os.environ.pop("SANDBOX_IMPLEMENTATION", None)
            mock_create.return_value = FakeSandboxService()
            result = _create_sandbox_service()
        assert result is not None
        mock_create.assert_called_once()

    def test_unknown_implementation_falls_back_to_docker(self):
        """Test unknown implementation falls back to docker."""
        with (
            patch.dict(
                "os.environ",
                {"SANDBOX_ENABLED": "true", "SANDBOX_IMPLEMENTATION": "unknown"},
                clear=False,
            ),
            patch("xagent.web.sandbox_manager._create_docker_service") as mock_create,
        ):
            mock_create.return_value = FakeSandboxService()
            _create_sandbox_service()
        mock_create.assert_called_once()

    def test_docker_selected(self):
        """Test docker implementation selection."""
        with (
            patch.dict(
                "os.environ",
                {"SANDBOX_ENABLED": "true", "SANDBOX_IMPLEMENTATION": "docker"},
                clear=False,
            ),
            patch("xagent.web.sandbox_manager._create_docker_service") as mock_create,
        ):
            mock_create.return_value = FakeSandboxService()
            result = _create_sandbox_service()
        assert result is not None
        mock_create.assert_called_once()


class TestGetSandboxManager:
    """Test get_sandbox_manager singleton."""

    def test_returns_none_when_service_none(self):
        """Test returns None when sandbox service creation fails."""
        with patch(
            "xagent.web.sandbox_manager._create_sandbox_service", return_value=None
        ):
            result = get_sandbox_manager()
        assert result is None

    def test_returns_manager_when_service_available(self):
        """Test returns SandboxManager when service is available."""
        mock_service = FakeSandboxService()
        with patch(
            "xagent.web.sandbox_manager._create_sandbox_service",
            return_value=mock_service,
        ):
            result = get_sandbox_manager()
        assert isinstance(result, SandboxManager)

    def test_singleton_returns_same_instance(self):
        """Test singleton pattern returns same instance."""
        mock_service = FakeSandboxService()
        with patch(
            "xagent.web.sandbox_manager._create_sandbox_service",
            return_value=mock_service,
        ):
            first = get_sandbox_manager()
            second = get_sandbox_manager()
        assert first is second

    def test_initialized_flag_prevents_retry_on_none(self):
        """Test that once initialized with None, it doesn't retry."""
        with patch(
            "xagent.web.sandbox_manager._create_sandbox_service", return_value=None
        ) as mock_create:
            get_sandbox_manager()
            get_sandbox_manager()
            get_sandbox_manager()
        # Should only be called once due to _initialized flag
        mock_create.assert_called_once()

    def test_thread_safety(self):
        """Test concurrent access returns same instance."""
        mock_service = FakeSandboxService()
        results = []
        barrier = threading.Barrier(5)

        def worker():
            barrier.wait()
            with patch(
                "xagent.web.sandbox_manager._create_sandbox_service",
                return_value=mock_service,
            ):
                results.append(get_sandbox_manager())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be the same instance
        assert all(r is results[0] for r in results)


try:
    from xagent.sandbox import BoxliteSandboxService  # noqa: F401

    _has_boxlite = True
except ImportError:
    _has_boxlite = False


@pytest.mark.skipif(not _has_boxlite, reason="boxlite not installed")
class TestCreateBoxliteService:
    """Test _create_boxlite_service function."""

    def test_custom_home_dir(self):
        """Test creating service with custom home directory."""
        with (
            patch.dict(
                "os.environ",
                {"BOXLITE_HOME_DIR": "/tmp/sandbox"},
                clear=False,
            ),
            patch(
                "xagent.sandbox.BoxliteSandboxService",
                return_value=FakeSandboxService(),
            ) as mock_cls,
            patch("xagent.sandbox.MemBoxliteStore", return_value=MagicMock()),
        ):
            _create_boxlite_service()

        assert mock_cls.call_args[1]["home_dir"] == "/tmp/sandbox"

    def test_creation_failure_returns_none(self):
        """Test that BoxliteSandboxService construction failure returns None."""
        with (
            patch(
                "xagent.sandbox.BoxliteSandboxService",
                side_effect=RuntimeError("docker not available"),
            ),
            patch("xagent.sandbox.MemBoxliteStore", return_value=MagicMock()),
        ):
            result = _create_boxlite_service()

        assert result is None


class TestCreateDockerService:
    """Test _create_docker_service function."""

    @pytest.fixture(autouse=True)
    def _namespace_env(self, monkeypatch):
        """A namespace is mandatory for the Docker sandbox implementation."""
        monkeypatch.setenv("XAGENT_SANDBOX_NAMESPACE", "test")

    def test_uses_db_store(self):
        """Test Docker sandbox service is created with persistent store."""
        with (
            patch("xagent.web.sandbox_store.DBDockerStore") as mock_store_cls,
            patch(
                "xagent.sandbox.DockerSandboxService",
                return_value=FakeSandboxService(),
            ) as mock_service_cls,
        ):
            _create_docker_service()

        mock_store_cls.assert_called_once_with()
        assert mock_service_cls.call_args[1]["store"] is mock_store_cls.return_value
        assert mock_service_cls.call_args[1]["namespace"] == "test"

    def test_creation_failure_returns_none(self):
        """Test that DockerSandboxService construction failure returns None."""
        with (
            patch("xagent.web.sandbox_store.DBDockerStore", return_value=MagicMock()),
            patch(
                "xagent.sandbox.DockerSandboxService",
                side_effect=RuntimeError("docker not available"),
            ),
        ):
            result = _create_docker_service()

        assert result is None

    def test_missing_namespace_is_fatal(self, monkeypatch):
        """Missing namespace must fail loudly, never degrade to unscoped mode."""
        monkeypatch.delenv("XAGENT_SANDBOX_NAMESPACE", raising=False)
        with pytest.raises(RuntimeError, match="XAGENT_SANDBOX_NAMESPACE is required"):
            _create_docker_service()

    def test_legacy_container_inventory_is_logged(self, caplog):
        """Startup must surface inactive legacy containers for removal."""

        class _ServiceWithLegacyCount(FakeSandboxService):
            def count_legacy_containers(self) -> tuple[int, int]:
                return 0, 2

        with (
            patch("xagent.web.sandbox_store.DBDockerStore", return_value=MagicMock()),
            patch(
                "xagent.sandbox.DockerSandboxService",
                return_value=_ServiceWithLegacyCount(),
            ),
        ):
            with caplog.at_level(logging.WARNING):
                result = _create_docker_service()

        assert result is not None
        assert (
            "2 inactive legacy xagent.managed=true sandbox container(s)" in caplog.text
        )
        assert "xagent.managed=true" in caplog.text

    def test_running_legacy_containers_are_logged_as_errors(self, caplog):
        class _ServiceWithLegacyCount(FakeSandboxService):
            def count_legacy_containers(self) -> tuple[int, int]:
                return 1, 2

        with (
            patch("xagent.web.sandbox_store.DBDockerStore", return_value=MagicMock()),
            patch(
                "xagent.sandbox.DockerSandboxService",
                return_value=_ServiceWithLegacyCount(),
            ),
            caplog.at_level(logging.ERROR),
        ):
            result = _create_docker_service()

        assert result is not None
        assert (
            "1 running legacy xagent.managed=true sandbox container(s)" in caplog.text
        )
        assert "Stop them before starting v2 sandbox workloads" in caplog.text

    def test_legacy_inventory_failure_does_not_misreport_creation(self, caplog):
        class _ServiceWithFailingLegacyCount(FakeSandboxService):
            def count_legacy_containers(self) -> tuple[int, int]:
                raise RuntimeError("legacy inventory unavailable")

        service = _ServiceWithFailingLegacyCount()
        with (
            patch("xagent.web.sandbox_store.DBDockerStore", return_value=MagicMock()),
            patch("xagent.sandbox.DockerSandboxService", return_value=service),
            caplog.at_level(logging.WARNING),
        ):
            result = _create_docker_service()

        assert result is service
        assert "Failed to inventory legacy sandbox containers" in caplog.text
        assert "Failed to create Docker sandbox service" not in caplog.text


class TestSandboxConfigParsing:
    """Test sandbox config parsing from environment variables."""

    def test_default_config_when_no_env_set(self):
        """Test default config when no env vars are set."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {}, clear=True):
            image, config = manager._get_sandbox_image_and_config()

        # Should use defaults - SandboxConfig has cpus=1, memory=512 as defaults
        assert image  # Should have some image value
        assert config.cpus == 1  # SandboxConfig default
        assert config.memory == 512  # SandboxConfig default
        assert config.env is None
        assert config.volumes is None

    def test_cpu_parsing_valid(self):
        """Test valid CPU value is parsed correctly."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_CPUS": "4"}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.cpus == 4

    def test_cpu_parsing_invalid(self):
        """Test invalid CPU value uses SandboxConfig default (1)."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_CPUS": "invalid"}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        # Invalid value is skipped, SandboxConfig default (1) is used
        assert config.cpus == 1

    def test_memory_parsing_valid(self):
        """Test valid memory value is parsed correctly."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_MEMORY": "2048"}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.memory == 2048

    def test_env_parsing_single_var(self):
        """Test parsing single environment variable."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_ENV": "KEY=value"}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env == {"KEY": "value"}

    def test_env_parsing_multiple_vars(self):
        """Test parsing multiple environment variables."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {"SANDBOX_ENV": "KEY1=value1;KEY2=value2;KEY3=value3"},
            clear=False,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env == {"KEY1": "value1", "KEY2": "value2", "KEY3": "value3"}

    def test_env_parsing_empty(self):
        """Test empty env string results in None."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_ENV": ""}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env is None

    def test_env_parsing_invalid_format(self):
        """Test invalid env format is skipped with warning."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_ENV": "VALID=1;INVALID;VALID2=2"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        # Should skip invalid entry
        assert config.env == {"VALID": "1", "VALID2": "2"}

    def test_env_parsing_with_spaces(self):
        """Test env vars with spaces are trimmed."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_ENV": " KEY = value ; ANOTHER = test "}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env == {"KEY": "value", "ANOTHER": "test"}

    def test_volume_parsing_single_volume(self):
        """Test parsing single volume mount."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "/host:/container:ro"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [("/host", "/container", "ro")]

    def test_volume_parsing_multiple_volumes(self):
        """Test parsing multiple volume mounts."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/host1:/container1:ro;/host2:/container2:rw"},
            clear=False,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [
            ("/host1", "/container1", "ro"),
            ("/host2", "/container2", "rw"),
        ]

    def test_volume_parsing_default_mode(self):
        """Test volume defaults to 'ro' mode when not specified."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "/host:/container"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [("/host", "/container", "ro")]

    def test_volume_parsing_with_tilde_expansion(self):
        """Test volume path with tilde expansion."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "~/data:/data:ro"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        # Should expand tilde to absolute path
        src_path = config.volumes[0][0]
        expected_path = os.path.abspath(os.path.expanduser("~/data"))
        assert "~" not in src_path
        assert src_path == expected_path
        assert config.volumes[0][1] == "/data"
        assert config.volumes[0][2] == "ro"

    def test_sibling_mode_rejects_tilde_sandbox_volume_source(self):
        """Docker sibling mode must not expand backend-container home paths."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": "/host/.xagent",
                "SANDBOX_VOLUMES": "~/data:/data:ro",
            },
            clear=True,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes is None

    def test_sibling_mode_rejects_relative_sandbox_volume_source(self):
        """Docker sibling mode requires host-side absolute volume sources."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {
                "XAGENT_SANDBOX_HOST_STORAGE_ROOT": "/host/.xagent",
                "SANDBOX_VOLUMES": "relative:/data:ro",
            },
            clear=True,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes is None

    def test_volume_parsing_empty(self):
        """Test empty volumes string results in None."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict("os.environ", {"SANDBOX_VOLUMES": ""}, clear=False):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes is None

    def test_volume_parsing_invalid_format(self):
        """Test invalid volume format is skipped with warning."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/valid:/valid:ro;invalid;/another:/another:rw"},
            clear=False,
        ):
            _, config = manager._get_sandbox_image_and_config()

        # Should skip invalid entries
        assert config.volumes == [
            ("/valid", "/valid", "ro"),
            ("/another", "/another", "rw"),
        ]

    def test_volume_parsing_invalid_mode_defaults_to_ro(self):
        """Test invalid volume mode defaults to 'ro'."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "/host:/container:xyz"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [("/host", "/container", "ro")]

    def test_volume_parsing_mode_case_insensitive(self):
        """Test volume mode is case-insensitive."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ", {"SANDBOX_VOLUMES": "/host:/container:RW"}, clear=False
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.volumes == [("/host", "/container", "rw")]

    def test_combined_config(self):
        """Test parsing all config options together."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {
                "SANDBOX_IMAGE": "custom/image:latest",
                "SANDBOX_CPUS": "2",
                "SANDBOX_MEMORY": "1024",
                "SANDBOX_ENV": "KEY1=val1;KEY2=val2",
                "SANDBOX_VOLUMES": "/host:/container:ro",
            },
            clear=False,
        ):
            image, config = manager._get_sandbox_image_and_config()

        assert image == "custom/image:latest"
        assert config.cpus == 2
        assert config.memory == 1024
        assert config.env == {"KEY1": "val1", "KEY2": "val2"}
        assert config.volumes == [("/host", "/container", "ro")]

    def test_volumes_with_semicolon_in_env(self):
        """Test env vars and volumes both use semicolon separator."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        with patch.dict(
            "os.environ",
            {
                "SANDBOX_ENV": "KEY=val",
                "SANDBOX_VOLUMES": "/host:/container:ro",
            },
            clear=False,
        ):
            _, config = manager._get_sandbox_image_and_config()

        assert config.env == {"KEY": "val"}
        assert config.volumes == [("/host", "/container", "ro")]


class TestSandboxLifecycleConfig:
    """Test sandbox lifecycle identity and runtime config consistency."""

    @pytest.mark.asyncio
    async def test_same_lifecycle_rejects_different_workspace_config(self, tmp_path):
        """Same sandbox name should not be deleted/recreated for another workspace."""
        service = FakeSandboxService()
        service.get_or_create = AsyncMock(return_value=MagicMock())
        service.delete = AsyncMock()
        manager = SandboxManager(service)

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[("/repo/src", "/app/src", "ro")],
            ),
        ):
            await manager.get_or_create_sandbox(
                "user",
                "42",
                mount_intent=SandboxMountIntent(mount_root=str(tmp_path / "user_42")),
            )
            with pytest.raises(
                SandboxRuntimeConflictError, match="different runtime configuration"
            ):
                await manager.get_or_create_sandbox(
                    "user",
                    "42",
                    mount_intent=SandboxMountIntent(
                        mount_root=str(tmp_path / "build_preview")
                    ),
                )

        service.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_distinct_lifecycles_allow_distinct_workspace_configs(self, tmp_path):
        """Build preview and chat should not compete for the same sandbox name."""
        service = FakeSandboxService()
        service.get_or_create = AsyncMock(side_effect=[MagicMock(), MagicMock()])
        manager = SandboxManager(service)

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[("/repo/src", "/app/src", "ro")],
            ),
        ):
            await manager.get_or_create_sandbox(
                "user",
                "42",
                mount_intent=SandboxMountIntent(mount_root=str(tmp_path / "user_42")),
            )
            await manager.get_or_create_sandbox(
                "build_preview",
                "42",
                mount_intent=SandboxMountIntent(
                    mount_root=str(tmp_path / "build_preview")
                ),
            )

        assert service.get_or_create.await_count == 2
        assert service.get_or_create.await_args_list[0].args[0] == "user::42"
        assert service.get_or_create.await_args_list[1].args[0] == "build_preview::42"


class TestSandboxLeaseProvider:
    """Test leasing primary and worker sandboxes for tool execution."""

    @pytest.mark.asyncio
    async def test_unsafe_lease_returns_primary_sandbox(self, tmp_path):
        """Tools that are not concurrency-safe should keep the existing behavior."""

        async def get_or_create(name, *args, **kwargs):
            sandbox = MagicMock()
            sandbox.name = name
            return sandbox

        service = FakeSandboxService()
        service.get_or_create = AsyncMock(side_effect=get_or_create)
        manager = SandboxManager(service)

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[("/repo/src", "/app/src", "ro")],
            ),
        ):
            provider = await manager.create_lease_provider(
                "user",
                "42",
                mount_intent=SandboxMountIntent(mount_root=str(tmp_path / "user_42")),
            )
            async with provider.lease(concurrency_safe=False) as sandbox:
                assert sandbox.name == "user::42"

        assert service.get_or_create.await_count == 1
        assert service.get_or_create.await_args_list[0].args[0] == "user::42"

    @pytest.mark.asyncio
    async def test_safe_concurrent_leases_use_distinct_workers(self, tmp_path):
        """Concurrent safe leases should execute on separate worker sandboxes."""

        async def get_or_create(name, *args, **kwargs):
            sandbox = MagicMock()
            sandbox.name = name
            return sandbox

        service = FakeSandboxService()
        service.get_or_create = AsyncMock(side_effect=get_or_create)
        manager = SandboxManager(service)

        with (
            patch.dict(
                "os.environ", {"XAGENT_SANDBOX_MAX_CONCURRENCY": "2"}, clear=True
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[("/repo/src", "/app/src", "ro")],
            ),
        ):
            provider = await manager.create_lease_provider(
                "user",
                "42",
                mount_intent=SandboxMountIntent(mount_root=str(tmp_path / "user_42")),
            )
            async with provider.lease(concurrency_safe=True) as first:
                async with provider.lease(concurrency_safe=True) as second:
                    assert first.name == "user::42::worker::0"
                    assert second.name == "user::42::worker::1"

        assert service.get_or_create.await_args_list[0].args[0] == "user::42"
        assert service.get_or_create.await_args_list[1].args[0] == (
            "user::42::worker::0"
        )
        assert service.get_or_create.await_args_list[2].args[0] == (
            "user::42::worker::1"
        )

    @pytest.mark.asyncio
    async def test_worker_sandboxes_reuse_primary_workspace_config(self, tmp_path):
        """Worker sandboxes should mount the same workspace roots as the primary."""

        async def get_or_create(name, *args, **kwargs):
            sandbox = MagicMock()
            sandbox.name = name
            return sandbox

        service = FakeSandboxService()
        service.get_or_create = AsyncMock(side_effect=get_or_create)
        manager = SandboxManager(service)
        workspace_dir = tmp_path / "user_42"

        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[("/repo/src", "/app/src", "ro")],
            ),
        ):
            provider = await manager.create_lease_provider(
                "user",
                "42",
                mount_intent=SandboxMountIntent(mount_root=str(workspace_dir)),
            )
            async with provider.lease(concurrency_safe=True):
                pass

        primary_config = service.get_or_create.await_args_list[0].kwargs["config"]
        worker_config = service.get_or_create.await_args_list[1].kwargs["config"]
        assert primary_config.volumes == worker_config.volumes
        assert any(
            volume[0] == str(workspace_dir) and volume[2] == "rw"
            for volume in worker_config.volumes
        )

    @pytest.mark.asyncio
    async def test_delete_sandbox_removes_cached_worker_sandboxes(self, tmp_path):
        """Deleting a lifecycle sandbox should delete its worker sandboxes too."""

        async def get_or_create(name, *args, **kwargs):
            sandbox = MagicMock()
            sandbox.name = name
            return sandbox

        service = FakeSandboxService()
        service.get_or_create = AsyncMock(side_effect=get_or_create)
        service.delete = AsyncMock()
        manager = SandboxManager(service)

        with (
            patch.dict(
                "os.environ", {"XAGENT_SANDBOX_MAX_CONCURRENCY": "2"}, clear=True
            ),
            patch(
                "xagent.web.sandbox_manager.build_code_mount_volumes",
                return_value=[("/repo/src", "/app/src", "ro")],
            ),
        ):
            provider = await manager.create_lease_provider(
                "user",
                "42",
                mount_intent=SandboxMountIntent(mount_root=str(tmp_path / "user_42")),
            )
            async with provider.lease(concurrency_safe=True):
                pass
            async with provider.lease(concurrency_safe=True):
                pass

            await manager.delete_sandbox("user", "42")

        deleted_names = {call.args[0] for call in service.delete.await_args_list}
        assert deleted_names == {
            "user::42",
            "user::42::worker::0",
            "user::42::worker::1",
        }
        assert "user::42" not in manager._cache
        assert "user::42::worker::0" not in manager._cache
        assert "user::42::worker::1" not in manager._cache

    @pytest.mark.asyncio
    async def test_delete_sandbox_removes_persisted_worker_sandboxes(self):
        """Delete should include worker sandboxes discovered from the service."""
        service = FakeSandboxService()
        service.delete = AsyncMock()
        service.list_sandboxes = AsyncMock(
            return_value=[
                _sandbox_info("user::42"),
                _sandbox_info("user::42::worker::0"),
                _sandbox_info("user::42::worker::1"),
                _sandbox_info("user::420::worker::0"),
                _sandbox_info("tools::42::worker::0"),
            ]
        )
        manager = SandboxManager(service)

        await manager.delete_sandbox("user", "42")

        deleted_names = {call.args[0] for call in service.delete.await_args_list}
        assert deleted_names == {
            "user::42",
            "user::42::worker::0",
            "user::42::worker::1",
        }


class TestSandboxManagerWarmup:
    """Test sandbox warmup functionality."""

    @pytest.mark.asyncio
    async def test_warmup_uses_empty_config(self):
        """Test warmup uses empty config to avoid unnecessary mounts."""
        mock_service = FakeSandboxService()
        manager = SandboxManager(mock_service)

        # Mock the service methods
        mock_sandbox = MagicMock()
        mock_sandbox.__aenter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__aexit__ = MagicMock(return_value=None)
        mock_service.get_or_create = MagicMock(return_value=mock_sandbox)
        mock_service.delete = MagicMock(return_value=None)

        # Set environment vars that would normally trigger mounts
        with patch.dict(
            "os.environ",
            {"SANDBOX_VOLUMES": "/nonexistent:/path:ro", "SANDBOX_ENV": "TEST=value"},
            clear=False,
        ):
            await manager.warmup()

        # Verify get_or_create was called with empty config (no volumes/env)
        mock_service.get_or_create.assert_called_once()
        call_args = mock_service.get_or_create.call_args
        config = call_args[1]["config"]

        # Verify warmup config is empty (no volumes/env)
        assert config.volumes is None
        assert config.env is None
        # Should have default cpus/memory from SandboxConfig
        assert config.cpus == 1
        assert config.memory == 512


class TestSandboxActivityTracking:
    """Test attach/release ref-counting and last-activity tracking."""

    @staticmethod
    def _make_manager() -> SandboxManager:
        service = FakeSandboxService()
        service.list_sandboxes = AsyncMock(return_value=[])
        service.delete = AsyncMock()
        return SandboxManager(service)

    @pytest.mark.asyncio
    async def test_attach_without_provider_returns_false(self):
        manager = self._make_manager()

        assert await manager.attach("user", "7") is False
        assert manager.ref_count("user", "7") == 0

    @pytest.mark.asyncio
    async def test_attach_release_ref_count_round_trip(self):
        manager = self._make_manager()
        manager._lease_providers["user::7"] = MagicMock()

        assert await manager.attach("user", "7") is True
        assert await manager.attach("user", "7") is True
        assert manager.ref_count("user", "7") == 2

        assert await manager.release("user", "7") is False
        assert manager.ref_count("user", "7") == 1
        assert "user::7" in manager._lease_providers

        assert await manager.release("user", "7") is True
        assert manager.ref_count("user", "7") == 0
        assert "user::7" not in manager._lease_providers

    @pytest.mark.asyncio
    async def test_release_without_attach_is_ignored(self):
        """A mismatched release must not tear down a fresh, not-yet-attached
        provider or re-run the worker cleanup."""
        manager = self._make_manager()
        manager._lease_providers["user::7"] = MagicMock()
        manager.delete_worker_sandboxes = AsyncMock()
        evictions: list[str] = []

        released = await manager.release(
            "user", "7", on_last_release=lambda: evictions.append("user::7")
        )

        assert released is False
        assert "user::7" in manager._lease_providers
        assert evictions == []
        manager.delete_worker_sandboxes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_release_to_zero_runs_cleanup_exactly_once(self):
        manager = self._make_manager()
        manager._lease_providers["user::7"] = MagicMock()
        manager.delete_worker_sandboxes = AsyncMock()
        evictions: list[str] = []

        attach_count = 5
        for _ in range(attach_count):
            await manager.attach("user", "7")

        results = await asyncio.gather(
            *(
                manager.release(
                    "user", "7", on_last_release=lambda: evictions.append("user::7")
                )
                for _ in range(attach_count)
            )
        )

        assert sum(results) == 1
        assert evictions == ["user::7"]
        manager.delete_worker_sandboxes.assert_awaited_once_with("user", "7")
        assert manager.ref_count("user", "7") == 0

    @pytest.mark.asyncio
    async def test_concurrent_attach_release_is_race_free(self):
        manager = self._make_manager()
        manager._lease_providers["user::7"] = MagicMock()
        manager.delete_worker_sandboxes = AsyncMock()

        async def attach_then_release() -> None:
            assert await manager.attach("user", "7")
            await asyncio.sleep(0)
            await manager.release("user", "7")

        # Keep one attachment alive so the provider survives the churn.
        await manager.attach("user", "7")
        await asyncio.gather(*(attach_then_release() for _ in range(20)))

        assert manager.ref_count("user", "7") == 1
        assert "user::7" in manager._lease_providers
        manager.delete_worker_sandboxes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_worker_lifecycle_id_maps_to_primary_activity(self):
        manager = self._make_manager()
        manager._lease_providers["user::7"] = MagicMock()

        assert await manager.attach("user", "7::worker::0") is True
        assert manager.ref_count("user", "7") == 1
        assert manager.ref_count("user", "7::worker::3") == 1

    @pytest.mark.asyncio
    async def test_last_activity_defaults_to_startup(self):
        manager = self._make_manager()

        assert manager.last_activity_at("user", "7") == manager._startup_monotonic

    @pytest.mark.asyncio
    async def test_attach_and_release_bump_last_activity(self):
        manager = self._make_manager()
        manager._lease_providers["user::7"] = MagicMock()

        baseline = manager.last_activity_at("user", "7")
        await manager.attach("user", "7")
        after_attach = manager.last_activity_at("user", "7")
        assert after_attach >= baseline

        await manager.release("user", "7")
        assert manager.last_activity_at("user", "7") >= after_attach

    @pytest.mark.asyncio
    async def test_lifecycle_lock_entries_do_not_leak(self):
        manager = self._make_manager()
        manager._lease_providers["user::7"] = MagicMock()

        await manager.attach("user", "7")
        await manager.release("user", "7")
        await manager.get_or_create_lease_provider("user", "8")

        assert manager._lifecycle_locks == {}


class TestGetOrCreateLeaseProvider:
    """Test the cached lease provider entry point."""

    @staticmethod
    def _make_manager() -> SandboxManager:
        service = FakeSandboxService()
        service.list_sandboxes = AsyncMock(return_value=[])
        service.delete = AsyncMock()
        return SandboxManager(service)

    @pytest.mark.asyncio
    async def test_returns_cached_provider_for_same_lifecycle(self):
        manager = self._make_manager()
        provider = MagicMock()
        manager._create_lease_provider_locked = AsyncMock(return_value=provider)

        first = await manager.get_or_create_lease_provider("user", "7")
        second = await manager.get_or_create_lease_provider("user", "7")

        assert first is provider
        assert second is provider
        manager._create_lease_provider_locked.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_requests_create_single_provider(self):
        manager = self._make_manager()

        async def create_provider(*_args, **_kwargs):
            await asyncio.sleep(0)
            return MagicMock()

        manager._create_lease_provider_locked = AsyncMock(side_effect=create_provider)

        providers = await asyncio.gather(
            *(manager.get_or_create_lease_provider("user", "7") for _ in range(5))
        )

        assert len({id(p) for p in providers}) == 1
        manager._create_lease_provider_locked.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_release_to_zero_drops_cache_and_recreates(self):
        manager = self._make_manager()
        providers = [MagicMock(), MagicMock()]
        manager._create_lease_provider_locked = AsyncMock(side_effect=providers)

        first = await manager.get_or_create_lease_provider("user", "7")
        await manager.attach("user", "7")
        await manager.release("user", "7")
        second = await manager.get_or_create_lease_provider("user", "7")

        assert first is providers[0]
        assert second is providers[1]
