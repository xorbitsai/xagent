"""Test sandbox manager functionality."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from xagent.web.sandbox_manager import (
    SandboxManager,
    _create_boxlite_service,
    _create_sandbox_service,
    get_sandbox_manager,
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

    def test_boxlite_default(self):
        """Test default implementation is boxlite."""
        with (
            patch.dict("os.environ", {"SANDBOX_ENABLED": "true"}, clear=False),
            patch("xagent.web.sandbox_manager._create_boxlite_service") as mock_create,
        ):
            mock_create.return_value = MagicMock()
            result = _create_sandbox_service()
        assert result is not None
        mock_create.assert_called_once()

    def test_unknown_implementation_falls_back_to_boxlite(self):
        """Test unknown implementation falls back to boxlite."""
        with (
            patch.dict(
                "os.environ",
                {"SANDBOX_ENABLED": "true", "SANDBOX_IMPLEMENTATION": "unknown"},
                clear=False,
            ),
            patch("xagent.web.sandbox_manager._create_boxlite_service") as mock_create,
        ):
            mock_create.return_value = MagicMock()
            _create_sandbox_service()
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
        mock_service = MagicMock()
        with patch(
            "xagent.web.sandbox_manager._create_sandbox_service",
            return_value=mock_service,
        ):
            result = get_sandbox_manager()
        assert isinstance(result, SandboxManager)

    def test_singleton_returns_same_instance(self):
        """Test singleton pattern returns same instance."""
        mock_service = MagicMock()
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
        mock_service = MagicMock()
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
                "xagent.sandbox.BoxliteSandboxService", return_value=MagicMock()
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
