"""Regression tests for ToolFactory release boundaries.

Issue #889 requires the config's DB connection to be released again before
sandbox workspace setup because override/allowlist reads may reopen it.
"""

import logging

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.tools.adapters.vibe.config import (
    ToolFactoryRuntimeSessionBoundaryError,
    run_with_tool_runtime_cleanup,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry


class _FakeSandbox:
    pass


class _HostMountedSandboxProvider:
    def __init__(self) -> None:
        self.primary_sandbox = _FakeSandbox()
        self.checked_directories: list[str] = []

    def workspace_dirs_are_host_mounted(self, directories) -> bool:
        self.checked_directories = list(directories)
        return True


class _FailingHostMountedSandboxProvider:
    def __init__(self) -> None:
        self.primary_sandbox = _FakeSandbox()

    def workspace_dirs_are_host_mounted(self, directories) -> bool:
        raise RuntimeError("coverage probe sentinel")


class _NotHostMountedSandboxProvider:
    def __init__(self) -> None:
        self.primary_sandbox = _FakeSandbox()

    def workspace_dirs_are_host_mounted(self, directories) -> bool:
        return False


class _FakeConfig:
    def __init__(self, calls):
        self._calls = calls

    def get_tool_selection_spec(self):
        return None

    def get_allowed_tools(self):
        return None

    def get_user_tool_overrides(self):
        self._calls.append("load_overrides")
        return {}

    def get_user_tool_allowlist(self):
        self._calls.append("load_allowlist")
        return None

    def release_db_connection(self):
        self._calls.append("release_db")

    def get_sandbox(self):
        return _FakeSandbox()

    def get_workspace_config(self):
        # ``_mock_`` selects MockWorkspace: no on-disk directories.
        return {"task_id": "_mock_", "base_dir": "/tmp"}

    def get_max_output_length(self):
        return 10000

    def get_max_field_count(self):
        return 100

    def get_max_recursion_depth(self):
        return 5


class _FailingPrepareConfig:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    async def prepare_factory_runtime(self) -> None:
        self._calls.append("prepare")
        raise RuntimeError("prepare failed")

    def release_prepared_factory_runtime(self) -> None:
        self._calls.append("release")


class _FailingPrepareAndReleaseConfig(_FailingPrepareConfig):
    def release_prepared_factory_runtime(self) -> None:
        super().release_prepared_factory_runtime()
        raise ValueError("release failed")


class _FailingReleaseConfig:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def release_prepared_factory_runtime(self) -> None:
        self._calls.append("release")
        raise ValueError("release failed")


class _FailingVerifiedHandoffConfig:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def handoff_factory_runtime(self) -> None:
        self._calls.append("handoff")
        raise ValueError("handoff failed")


class _AbortableFailingPrepareConfig(_FailingPrepareConfig):
    def abort_factory_runtime(self) -> None:
        self._calls.append("abort")

    def handoff_factory_runtime(self) -> None:
        self._calls.append("handoff")


class _CleanupBaseException(BaseException):
    pass


@pytest.mark.asyncio
async def test_shared_cleanup_surfaces_a_stable_boundary_error_after_success():
    cleanup_fault = ValueError("private cleanup detail")

    async def body() -> str:
        return "done"

    def cleanup() -> None:
        raise cleanup_fault

    with pytest.raises(ToolFactoryRuntimeSessionBoundaryError) as caught:
        await run_with_tool_runtime_cleanup(
            body, cleanup, logger=logging.getLogger(__name__)
        )

    assert str(caught.value) == "Tool runtime cleanup could not be completed."
    assert caught.value.__cause__ is cleanup_fault


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fault", [SystemExit("exit"), _CleanupBaseException()])
async def test_shared_cleanup_preserves_successful_body_base_exception_identity(
    cleanup_fault,
):
    async def body() -> None:
        return None

    def cleanup() -> None:
        raise cleanup_fault

    with pytest.raises(type(cleanup_fault)) as caught:
        await run_with_tool_runtime_cleanup(
            body, cleanup, logger=logging.getLogger(__name__)
        )

    assert caught.value is cleanup_fault


@pytest.mark.asyncio
async def test_shared_cleanup_preserves_primary_base_exception_identity(caplog):
    primary = KeyboardInterrupt("primary sentinel")
    cleanup_fault = SystemExit("cleanup sentinel")

    async def body() -> None:
        raise primary

    def cleanup() -> None:
        raise cleanup_fault

    with pytest.raises(KeyboardInterrupt) as caught:
        await run_with_tool_runtime_cleanup(
            body,
            cleanup,
            logger=logging.getLogger("xagent.runtime.cleanup.test"),
        )

    assert caught.value is primary
    assert (
        "Tool runtime cleanup failed after the primary operation failed" in caplog.text
    )


@pytest.mark.asyncio
async def test_shared_cleanup_returns_when_body_and_cleanup_succeed():
    calls: list[str] = []

    async def body() -> str:
        calls.append("body")
        return "done"

    def cleanup() -> None:
        calls.append("cleanup")

    assert (
        await run_with_tool_runtime_cleanup(
            body, cleanup, logger=logging.getLogger(__name__)
        )
        == "done"
    )
    assert calls == ["body", "cleanup"]


@pytest.mark.asyncio
async def test_release_prepared_runtime_when_prepare_fails():
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="prepare failed"):
        await ToolFactory.create_all_tools(_FailingPrepareConfig(calls))

    assert calls == ["prepare", "release"]


@pytest.mark.asyncio
async def test_prepare_failure_prefers_concrete_abort_over_success_handoff():
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="prepare failed"):
        await ToolFactory.create_all_tools(_AbortableFailingPrepareConfig(calls))

    assert calls == ["prepare", "abort"]


@pytest.mark.asyncio
async def test_prepare_error_wins_when_release_also_fails(caplog):
    calls: list[str] = []

    with pytest.raises(RuntimeError, match="prepare failed"):
        await ToolFactory.create_all_tools(_FailingPrepareAndReleaseConfig(calls))

    assert calls == ["prepare", "release"]
    assert "Failed to finalize tool-factory runtime" in caplog.text


@pytest.mark.asyncio
async def test_release_error_propagates_without_primary_error(monkeypatch):
    calls: list[str] = []

    async def build_tools(config, apply_user_override_filter=True):
        calls.append("build")
        return []

    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)

    with pytest.raises(ToolFactoryRuntimeSessionBoundaryError) as caught:
        await ToolFactory.create_all_tools(_FailingReleaseConfig(calls))

    assert str(caught.value) == "Tool runtime cleanup could not be completed."
    assert calls == ["build", "release"]


@pytest.mark.asyncio
async def test_primary_build_error_wins_when_verified_handoff_fails(
    monkeypatch, caplog
):
    calls: list[str] = []

    async def build_tools(config, apply_user_override_filter=True):
        calls.append("build")
        raise RuntimeError("build sentinel")

    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)

    with pytest.raises(RuntimeError, match="build sentinel"):
        await ToolFactory.create_all_tools(_FailingVerifiedHandoffConfig(calls))

    assert calls == ["build", "handoff"]
    assert "Failed to finalize tool-factory runtime" in caplog.text


@pytest.mark.asyncio
async def test_real_web_config_primary_error_identity_wins_over_handoff_fault(
    monkeypatch, tmp_path, caplog
):
    from xagent.web.models.tool_config import ToolConfig
    from xagent.web.tools.config import WebToolConfig

    engine = create_engine(
        f"sqlite:///{tmp_path / 'handoff-primary.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    live_db = factory()
    config = WebToolConfig(db=live_db, db_factory=factory, request=None, user_id=1)
    sentinel = RuntimeError("build sentinel")
    abort_fault = ValueError("independent abort fault")
    real_abort = WebToolConfig.abort_factory_runtime

    async def prepare(_config):
        return None

    async def build_tools(config, apply_user_override_filter=True):
        config.db.query(ToolConfig).all()
        raise sentinel

    def failing_abort(config):
        real_abort(config)
        raise abort_fault

    monkeypatch.setattr(WebToolConfig, "prepare_factory_runtime", prepare)
    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)
    monkeypatch.setattr(WebToolConfig, "abort_factory_runtime", failing_abort)
    try:
        with pytest.raises(RuntimeError) as caught:
            await ToolFactory.create_all_tools(config)

        assert caught.value is sentinel
        assert engine.pool.checkedout() == 0
        assert "Failed to finalize tool-factory runtime" in caplog.text
    finally:
        live_db.close()
        config.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_primary_error_identity_wins_over_real_session_boundary_failure(
    monkeypatch, tmp_path, caplog
):
    from xagent.web.models.tool_config import ToolConfig
    from xagent.web.models.user import User
    from xagent.web.tools.config import WebToolConfig

    engine = create_engine(
        f"sqlite:///{tmp_path / 'handoff-boundary-primary.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    User.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    live_db = factory()
    config = WebToolConfig(db=live_db, db_factory=factory, request=object(), user_id=1)
    sentinel = RuntimeError("primary sentinel")

    async def prepare(_config):
        return None

    async def build_tools(_config, apply_user_override_filter=True):
        raise sentinel

    monkeypatch.setattr(WebToolConfig, "prepare_factory_runtime", prepare)
    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", build_tools)
    try:
        live_db.query(ToolConfig).all()
        live_db.add(
            User(username="boundary-pending", password_hash="hash", is_admin=False)
        )
        assert engine.pool.checkedout() == 1

        with pytest.raises(RuntimeError) as caught:
            await ToolFactory.create_all_tools(config)

        assert caught.value is sentinel
        assert config._live_db is live_db
        assert engine.pool.checkedout() == 1
        assert "Failed to finalize tool-factory runtime" in caplog.text
    finally:
        live_db.rollback()
        live_db.close()
        config.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_real_prepare_then_build_failure_aborts_without_retaining_runtime(
    monkeypatch, tmp_path
):
    from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
    from xagent.web.models.tool_config import ToolConfig
    from xagent.web.tools.config import WebToolConfig

    engine = create_engine(
        f"sqlite:///{tmp_path / 'abort-runtime.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    live_db = factory()
    request = object()
    user = object()
    config = WebToolConfig(
        db=live_db,
        db_factory=factory,
        request=request,
        user=user,
        user_id=1,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=[]),
    )
    sentinel = RuntimeError("build after prepare sentinel")

    async def fail_build(_config, apply_user_override_filter=True):
        raise sentinel

    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", fail_build)
    try:
        with pytest.raises(RuntimeError) as caught:
            await ToolFactory.create_all_tools(config)

        assert caught.value is sentinel
        assert config._factory_runtime_snapshot is None
        assert config._live_db is None
        assert config.request is None
        assert config._user is None
        assert engine.pool.checkedout() == 0
    finally:
        live_db.close()
        config.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_partial_prepare_base_exception_uses_real_abort_not_handoff(
    monkeypatch, tmp_path
):
    from xagent.web.models.tool_config import ToolConfig
    from xagent.web.tools.config import WebToolConfig

    class PrepareFailure(BaseException):
        pass

    engine = create_engine(
        f"sqlite:///{tmp_path / 'partial-prepare-abort.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    live_db = factory()
    request = object()
    user = object()
    config = WebToolConfig(
        db=live_db,
        db_factory=factory,
        request=request,
        user=user,
        user_id=1,
    )
    partial_snapshot = object()
    primary = PrepareFailure()
    handoff_calls: list[str] = []

    async def partial_prepare(candidate):
        candidate._factory_runtime_snapshot = partial_snapshot
        raise primary

    def unexpected_handoff(_candidate):
        handoff_calls.append("handoff")

    monkeypatch.setattr(WebToolConfig, "prepare_factory_runtime", partial_prepare)
    monkeypatch.setattr(WebToolConfig, "handoff_factory_runtime", unexpected_handoff)
    try:
        with pytest.raises(PrepareFailure) as caught:
            await ToolFactory.create_all_tools(config)

        assert caught.value is primary
        assert handoff_calls == []
        assert config._factory_runtime_snapshot is None
        assert config._live_db is None
        assert config.request is None
        assert config._user is None
    finally:
        live_db.close()
        config.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_release_db_before_sandbox_workspace_setup(monkeypatch):
    calls: list[str] = []

    async def fake_create_registered_tools(config):
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        staticmethod(fake_create_registered_tools),
    )

    from xagent.core.tools.adapters.vibe.sandboxed_tool import (
        sandboxed_tool_wrapper,
    )

    async def fake_create_workspace_in_sandbox(sandbox, workspace):
        calls.append("sandbox_exec")

    monkeypatch.setattr(
        sandboxed_tool_wrapper,
        "create_workspace_in_sandbox",
        fake_create_workspace_in_sandbox,
    )

    await ToolFactory.create_all_tools(_FakeConfig(calls))

    assert "sandbox_exec" in calls
    assert "release_db" in calls
    # The DB release happens after the last config DB reads (overrides /
    # allowlist) and before the sandbox workspace exec.
    assert calls.index("release_db") > calls.index("load_overrides")
    assert calls.index("release_db") > calls.index("load_allowlist")
    assert calls.index("release_db") < calls.index("sandbox_exec")


@pytest.mark.asyncio
async def test_host_mounted_workspace_skips_sandbox_exec(monkeypatch, tmp_path):
    calls: list[str] = []
    provider = _HostMountedSandboxProvider()

    async def fake_create_registered_tools(config):
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        staticmethod(fake_create_registered_tools),
    )

    from xagent.core.tools.adapters.vibe.sandboxed_tool import (
        sandboxed_tool_wrapper,
    )

    async def unexpected_create_workspace_in_sandbox(sandbox, workspace):
        raise AssertionError("host-mounted workspace must not run sandbox mkdir")

    monkeypatch.setattr(
        sandboxed_tool_wrapper,
        "create_workspace_in_sandbox",
        unexpected_create_workspace_in_sandbox,
    )

    config = _FakeConfig(calls)
    config.get_sandbox = lambda: provider
    config.get_workspace_config = lambda: {
        "task_id": "task-1",
        "base_dir": str(tmp_path),
    }
    await ToolFactory.create_all_tools(config)

    assert provider.checked_directories
    assert "release_db" in calls


@pytest.mark.asyncio
async def test_mock_workspace_keeps_sandbox_exec_when_mount_is_covered(monkeypatch):
    """Virtual workspace paths are not made real by a covering host mount."""
    calls: list[str] = []
    provider = _HostMountedSandboxProvider()

    async def fake_create_registered_tools(config):
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        staticmethod(fake_create_registered_tools),
    )

    from xagent.core.tools.adapters.vibe.sandboxed_tool import (
        sandboxed_tool_wrapper,
    )

    async def fake_create_workspace_in_sandbox(sandbox, workspace):
        calls.append("sandbox_exec")

    monkeypatch.setattr(
        sandboxed_tool_wrapper,
        "create_workspace_in_sandbox",
        fake_create_workspace_in_sandbox,
    )

    config = _FakeConfig(calls)
    config.get_sandbox = lambda: provider
    await ToolFactory.create_all_tools(config)

    assert "sandbox_exec" in calls
    assert provider.checked_directories == []


@pytest.mark.asyncio
async def test_uncovered_workspace_falls_back_to_sandbox_exec(monkeypatch, tmp_path):
    calls: list[str] = []
    provider = _NotHostMountedSandboxProvider()

    async def fake_create_registered_tools(config):
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        staticmethod(fake_create_registered_tools),
    )

    from xagent.core.tools.adapters.vibe.sandboxed_tool import (
        sandboxed_tool_wrapper,
    )

    async def fake_create_workspace_in_sandbox(sandbox, workspace):
        calls.append("sandbox_exec")

    monkeypatch.setattr(
        sandboxed_tool_wrapper,
        "create_workspace_in_sandbox",
        fake_create_workspace_in_sandbox,
    )

    config = _FakeConfig(calls)
    config.get_sandbox = lambda: provider
    config.get_workspace_config = lambda: {
        "task_id": "task-1",
        "base_dir": str(tmp_path),
    }
    await ToolFactory.create_all_tools(config)

    assert "sandbox_exec" in calls


@pytest.mark.asyncio
async def test_host_mount_probe_failure_falls_back_to_sandbox_exec(
    monkeypatch, caplog, tmp_path
):
    calls: list[str] = []
    provider = _FailingHostMountedSandboxProvider()

    async def fake_create_registered_tools(config):
        return []

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        staticmethod(fake_create_registered_tools),
    )

    from xagent.core.tools.adapters.vibe.sandboxed_tool import (
        sandboxed_tool_wrapper,
    )

    async def fake_create_workspace_in_sandbox(sandbox, workspace):
        calls.append("sandbox_exec")

    monkeypatch.setattr(
        sandboxed_tool_wrapper,
        "create_workspace_in_sandbox",
        fake_create_workspace_in_sandbox,
    )

    config = _FakeConfig(calls)
    config.get_sandbox = lambda: provider
    config.get_workspace_config = lambda: {
        "task_id": "task-1",
        "base_dir": str(tmp_path),
    }
    await ToolFactory.create_all_tools(config)

    assert "sandbox_exec" in calls
    assert "Workspace host-mount coverage check failed" in caplog.text
