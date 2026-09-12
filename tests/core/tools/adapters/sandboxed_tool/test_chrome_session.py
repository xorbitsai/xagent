from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock

import pytest

from xagent.core.tools.adapters.vibe.sandboxed_tool.chrome_session import (
    CHROME_DEVTOOLS_PACKAGE,
    ChromeDaemonClient,
    ChromeDaemonLaunchSpec,
    ChromeExecutionScope,
    ChromeExecutionSession,
    ChromeExecutionSessionPool,
    ChromeSandboxHandle,
    ChromeSessionContractError,
    ChromeSessionTransportError,
)
from xagent.sandbox.base import ExecResult, Sandbox


def _connection(*, suffix: list[str] | None = None):
    return {
        "transport": "stdio",
        "command": "npx",
        "args": [
            "-y",
            "--prefer-offline",
            CHROME_DEVTOOLS_PACKAGE,
            "--headless",
            "--isolated",
            *(suffix or []),
        ],
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _scope(value: str) -> ChromeExecutionScope:
    return ChromeExecutionScope(key=(value,), digest=_digest(value))


def _sandbox() -> AsyncMock:
    sandbox = AsyncMock(spec=Sandbox)
    sandbox.name = "chrome-execution::test"
    return sandbox


def _healthy_status(*, suffix: list[str] | None = None):
    return {
        "version": "1.6.0",
        "args": [
            "--channel",
            "stable",
            "--headless",
            "--isolated",
            *(suffix or []),
            "--viaCli",
            "--experimentalStructuredContent",
        ],
    }


class TestChromeDaemonLaunchSpec:
    def test_accepts_only_the_canonical_pinned_npx_shape(self):
        spec = ChromeDaemonLaunchSpec.from_connection(
            _connection(suffix=["--no-usage-statistics"])
        )
        assert spec.server_args == (
            "--headless",
            "--isolated",
            "--no-usage-statistics",
        )

    @pytest.mark.parametrize(
        "connection",
        [
            {"transport": "sse", "command": "npx", "args": []},
            {"transport": "stdio", "command": "node", "args": []},
            {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "chrome-devtools-mcp@latest"],
            },
            {
                "transport": "stdio",
                "command": "npx",
                "args": ["--prefer-offline", "-y", CHROME_DEVTOOLS_PACKAGE],
            },
            {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "--prefer-offline", CHROME_DEVTOOLS_PACKAGE],
            },
            _connection(suffix=["--sessionId=unmanaged"]),
            _connection(suffix=["--chrome-arg=--user-data-dir=/shared"]),
        ],
    )
    def test_rejects_noncanonical_or_unisolated_launches(self, connection):
        with pytest.raises(ChromeSessionContractError):
            ChromeDaemonLaunchSpec.from_connection(connection)


class TestChromeDaemonClient:
    @pytest.mark.asyncio
    async def test_every_operation_stays_inside_sandbox_and_passes_safe_env(self):
        sandbox = _sandbox()
        sandbox.exec.side_effect = [
            ExecResult(exit_code=0, stdout="", stderr=""),
            ExecResult(exit_code=0, stdout="16", stderr=""),
            ExecResult(exit_code=0, stdout="", stderr=""),
        ]
        sandbox.read_file.return_value = '{"running":true}'
        client = ChromeDaemonClient(sandbox, session_id="a" * 32)

        result = await client.status()

        assert result == {"running": True}
        run = sandbox.exec.await_args_list[0]
        assert run.args[0] == "python"
        assert "chrome_daemon_runner.py" in run.args[1]
        assert run.kwargs["env"] == {
            "NPM_CONFIG_CACHE": "/opt/npm-cache",
            "CHROME_DEVTOOLS_MCP_NO_UPDATE_CHECKS": "1",
        }
        assert sandbox.exec.await_args_list[1].args[:3] == ("stat", "-c", "%s")
        assert sandbox.exec.await_args_list[2].args[:2] == ("rm", "-f")

    @pytest.mark.asyncio
    async def test_sandbox_failure_has_no_direct_process_fallback(self):
        sandbox = _sandbox()
        sandbox.exec.side_effect = [
            ExecResult(exit_code=17, stdout="secret", stderr="secret"),
            ExecResult(exit_code=0, stdout="", stderr=""),
        ]
        client = ChromeDaemonClient(sandbox, session_id="b" * 32)

        with pytest.raises(ChromeSessionTransportError, match="exit code 17") as exc:
            await client.status()

        assert sandbox.exec.await_count == 2
        assert "secret" not in str(exc.value)

    @pytest.mark.asyncio
    async def test_result_size_is_bounded_before_reading(self):
        sandbox = _sandbox()
        sandbox.exec.side_effect = [
            ExecResult(exit_code=0, stdout="", stderr=""),
            ExecResult(exit_code=0, stdout=str(64 * 1024 * 1024 + 1), stderr=""),
            ExecResult(exit_code=0, stdout="", stderr=""),
        ]
        client = ChromeDaemonClient(sandbox, session_id="b" * 32)

        with pytest.raises(ChromeSessionTransportError, match="size limit"):
            await client.status()

        sandbox.read_file.assert_not_awaited()

    def test_rejects_untrusted_session_names(self):
        with pytest.raises(ValueError, match="lowercase hex"):
            ChromeDaemonClient(_sandbox(), session_id="../../shared")


class TestChromeExecutionSession:
    @pytest.mark.asyncio
    async def test_reuses_one_healthy_daemon_and_serializes_calls(self):
        session = ChromeExecutionSession(
            ChromeSandboxHandle(sandbox=_sandbox(), delete=AsyncMock()),
            session_id="c" * 32,
            launch=ChromeDaemonLaunchSpec.from_connection(_connection()),
        )
        client = AsyncMock()
        client.start.return_value = _healthy_status()
        client.status.return_value = {"running": True, "status": _healthy_status()}
        client.invoke_tool.side_effect = [{"value": 1}, {"value": 2}]
        session._client = client

        first, second = await asyncio.gather(
            session.invoke_tool("navigate_page", {"url": "https://one.invalid"}),
            session.invoke_tool("take_snapshot", {}),
        )

        assert (first, second) == ({"value": 1}, {"value": 2})
        client.start.assert_awaited_once()
        client.status.assert_awaited_once()
        assert client.invoke_tool.await_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_deletes_sandbox_even_when_daemon_stop_fails(self, caplog):
        delete = AsyncMock()
        session = ChromeExecutionSession(
            ChromeSandboxHandle(sandbox=_sandbox(), delete=delete),
            session_id="d" * 32,
            launch=ChromeDaemonLaunchSpec.from_connection(_connection()),
        )
        session._started = True
        client = AsyncMock()
        client.stop.side_effect = RuntimeError("stop failed")
        session._client = client

        await session.close()
        await session.close()

        client.stop.assert_awaited_once()
        delete.assert_awaited_once()
        assert any(record.exc_info is not None for record in caplog.records)

    @pytest.mark.asyncio
    async def test_cleanup_continues_after_waiter_cancellation(self):
        entered = asyncio.Event()
        release = asyncio.Event()
        deleted = asyncio.Event()

        async def delete():
            entered.set()
            await release.wait()
            deleted.set()

        session = ChromeExecutionSession(
            ChromeSandboxHandle(sandbox=_sandbox(), delete=delete),
            session_id="e" * 32,
            launch=ChromeDaemonLaunchSpec.from_connection(_connection()),
        )
        waiter = asyncio.create_task(session.close())
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await asyncio.wait_for(deleted.wait(), timeout=1)

    @pytest.mark.asyncio
    async def test_status_mismatch_fails_closed(self):
        delete = AsyncMock()
        session = ChromeExecutionSession(
            ChromeSandboxHandle(sandbox=_sandbox(), delete=delete),
            session_id="f" * 32,
            launch=ChromeDaemonLaunchSpec.from_connection(_connection()),
        )
        session._started = True
        client = AsyncMock()
        client.status.return_value = {
            "running": True,
            "status": {**_healthy_status(), "version": "unexpected"},
        }
        session._client = client

        with pytest.raises(ChromeSessionContractError, match="status does not match"):
            await session.invoke_tool("take_snapshot", {})

        client.invoke_tool.assert_not_awaited()

    def test_real_daemon_status_defaults_and_serialization_are_accepted(self):
        session = ChromeExecutionSession(
            ChromeSandboxHandle(sandbox=_sandbox(), delete=AsyncMock()),
            session_id="f" * 32,
            launch=ChromeDaemonLaunchSpec.from_connection(
                _connection(suffix=["--chrome-arg=--disable-gpu"])
            ),
        )

        assert session._status_matches_launch(
            _healthy_status(suffix=["--chrome-arg", "--disable-gpu"])
        )

    @pytest.mark.parametrize(
        "unsafe_args",
        [
            ["--user-data-dir=/shared"],
            ["--chrome-arg", "--profile-directory=shared"],
        ],
    )
    def test_status_rejects_profile_or_connection_overrides(self, unsafe_args):
        session = ChromeExecutionSession(
            ChromeSandboxHandle(sandbox=_sandbox(), delete=AsyncMock()),
            session_id="f" * 32,
            launch=ChromeDaemonLaunchSpec.from_connection(_connection()),
        )
        status = _healthy_status()
        status["args"].extend(unsafe_args)

        assert not session._status_matches_launch(status)


class TestChromeExecutionSessionPool:
    @pytest.mark.asyncio
    async def test_factory_is_once_per_opaque_full_execution_digest(self):
        factory = AsyncMock(
            side_effect=lambda _digest: ChromeSandboxHandle(
                sandbox=_sandbox(), delete=AsyncMock()
            )
        )
        pool = ChromeExecutionSessionPool(factory)
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        one = _scope("exact execution one")
        two = _scope("exact execution two")

        first, duplicate = await asyncio.gather(
            pool.get_or_create(one, launch), pool.get_or_create(one, launch)
        )
        distinct = await pool.get_or_create(two, launch)

        assert first is duplicate
        assert distinct is not first
        assert factory.await_args_list[0].args == (one.digest,)
        assert factory.await_args_list[1].args == (two.digest,)

    @pytest.mark.asyncio
    async def test_factory_failure_propagates_without_fallback(self):
        factory = AsyncMock(side_effect=RuntimeError("sandbox unavailable"))
        pool = ChromeExecutionSessionPool(factory)
        with pytest.raises(RuntimeError, match="sandbox unavailable"):
            await pool.get_or_create(
                _scope("execution"),
                ChromeDaemonLaunchSpec.from_connection(_connection()),
            )

    @pytest.mark.asyncio
    async def test_same_scope_rejects_launch_spec_substitution(self):
        pool = ChromeExecutionSessionPool(
            AsyncMock(
                return_value=ChromeSandboxHandle(sandbox=_sandbox(), delete=AsyncMock())
            )
        )
        scope = _scope("execution")
        await pool.get_or_create(
            scope, ChromeDaemonLaunchSpec.from_connection(_connection())
        )

        with pytest.raises(ChromeSessionContractError, match="cannot change"):
            await pool.get_or_create(
                scope,
                ChromeDaemonLaunchSpec.from_connection(
                    _connection(suffix=["--no-usage-statistics"])
                ),
            )

    @pytest.mark.asyncio
    async def test_recreate_waits_until_old_sandbox_is_deleted(self):
        delete_entered = asyncio.Event()
        allow_delete = asyncio.Event()

        async def slow_delete():
            delete_entered.set()
            await allow_delete.wait()

        factory = AsyncMock(
            side_effect=[
                ChromeSandboxHandle(sandbox=_sandbox(), delete=slow_delete),
                ChromeSandboxHandle(sandbox=_sandbox(), delete=AsyncMock()),
            ]
        )
        pool = ChromeExecutionSessionPool(factory)
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        scope = _scope("execution")
        first = await pool.get_or_create(scope, launch)

        closing = asyncio.create_task(pool.close(scope))
        await delete_entered.wait()
        recreating = asyncio.create_task(pool.get_or_create(scope, launch))
        await asyncio.sleep(0)
        assert factory.await_count == 1

        allow_delete.set()
        await closing
        second = await recreating

        assert second is not first
        assert factory.await_count == 2

    def test_rejects_partial_or_non_digest_scope(self):
        with pytest.raises(ValueError, match="SHA-256"):
            ChromeExecutionScope(
                key=("full", "identity"), digest="connection-identity-only"
            )

    @pytest.mark.asyncio
    async def test_digest_collision_fails_closed_on_full_in_memory_key(self):
        pool = ChromeExecutionSessionPool(
            AsyncMock(
                return_value=ChromeSandboxHandle(sandbox=_sandbox(), delete=AsyncMock())
            )
        )
        digest = _digest("collision")
        first = ChromeExecutionScope(key=("one",), digest=digest)
        second = ChromeExecutionScope(key=("two",), digest=digest)
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        await pool.get_or_create(first, launch)

        with pytest.raises(ChromeSessionContractError, match="digest collision"):
            await pool.get_or_create(second, launch)

    @pytest.mark.asyncio
    async def test_daemon_status_mismatch_deletes_sandbox_without_retry(self):
        delete = AsyncMock()
        pool = ChromeExecutionSessionPool(
            AsyncMock(
                return_value=ChromeSandboxHandle(sandbox=_sandbox(), delete=delete)
            )
        )
        scope = _scope("status-mismatch")
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        session = await pool.get_or_create(scope, launch)
        session._started = True
        session._client = AsyncMock()
        session._client.status.return_value = {
            "running": True,
            "status": {**_healthy_status(), "args": ["--headless"]},
        }

        with pytest.raises(ChromeSessionContractError, match="status does not match"):
            await pool.invoke_tool(scope, launch, "take_snapshot", {})

        session._client.invoke_tool.assert_not_awaited()
        delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_transport_failure_preserves_session_for_next_health_check(self):
        delete = AsyncMock()
        pool = ChromeExecutionSessionPool(
            AsyncMock(
                return_value=ChromeSandboxHandle(sandbox=_sandbox(), delete=delete)
            )
        )
        scope = _scope("transport-failure")
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        session = await pool.get_or_create(scope, launch)
        session.invoke_tool = AsyncMock(
            side_effect=ChromeSessionTransportError("temporary timeout")
        )

        with pytest.raises(ChromeSessionTransportError, match="temporary timeout"):
            await pool.invoke_tool(scope, launch, "take_snapshot", {})

        assert pool._sessions[scope.digest][1] is session
        delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancellation_removes_session_and_deletes_sandbox(self):
        delete = AsyncMock()
        pool = ChromeExecutionSessionPool(
            AsyncMock(
                return_value=ChromeSandboxHandle(sandbox=_sandbox(), delete=delete)
            )
        )
        scope = _scope("cancelled-call")
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        session = await pool.get_or_create(scope, launch)
        session.invoke_tool = AsyncMock(side_effect=asyncio.CancelledError)

        with pytest.raises(asyncio.CancelledError):
            await pool.invoke_tool(scope, launch, "take_snapshot", {})

        assert scope.digest not in pool._sessions
        delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raw_session_failure_removes_cached_session(self):
        delete = AsyncMock()
        pool = ChromeExecutionSessionPool(
            AsyncMock(
                return_value=ChromeSandboxHandle(sandbox=_sandbox(), delete=delete)
            )
        )
        scope = _scope("raw-failure")
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        session = await pool.get_or_create(scope, launch)
        session.invoke_tool = AsyncMock(side_effect=RuntimeError("unexpected"))

        with pytest.raises(RuntimeError, match="unexpected"):
            await pool.invoke_tool(scope, launch, "take_snapshot", {})

        assert scope.digest not in pool._sessions
        delete.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "primary_failure",
        [
            asyncio.CancelledError("cancelled"),
            ChromeSessionContractError("contract failure"),
            RuntimeError("tool failure"),
        ],
    )
    async def test_cleanup_failure_never_replaces_primary_failure(
        self, primary_failure, caplog
    ):
        pool = ChromeExecutionSessionPool(AsyncMock())
        scope = _scope(type(primary_failure).__name__)
        launch = ChromeDaemonLaunchSpec.from_connection(_connection())
        session = AsyncMock()
        session.invoke_tool.side_effect = primary_failure
        pool.get_or_create = AsyncMock(return_value=session)
        cleanup_failure = RuntimeError("cleanup failure")
        pool.close_shielded = AsyncMock(side_effect=cleanup_failure)

        with pytest.raises(type(primary_failure)) as caught:
            await pool.invoke_tool(scope, launch, "take_snapshot", {})

        assert caught.value is primary_failure
        pool.close_shielded.assert_awaited_once_with(scope)
        assert any(
            record.message
            == "Chrome cleanup failed while preserving the primary failure"
            and record.exc_info is not None
            and record.exc_info[1] is cleanup_failure
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_cancelled_shielded_cleanup_logs_late_failure(self, caplog):
        pool = ChromeExecutionSessionPool(AsyncMock())
        entered = asyncio.Event()
        release = asyncio.Event()

        async def fail_late(_scope):
            entered.set()
            await release.wait()
            raise RuntimeError("delete failed")

        pool.close = fail_late
        waiter = asyncio.create_task(pool.close_shielded(_scope("late-failure")))
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert any(
            record.message == "Shielded Chrome cleanup failed"
            and record.exc_info is not None
            for record in caplog.records
        )
