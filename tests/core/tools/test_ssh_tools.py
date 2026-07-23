from types import SimpleNamespace

import pytest

from xagent.core.ssh import (
    ActorRef,
    BoundTargetInfo,
    PrincipalRef,
    ResolvedSshTarget,
    SshError,
    SshErrorCode,
    SshExecutionContext,
    SshSecretHandle,
)
from xagent.core.ssh.executor import SshExecuteOutcome
from xagent.core.tools.adapters.vibe.ssh_tools import (
    SshDownloadTool,
    SshExecuteTool,
    SshListTargetsTool,
    SshUploadTool,
    _agent_id_from_task,
    _egress_from_env,
    _numeric_task_id,
)


def _ctx() -> SshExecutionContext:
    return SshExecutionContext(
        actor=ActorRef(actor_type="user", actor_id="u"),
        execution_principal=PrincipalRef(principal_type="user", principal_id="u"),
        agent_id=1,
        task_id=None,
        turn_id=None,
        request_id="r",
    )


class _Provider:
    def __init__(self, resolved=None, targets=None, error=None):
        self._resolved = resolved
        self._targets = targets or []
        self._error = error

    async def resolve(self, context, target_alias):
        if self._error is not None:
            raise self._error
        return self._resolved

    async def read_version(self, secret_handle):
        raise NotImplementedError

    async def list_bound_targets(self, context):
        return self._targets


def _resolved(caps=("execute",)) -> ResolvedSshTarget:
    return ResolvedSshTarget(
        target_public_id="t",
        hostname="h",
        port=22,
        username="d",
        remote_root=None,
        capabilities=frozenset(caps),
        approval_policy="always",
        secret_handle=SshSecretHandle(credential_id="c", version_id="v"),
        known_hosts="h ssh-ed25519 AAAA\n",
        credential_public_id="c",
        credential_version_id="v",
        host_key_fingerprint="SHA256:x",
    )


async def test_list_targets_returns_aliases() -> None:
    provider = _Provider(
        targets=[
            BoundTargetInfo(alias="prod", display_name="Prod", capabilities=frozenset({"execute"})),
        ]
    )
    tool = SshListTargetsTool(provider=provider, context=_ctx())
    out = await tool.run_json_async({})
    assert out["targets"][0]["alias"] == "prod"
    assert "execute" in out["targets"][0]["capabilities"]


class _Executor:
    def __init__(self, outcome=None, error=None):
        self._outcome = outcome
        self._error = error

    async def execute(self, context, *, target_alias, command, timeout_seconds):
        if self._error is not None:
            raise self._error
        return self._outcome


async def test_execute_returns_outcome() -> None:
    outcome = SshExecuteOutcome(exit_code=0, stdout="ok", stderr="", truncated=False, duration_ms=5)
    tool = SshExecuteTool(executor=_Executor(outcome=outcome), context=_ctx())
    out = await tool.run_json_async({"target": "prod", "command": "uptime"})
    assert out["ok"] is True
    assert out["exit_code"] == 0
    assert out["stdout"] == "ok"
    assert out["truncated"] is False


async def test_execute_surfaces_ssh_error() -> None:
    tool = SshExecuteTool(
        executor=_Executor(error=SshError(SshErrorCode.TARGET_DISABLED, "disabled")),
        context=_ctx(),
    )
    out = await tool.run_json_async({"target": "prod", "command": "uptime"})
    assert out["ok"] is False
    assert out["error_code"] == "ssh_target_disabled"


def test_execute_sync_not_supported() -> None:
    tool = SshExecuteTool(executor=_Executor(), context=_ctx())
    with pytest.raises(NotImplementedError):
        tool.run_json_sync({"target": "prod", "command": "x"})


class _RecordingTransferExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def upload(self, context, *, target_alias, local_path, remote_path, overwrite):
        self.calls.append(("upload", target_alias, local_path, remote_path, overwrite))

    async def download(self, context, *, target_alias, remote_path, local_path, overwrite):
        self.calls.append(("download", target_alias, remote_path, local_path, overwrite))


class _FakeWorkspace:
    """Minimal workspace: resolves under a root and rejects escapes / missing
    files, mirroring TaskWorkspace's containment contract for these tools."""

    def __init__(self, root):
        self.root = root

    def _contained(self, p: str):
        from pathlib import Path

        resolved = (self.root / p).resolve()
        if not str(resolved).startswith(str(self.root.resolve())):
            raise ValueError("path escapes workspace")
        return resolved

    def resolve_path_with_search(self, p: str):
        resolved = self._contained(p)
        if not resolved.exists():
            raise FileNotFoundError(p)
        return resolved

    def resolve_path(self, p: str, default_dir: str = "output"):
        return self._contained(p)


async def test_upload_tool_passes_resolved_path_to_executor(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("x")
    ex = _RecordingTransferExecutor()
    tool = SshUploadTool(executor=ex, workspace=_FakeWorkspace(tmp_path), context=_ctx())
    out = await tool.run_json_async(
        {"target": "prod", "local_path": "f.txt", "remote_path": "/srv/f.txt"}
    )
    assert out["ok"] is True
    assert ex.calls[0][0] == "upload"
    assert ex.calls[0][3] == "/srv/f.txt"


async def test_upload_tool_rejects_workspace_escape(tmp_path) -> None:
    ex = _RecordingTransferExecutor()
    tool = SshUploadTool(executor=ex, workspace=_FakeWorkspace(tmp_path), context=_ctx())
    out = await tool.run_json_async(
        {"target": "prod", "local_path": "../../etc/passwd", "remote_path": "/srv/x"}
    )
    assert out["ok"] is False
    assert out["error_code"] == "ssh_operation_not_allowed"
    assert ex.calls == []  # executor never reached — no connection, no secret


async def test_download_tool_writes_into_workspace(tmp_path) -> None:
    ex = _RecordingTransferExecutor()
    tool = SshDownloadTool(executor=ex, workspace=_FakeWorkspace(tmp_path), context=_ctx())
    out = await tool.run_json_async(
        {"target": "prod", "remote_path": "/srv/f", "local_path": "out.txt"}
    )
    assert out["ok"] is True
    assert ex.calls[0][0] == "download"


async def test_transfer_tool_without_workspace_fails_closed() -> None:
    ex = _RecordingTransferExecutor()
    tool = SshUploadTool(executor=ex, workspace=None, context=_ctx())
    out = await tool.run_json_async({"target": "prod", "local_path": "f", "remote_path": "/srv/f"})
    assert out["ok"] is False
    assert ex.calls == []


def test_egress_from_env_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_SSH_ALLOW_CIDRS", "127.0.0.0/8, 10.0.0.0/8")
    assert _egress_from_env().allow_cidrs == ("127.0.0.0/8", "10.0.0.0/8")


def test_egress_from_env_default_empty(monkeypatch) -> None:
    monkeypatch.delenv("XAGENT_SSH_ALLOW_CIDRS", raising=False)
    assert _egress_from_env().allow_cidrs == ()


def test_numeric_task_id_parses_workspace_prefixed_id() -> None:
    # config.get_task_id() hands the tool a workspace-scoped string, not the
    # bare DB id — e.g. "web_task_30". Non-task ids ("tools_list") yield None.
    assert _numeric_task_id("web_task_30") == 30
    assert _numeric_task_id(30) == 30
    assert _numeric_task_id("30") == 30
    assert _numeric_task_id("tools_list") is None
    assert _numeric_task_id(None) is None


def test_agent_id_from_task_normal() -> None:
    task = SimpleNamespace(agent_id=5, agent_config=None)
    assert _agent_id_from_task(task) == 5


def test_agent_id_from_task_preview_fallback() -> None:
    # Preview tasks (#459) carry agent_id=None; the edited agent id lives in
    # agent_config["preview_agent_id"].
    task = SimpleNamespace(agent_id=None, agent_config={"preview_agent_id": 7})
    assert _agent_id_from_task(task) == 7


def test_agent_id_from_task_none_when_unresolvable() -> None:
    assert _agent_id_from_task(None) is None
    assert _agent_id_from_task(SimpleNamespace(agent_id=None, agent_config=None)) is None


def test_ssh_creator_registered_under_ssh_category() -> None:
    # Managed SSH tools get their own assignable "ssh" category so the agent
    # editor can auto-enable it when a target is bound (mirrors connectors).
    from xagent.core.tools.adapters.vibe.factory import ToolRegistry

    ToolRegistry._import_tool_modules()
    entry = next(e for e in ToolRegistry._tool_creators if e[0].__name__ == "create_ssh_tools")
    assert entry[1] == frozenset({"ssh"})


def test_ssh_tools_carry_ssh_category() -> None:
    # Category must be SSH (not OTHER) so compute_allowed_names admits them
    # when the "ssh" category is selected.
    assert SshExecuteTool(executor=_Executor(), context=_ctx()).metadata.category.value == "ssh"
    assert SshListTargetsTool(provider=_Provider(), context=_ctx()).metadata.category.value == "ssh"
