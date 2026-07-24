"""Managed SSH MCP tools (in-process). ``execute`` runs commands and
``upload``/``download`` transfer files (SFTP) for real via the SshExecutor;
``list_targets`` works via the injected SshTargetProvider. Local transfer paths
are containment-checked against the task workspace. Secrets never touch
env/argv/tool-serialization."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from pydantic import BaseModel, Field

from ....ssh import (
    ActorRef,
    PrincipalRef,
    SshError,
    SshErrorCode,
    SshExecutionContext,
    SshSecretStore,
    SshTargetProvider,
)
from ....ssh.egress import EgressPolicyConfig
from ....ssh.executor import SshExecutor
from ....ssh.materializer import LocalTmpSecretMaterializer
from ....ssh.runner import AsyncsshRunner
from ....ssh.sandbox_materializer import SandboxTmpfsSecretMaterializer
from ....ssh.sandbox_runner import SandboxSshRunner
from .base import AbstractBaseTool, ToolCategory
from .factory import ToolFactory, register_tool

logger = logging.getLogger(__name__)

_ALLOW_CIDRS_ENV = "XAGENT_SSH_ALLOW_CIDRS"


def _egress_from_env() -> EgressPolicyConfig:
    """Egress policy for this deployment. Denies loopback/link-local/private/
    metadata by default; ``XAGENT_SSH_ALLOW_CIDRS`` (comma-separated) allowlists
    networks — e.g. set it to ``127.0.0.0/8`` to test against a local sshd.
    Richer deployment injection (VPC connectors) arrives in a later part."""
    raw = os.getenv(_ALLOW_CIDRS_ENV, "")
    cidrs = tuple(c.strip() for c in raw.split(",") if c.strip())
    return EgressPolicyConfig(allow_cidrs=cidrs)


class _EmptyArgs(BaseModel):
    pass


class ListTargetsResult(BaseModel):
    targets: list[dict[str, Any]] = Field(default_factory=list)


class ExecuteArgs(BaseModel):
    target: str = Field(description="A target alias bound to this agent.")
    command: str = Field(description="Non-interactive remote command to run.")
    timeout_seconds: int = Field(default=60, description="Max seconds to wait.")


class TransferArgs(BaseModel):
    target: str = Field(description="A target alias bound to this agent.")
    local_path: str = Field(description="Path within the task workspace.")
    remote_path: str = Field(description="Absolute remote path.")
    overwrite: bool = Field(default=False)


class SshOpResult(BaseModel):
    ok: bool
    error_code: str | None = None
    message: str = ""
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    truncated: bool | None = None
    duration_ms: int | None = None


class _SshToolBase(AbstractBaseTool):
    """Shared boilerplate for every SSH tool: the SSH category, async-only
    execution, and the SshOpResult return type. Subclasses add their own
    dependencies on top of the common ``context``."""

    category: ToolCategory = ToolCategory.SSH

    def __init__(self, *, context: SshExecutionContext) -> None:
        self._context = context

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("SSH tools are async-only.")

    def return_type(self) -> type[BaseModel]:
        return SshOpResult


class SshListTargetsTool(_SshToolBase):
    def __init__(
        self, *, provider: SshTargetProvider, context: SshExecutionContext
    ) -> None:
        super().__init__(context=context)
        self._provider = provider

    @property
    def name(self) -> str:
        return "ssh_list_targets"

    @property
    def description(self) -> str:
        return "List the SSH targets this agent may use (alias + allowed operations)."

    def args_type(self) -> type[BaseModel]:
        return _EmptyArgs

    def return_type(self) -> type[BaseModel]:
        return ListTargetsResult

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        infos = await self._provider.list_bound_targets(self._context)
        return ListTargetsResult(
            targets=[
                {
                    "alias": i.alias,
                    "display_name": i.display_name,
                    "capabilities": sorted(i.capabilities),
                }
                for i in infos
            ]
        ).model_dump()


class _SshTransferTool(_SshToolBase):
    """Base for the SFTP tools: executor-backed and workspace-aware. The
    workspace resolves and containment-checks the local path (no escaping the
    task workspace) before any transfer; the executor enforces capability,
    egress, and the remote_root constraint."""

    def __init__(
        self, *, executor: SshExecutor, workspace: Any, context: SshExecutionContext
    ) -> None:
        super().__init__(context=context)
        self._executor = executor
        self._workspace = workspace

    def args_type(self) -> type[BaseModel]:
        return TransferArgs

    @staticmethod
    def _fail(code: str | None, message: str) -> Any:
        return SshOpResult(ok=False, error_code=code, message=message).model_dump()


class SshExecuteTool(_SshToolBase):
    """Runs a command on a bound target via the SshExecutor (resolve → egress →
    decrypt → materialize → run → cap → cleanup)."""

    def __init__(self, *, executor: SshExecutor, context: SshExecutionContext) -> None:
        super().__init__(context=context)
        self._executor = executor

    @property
    def name(self) -> str:
        return "ssh_execute"

    @property
    def description(self) -> str:
        return "Run a non-interactive command on a bound SSH target."

    def args_type(self) -> type[BaseModel]:
        return ExecuteArgs

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        try:
            outcome = await self._executor.execute(
                self._context,
                target_alias=str(args.get("target", "")),
                command=str(args.get("command", "")),
                timeout_seconds=int(args.get("timeout_seconds", 60)),
            )
        except SshError as exc:
            return SshOpResult(
                ok=False, error_code=exc.code.value, message=str(exc)
            ).model_dump()
        return SshOpResult(
            ok=True,
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            truncated=outcome.truncated,
            duration_ms=outcome.duration_ms,
        ).model_dump()


class SshUploadTool(_SshTransferTool):
    @property
    def name(self) -> str:
        return "ssh_upload"

    @property
    def description(self) -> str:
        return "Upload a workspace file to a bound SSH target via SFTP."

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        if self._workspace is None:
            return self._fail(None, "no task workspace available for file transfer")
        # The source must exist inside the task workspace; resolve_path_with_search
        # raises on a missing file or a path that escapes the workspace.
        try:
            local = self._workspace.resolve_path_with_search(
                str(args.get("local_path", ""))
            )
        except (ValueError, FileNotFoundError) as exc:
            return self._fail(
                SshErrorCode.OPERATION_NOT_ALLOWED.value, f"local path rejected: {exc}"
            )
        try:
            await self._executor.upload(
                self._context,
                target_alias=str(args.get("target", "")),
                local_path=str(local),
                remote_path=str(args.get("remote_path", "")),
                overwrite=bool(args.get("overwrite", False)),
            )
        except SshError as exc:
            return self._fail(exc.code.value, str(exc))
        return SshOpResult(ok=True, message="uploaded").model_dump()


class SshDownloadTool(_SshTransferTool):
    @property
    def name(self) -> str:
        return "ssh_download"

    @property
    def description(self) -> str:
        return "Download a file from a bound SSH target into the task workspace."

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        if self._workspace is None:
            return self._fail(None, "no task workspace available for file transfer")
        # The destination need not exist yet, but must resolve within the
        # workspace (defaults under output/); resolve_path raises on escape.
        try:
            local = self._workspace.resolve_path(
                str(args.get("local_path", "")), default_dir="output"
            )
        except ValueError as exc:
            return self._fail(
                SshErrorCode.OPERATION_NOT_ALLOWED.value, f"local path rejected: {exc}"
            )
        try:
            await self._executor.download(
                self._context,
                target_alias=str(args.get("target", "")),
                remote_path=str(args.get("remote_path", "")),
                local_path=str(local),
                overwrite=bool(args.get("overwrite", False)),
            )
        except SshError as exc:
            return self._fail(exc.code.value, str(exc))
        return SshOpResult(ok=True, message="downloaded").model_dump()


def _agent_id_from_task(task: Any) -> int | None:
    """Resolve the agent id a task runs as. Normal tasks carry ``agent_id``;
    build-preview tasks (#459) leave it NULL and carry the edited agent id in
    ``agent_config["preview_agent_id"]``."""
    if task is None:
        return None
    if task.agent_id is not None:
        return int(task.agent_id)
    cfg = task.agent_config or {}
    pid = cfg.get("preview_agent_id") if isinstance(cfg, dict) else None
    return int(pid) if pid is not None else None


def _numeric_task_id(task_id: Any) -> int | None:
    """Extract the DB task id. The tool config hands us a workspace-scoped
    string like ``"web_task_30"`` (or a non-task id like ``"tools_list"``),
    not the bare integer primary key."""
    if task_id is None:
        return None
    # Assumption: the DB id is the trailing integer of the workspace-scoped id
    # (``web_task_30`` → 30); ids with no trailing digits (``tools_list``) are
    # intentionally treated as "no task". If the id format ever grows an
    # internal number this trailing-match would need revisiting.
    match = re.search(r"(\d+)$", str(task_id))
    return int(match.group(1)) if match else None


def _agent_id_for_task(session_factory: Any, numeric_task_id: int | None) -> int | None:
    if numeric_task_id is None:
        return None
    from .....web.models.task import Task
    from .db_session import tool_session_scope

    with tool_session_scope(session_factory) as db:
        task = db.query(Task).filter(Task.id == numeric_task_id).first()
        return _agent_id_from_task(task)


def _make_ssh_sandbox_lease(
    task_id: int | None, agent_id: int
) -> Callable[[], AbstractAsyncContextManager[object]] | None:
    """Build a lazy lease over a dedicated per-task ``ssh::<id>`` sandbox, or
    None when no sandbox subsystem is running (self-hosted → in-process runner).

    The sandbox is distinct from the agent's own code sandbox, so key material
    and the ssh binary never live where the agent can reach them. Leasing is
    lazy (only a task that actually runs SSH pays for it) and fail-closed:
    capacity exhaustion raises SANDBOX_UNAVAILABLE rather than falling back to
    the backend host (design §15.2)."""
    from .....web.sandbox_manager import get_sandbox_manager

    manager = get_sandbox_manager()
    if manager is None:
        return None
    # Tie the sandbox lifecycle to the task; fall back to the agent when a task
    # id is unavailable (e.g. preview) so it stays sandboxed, never host-bound.
    lifecycle_id = str(task_id) if task_id is not None else f"agent-{agent_id}"

    @asynccontextmanager
    async def _lease() -> AsyncIterator[object]:
        from .....web.sandbox_manager import SandboxCapacityError

        try:
            provider = await manager.get_or_create_lease_provider("ssh", lifecycle_id)
        except SandboxCapacityError as exc:
            raise SshError(
                SshErrorCode.SANDBOX_UNAVAILABLE,
                "no sandbox capacity available for ssh",
            ) from exc
        async with provider.lease(concurrency_safe=False) as sandbox:
            yield sandbox

    return lambda: _lease()


@register_tool(categories={"ssh"})
async def create_ssh_tools(config: Any) -> list[AbstractBaseTool]:
    """Emit SSH tools only when a provider is installed and the executing agent
    has at least one bound target."""
    from .....web.services.ssh_runtime import (
        get_ssh_audit_sink,
        get_ssh_target_provider,
    )

    try:
        session_factory = config.get_session_factory()
        user_id = config.get_user_id()
        task_id = config.get_task_id()
    except Exception:  # noqa: BLE001
        logger.info("ssh tools: config accessors unavailable; skipping")
        return []
    if not user_id or session_factory is None:
        logger.info(
            "ssh tools: skip (user_id=%r, has_session_factory=%s)",
            user_id,
            session_factory is not None,
        )
        return []

    # Hand the provider the factory, not a live session: it opens its own
    # one-shot session per resolve/list call (session would otherwise be closed
    # by the time the tools run).
    provider = get_ssh_target_provider(session_factory)
    if provider is None:
        logger.info("ssh tools: skip (no provider hook installed)")
        return []
    # The provider adapter must also be the secret store (resolve + read_version
    # on one object). Verify structurally instead of a bare cast so a mis-wired
    # provider skips cleanly rather than failing mid-execute on a missing method.
    secret_store = provider
    if not isinstance(secret_store, SshSecretStore):
        logger.error("ssh tools: provider does not implement SshSecretStore; skipping")
        return []
    numeric_task_id = _numeric_task_id(task_id)
    agent_id = _agent_id_for_task(session_factory, numeric_task_id)
    if agent_id is None:
        logger.info("ssh tools: skip (unresolved agent_id for task_id=%r)", task_id)
        return []

    context = SshExecutionContext(
        actor=ActorRef(actor_type="user", actor_id=str(user_id)),
        execution_principal=PrincipalRef(
            principal_type="user", principal_id=str(user_id)
        ),
        agent_id=agent_id,
        task_id=numeric_task_id,
        turn_id=None,
        request_id=str(task_id or ""),
    )
    try:
        bound = await provider.list_bound_targets(context)
    except SshError as exc:
        logger.info("ssh tools: skip (list_bound_targets raised %s)", exc.code.value)
        return []
    except Exception:  # noqa: BLE001
        logger.exception("ssh tools: list_bound_targets failed for agent %s", agent_id)
        return []
    if not bound:
        logger.info("ssh tools: skip (agent %s has no bound targets)", agent_id)
        return []
    logger.info(
        "ssh tools: emitting tools for agent %s (%d bound target(s))",
        agent_id,
        len(bound),
    )

    # The provider is also the secret store (resolve + read_version on the
    # same adapter). When a sandbox subsystem is available (xagent-cloud), SSH
    # runs inside a dedicated per-task sandbox — isolated from the agent's own
    # code sandbox, so the agent can neither read the key nor ssh directly
    # (design §15.2). Without one (self-hosted), the in-process runner
    # materializes to a local private dir and connects from the backend.
    sandbox_lease = _make_ssh_sandbox_lease(numeric_task_id, agent_id)
    if sandbox_lease is not None:
        # SSH runs inside the leased sandbox, but the Boxlite backend buffers a
        # command's whole output before trimming (host memory unbounded — see
        # boxlite_sandbox.exec), so ssh_execute on unbounded remote output is a
        # host-memory DoS there. The Docker backend and the in-process runner
        # both stream-cap incrementally; gate SSH tools off Boxlite until it
        # does too (M2).
        if os.getenv("SANDBOX_IMPLEMENTATION", "docker") == "boxlite":
            logger.info("ssh tools: skip (boxlite sandbox backend not supported)")
            return []
        materializer: Any = SandboxTmpfsSecretMaterializer()
        runner: Any = SandboxSshRunner()
    else:
        materializer = LocalTmpSecretMaterializer()
        runner = AsyncsshRunner()
    executor = SshExecutor(
        provider=provider,
        secret_store=secret_store,
        materializer=materializer,
        runner=runner,
        egress_config=_egress_from_env(),
        sandbox_lease=sandbox_lease,
        audit_sink=get_ssh_audit_sink(session_factory),
    )
    # SFTP tools resolve/containment-check local paths against the task
    # workspace; None (e.g. tool-listing) disables transfers, not execute.
    workspace = ToolFactory._create_workspace(config.get_workspace_config())
    return [
        SshListTargetsTool(provider=provider, context=context),
        SshExecuteTool(executor=executor, context=context),
        SshUploadTool(executor=executor, workspace=workspace, context=context),
        SshDownloadTool(executor=executor, workspace=workspace, context=context),
    ]
