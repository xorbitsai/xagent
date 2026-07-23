"""Managed SSH MCP tools (in-process). ``execute`` runs for real via the
SshExecutor; ``upload``/``download`` are still stubbed (SFTP arrives in a later
part). ``list_targets`` works via the injected SshTargetProvider. Secrets never
touch env/argv/tool-serialization."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from typing import Any, cast

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
from .base import AbstractBaseTool, ToolCategory
from .factory import register_tool

logger = logging.getLogger(__name__)

_NOT_ENABLED = "SSH execution is not enabled yet (arrives in Phase 3)."

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
    category: ToolCategory = ToolCategory.SSH

    def __init__(self, *, provider: SshTargetProvider, context: SshExecutionContext) -> None:
        self._provider = provider
        self._context = context

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("SSH tools are async-only.")

    def return_type(self) -> type[BaseModel]:
        return SshOpResult


class SshListTargetsTool(_SshToolBase):
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


class _SshOpTool(_SshToolBase):
    _capability: str = ""

    async def _resolve_or_error(self, target: str) -> tuple[Any | None, dict[str, Any] | None]:
        try:
            resolved = await self._provider.resolve(self._context, target)
        except SshError as exc:
            return None, SshOpResult(
                ok=False, error_code=exc.code.value, message=str(exc)
            ).model_dump()
        if self._capability not in resolved.capabilities:
            return None, SshOpResult(
                ok=False,
                error_code=SshErrorCode.OPERATION_NOT_ALLOWED.value,
                message=f"binding does not allow {self._capability}",
            ).model_dump()
        return resolved, None


class SshExecuteTool(AbstractBaseTool):
    """Runs a command on a bound target via the SshExecutor (resolve → egress →
    decrypt → materialize → run → cap → cleanup)."""

    category: ToolCategory = ToolCategory.SSH

    def __init__(self, *, executor: SshExecutor, context: SshExecutionContext) -> None:
        self._executor = executor
        self._context = context

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        raise NotImplementedError("SSH tools are async-only.")

    def return_type(self) -> type[BaseModel]:
        return SshOpResult

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
            return SshOpResult(ok=False, error_code=exc.code.value, message=str(exc)).model_dump()
        return SshOpResult(
            ok=True,
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            truncated=outcome.truncated,
            duration_ms=outcome.duration_ms,
        ).model_dump()


class SshUploadTool(_SshOpTool):
    _capability = "upload"

    @property
    def name(self) -> str:
        return "ssh_upload"

    @property
    def description(self) -> str:
        return "Upload a workspace file to a bound SSH target via SFTP."

    def args_type(self) -> type[BaseModel]:
        return TransferArgs

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        _resolved, err = await self._resolve_or_error(str(args.get("target", "")))
        if err is not None:
            return err
        return SshOpResult(ok=False, error_code=None, message=_NOT_ENABLED).model_dump()


class SshDownloadTool(_SshOpTool):
    _capability = "download"

    @property
    def name(self) -> str:
        return "ssh_download"

    @property
    def description(self) -> str:
        return "Download a file from a bound SSH target into the task workspace."

    def args_type(self) -> type[BaseModel]:
        return TransferArgs

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        _resolved, err = await self._resolve_or_error(str(args.get("target", "")))
        if err is not None:
            return err
        return SshOpResult(ok=False, error_code=None, message=_NOT_ENABLED).model_dump()


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


@register_tool(categories={"ssh"})
async def create_ssh_tools(config: Any) -> list[AbstractBaseTool]:
    """Emit SSH tools only when a provider is installed and the executing agent
    has at least one bound target."""
    from .....web.services.ssh_runtime import get_ssh_target_provider

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
    numeric_task_id = _numeric_task_id(task_id)
    agent_id = _agent_id_for_task(session_factory, numeric_task_id)
    if agent_id is None:
        logger.info("ssh tools: skip (unresolved agent_id for task_id=%r)", task_id)
        return []

    context = SshExecutionContext(
        actor=ActorRef(actor_type="user", actor_id=str(user_id)),
        execution_principal=PrincipalRef(principal_type="user", principal_id=str(user_id)),
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
    logger.info("ssh tools: emitting tools for agent %s (%d bound target(s))", agent_id, len(bound))

    # The SaaS provider is also the secret store (resolve + read_version on the
    # same adapter); the in-process runner materializes to a local private dir.
    executor = SshExecutor(
        provider=provider,
        secret_store=cast(SshSecretStore, provider),
        materializer=LocalTmpSecretMaterializer(),
        runner=AsyncsshRunner(),
        egress_config=_egress_from_env(),
    )
    return [
        SshListTargetsTool(provider=provider, context=context),
        SshExecuteTool(executor=executor, context=context),
        SshUploadTool(provider=provider, context=context),
        SshDownloadTool(provider=provider, context=context),
    ]
