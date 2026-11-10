"""
Generic sandboxed tool wrapper

Execute tool's run_json_sync/async methods in sandbox environment.
"""

import asyncio
import copy
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Optional,
    Protocol,
    Type,
    TypeGuard,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel

from ......config import SANDBOX_TOOL_RUNNER
from ......sandbox.base import Sandbox
from ......sandbox.tool_runtime.bundle import build_runtime_command, sandbox_connection
from .....workspace import TaskWorkspace
from ....artifacts import build_generated_file_metadata, build_inline_artifact
from ..base import AbstractBaseTool, ToolMetadata
from ..function import FunctionTool
from .sandbox_config import (
    extract_bound_method_target,
    resolve_sandbox_config,
)

logger = logging.getLogger(__name__)

# `timeout -s KILL` SIGKILLs its whole process group at the deadline, itself
# included. Docker reports 128+9, while subprocess-compatible adapters preserve
# the negative signal return code. Both representations can occur without BoxLite.
_SIGKILL_EXIT_CODES = (137, -9)

SANDBOX_BASE_DEPENDENCIES = [
    "pydantic>=2.0.0",
    "pydantic-settings>=2.16.0",
    "cloudpickle>=3.0.0",
]


async def _write_sandbox_file_cancellation_safe(
    sandbox: Sandbox,
    content: str,
    remote_path: str,
    *,
    overwrite: bool = False,
) -> None:
    """Finish an in-flight upload before propagating caller cancellation.

    Docker file writes use ``asyncio.to_thread`` internally. Cancelling the
    await does not stop that worker, so cleanup could otherwise delete the
    destination before the worker recreates it. Shield and drain the owned
    upload task first; its own failure still propagates normally when the
    caller was not cancelled.
    """
    upload = asyncio.create_task(
        sandbox.write_file(content, remote_path, overwrite=overwrite)
    )
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(upload)
        except asyncio.CancelledError as error:
            current = asyncio.current_task()
            caller_is_cancelling = current is not None and current.cancelling() > 0
            if not caller_is_cancelling:
                raise
            if cancellation is None:
                cancellation = error
            if not upload.done():
                continue
            try:
                upload.result()
            except BaseException as upload_error:
                raise cancellation from upload_error
        except BaseException as upload_error:
            if cancellation is not None:
                raise cancellation from upload_error
            raise
        if cancellation is not None:
            raise cancellation
        return


class _StaticSandboxLease:
    """Async context manager that exposes one fixed sandbox."""

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    async def __aenter__(self) -> Sandbox:
        return self._sandbox

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        return None


@runtime_checkable
class _SandboxLeaseProviderLike(Protocol):
    """Structural shape of a SandboxLeaseProvider (src/xagent/web/sandbox_manager.py).

    A Protocol rather than an import of that class: this module lives under
    core/ and must not depend on web/. Declaring the shape here lets
    resolve_primary_sandbox's signature express what it actually accepts
    instead of ``Any``.
    """

    primary_sandbox: Sandbox

    def lease(self, *, concurrency_safe: bool) -> Any: ...


def _is_sandbox_lease_provider(
    value: Any,
) -> TypeGuard[_SandboxLeaseProviderLike]:
    """Return whether an object is a real sandbox lease provider.

    Checks only for a callable ``lease`` -- the one thing _lease_sandbox()
    actually needs to decide whether to call `.lease(...)` or wrap a plain
    Sandbox. This makes the TypeGuard not fully sound (a `.lease`-only
    object narrows to _SandboxLeaseProviderLike here even without
    `.primary_sandbox`): tightening it to also require `.primary_sandbox`
    was tried and reverted -- it broke
    tests/core/tools/adapters/sandboxed_tool/test_sandboxed_tool_lease_provider.py's
    FakeLeaseProvider, a legitimate lease-provider double that only
    implements `.lease()` and names its own bookkeeping attribute
    ``primary``, not ``primary_sandbox``. resolve_primary_sandbox is the
    only caller that needs `.primary_sandbox` to actually exist, and in
    production it's only ever given the real SandboxLeaseProvider (which
    always has it) or a plain Sandbox (which this check correctly rejects
    -- no callable `.lease` on the class).
    """
    return callable(getattr(type(value), "lease", None))


def _has_primary_sandbox(value: Any) -> TypeGuard[_SandboxLeaseProviderLike]:
    """Return whether an object exposes ``.primary_sandbox``.

    A separate, narrower check from _is_sandbox_lease_provider: this is
    the one attribute resolve_primary_sandbox actually dereferences, and
    checking for it directly (rather than reusing the lease-only check)
    keeps this sound for resolve_primary_sandbox's own contract without
    touching _is_sandbox_lease_provider's existing, battle-tested
    definition that _lease_sandbox()/FakeLeaseProvider-style test doubles
    depend on (see that function's docstring for why it can't require
    ``.primary_sandbox`` too).

    Sound for the two real production shapes (a concrete Sandbox subclass
    never has this attribute; SandboxLeaseProvider always does), but an
    unspecced ``MagicMock()`` auto-vivifies *any* attribute access,
    including this one -- a mock meant to stand in for a plain Sandbox
    must be built with ``spec=Sandbox`` or it will misclassify as a lease
    provider here.
    """
    return hasattr(value, "primary_sandbox")


def resolve_primary_sandbox(sandbox: "Sandbox | _SandboxLeaseProviderLike") -> Sandbox:
    """Unwrap a SandboxLeaseProvider to its primary Sandbox, if given one.

    For one-shot/setup-style operations (dependency install, MCP tool
    listing) that don't need per-call worker-slot leasing, callers should
    use this instead of re-deriving the same ``.primary_sandbox`` check --
    a SandboxLeaseProvider has no ``.exec``/``.read_file``/``.name`` of its
    own, so operating on it directly raises AttributeError.

    Checks ``isinstance(sandbox, Sandbox)`` first, before the ``.primary_sandbox``
    duck-type check: a concrete Sandbox is never misclassified this way
    regardless of what attributes it happens to expose, which matters
    because SandboxLeaseProvider is not a Sandbox subclass and vice versa
    -- the two shapes don't otherwise overlap in production, but nothing
    stops an untyped test double from claiming to be either.
    """
    if sandbox is None:
        raise ValueError("sandbox cannot be None")
    if isinstance(sandbox, Sandbox):
        return sandbox
    if _has_primary_sandbox(sandbox):
        return sandbox.primary_sandbox
    return cast(Sandbox, sandbox)


class SandboxDependencyManager:
    """Track incremental dependency installation per sandbox."""

    _sandbox_installed_requirements: dict[str, set[str]] = {}
    _sandbox_locks: dict[str, asyncio.Lock] = {}
    _locks_lock = asyncio.Lock()

    @classmethod
    def reset(cls) -> None:
        """Clear all tracked state. Intended for test teardown."""
        cls._sandbox_installed_requirements.clear()
        cls._sandbox_locks.clear()

    @classmethod
    async def ensure_requirements(
        cls,
        sandbox: Sandbox,
        requirements: list[str],
    ) -> None:
        """Install any missing requirements into the target sandbox."""
        sandbox_key = sandbox.name
        required = set(requirements)
        if not required:
            return

        installed = cls._sandbox_installed_requirements.get(sandbox_key, set())
        missing = sorted(required - installed)
        if not missing:
            return

        if sandbox_key not in cls._sandbox_locks:
            async with cls._locks_lock:
                if sandbox_key not in cls._sandbox_locks:
                    cls._sandbox_locks[sandbox_key] = asyncio.Lock()
        lock = cls._sandbox_locks[sandbox_key]

        async with lock:
            installed = cls._sandbox_installed_requirements.get(sandbox_key, set())
            missing = sorted(required - installed)
            if not missing:
                return

            requirements_txt = "\n".join(missing)
            try:
                await sandbox.write_file(
                    content=requirements_txt,
                    remote_path="/tmp/requirements.txt",
                    overwrite=True,
                )

                try:
                    result = await asyncio.wait_for(
                        sandbox.exec(
                            "pip",
                            "install",
                            "--user",
                            "--break-system-packages",
                            "-r",
                            "/tmp/requirements.txt",
                        ),
                        timeout=300,
                    )
                except asyncio.TimeoutError:
                    logger.error("pip install timed out after 300s")
                    raise RuntimeError(
                        "Dependency installation timed out after 300 seconds"
                    )

                if result.exit_code != 0:
                    logger.error(f"Failed to install dependencies: {result.stderr}")
                    raise RuntimeError(
                        f"Dependency installation failed: {result.stderr}"
                    )

                cls._sandbox_installed_requirements[sandbox_key] = installed | required
            except RuntimeError:
                raise
            except Exception as e:
                logger.error(f"Unexpected error installing dependencies: {e}")
                raise


class SandboxedToolWrapper(AbstractBaseTool):
    """
    Generic sandboxed tool wrapper

    Wrap any AbstractBaseTool as a sandboxed execution version.
    Execute supported tools with an execution-only runtime, without platform source mounts.
    """

    def __init__(
        self,
        target_tool: AbstractBaseTool,
        sandbox: Any,
    ):
        """
        Initialize sandboxed tool wrapper

        Args:
            target_tool: Target tool to wrap
            sandbox: Sandbox instance or lease provider
        """
        self._target = target_tool
        self._sandbox = sandbox

        sandbox_config = resolve_sandbox_config(target_tool)
        if sandbox_config is None or not sandbox_config.enabled:
            raise RuntimeError(
                f"Tool '{target_tool.name}' is not configured for sandbox runtime."
            )

        # base dependencies + tool dependencies
        self._requirements = SANDBOX_BASE_DEPENDENCIES + list(sandbox_config.packages)
        self._env_vars = list(sandbox_config.env_vars)
        self._exec_timeout = sandbox_config.exec_timeout

        # Proxy target tool attributes
        self._visibility = getattr(target_tool, "_visibility", None)
        self._allow_users = getattr(target_tool, "_allow_users", None)

        reconstruction_target = self._resolve_execution_target()
        self._reconstruction_target = reconstruction_target

        # MCP permission checks, schema validation, delegated credentials and
        # retries stay on the host. Only the transport call runs in the guest.
        from ..mcp_adapter import MCPToolAdapter

        self._mcp_target = (
            copy.copy(reconstruction_target)
            if isinstance(reconstruction_target, MCPToolAdapter)
            else None
        )
        # An adapter can be supplied to multiple task factories. Never bind a
        # task-specific sandbox transport on the shared original instance.
        if self._mcp_target is not None:
            self._mcp_target._sandbox_transport = self._execute_mcp_in_sandbox

    @property
    def is_sandboxed(self) -> bool:
        """Marker for sandboxed."""
        return True

    @property
    def name(self) -> str:
        return self._target.name

    @property
    def description(self) -> str:
        return self._target.description

    @property
    def tags(self) -> list[str]:
        return self._target.tags

    @property
    def metadata(self) -> ToolMetadata:
        return self._target.metadata

    def args_type(self) -> Type[BaseModel]:
        return self._target.args_type()

    def return_type(self) -> Type[BaseModel]:
        return self._target.return_type()

    def state_type(self) -> Optional[Type[BaseModel]]:
        return self._target.state_type()

    def return_value_as_string(self, value: Any) -> str:
        """Delegate result formatting to the wrapped tool."""
        return self._target.return_value_as_string(value)

    def _build_execution_env(self) -> dict[str, str]:
        """Build per-exec environment variables (scoped to this process, not the sandbox)."""
        env: dict[str, str] = {}

        for env_var in self._env_vars:
            value = os.getenv(env_var)
            if value is not None:
                env[env_var] = value
            else:
                logger.warning(f"Environment variable {env_var} not found in host")

        # Seeded last: a tool-declared env_var must not be able to override
        # these. A host PYTHONPATH would point the runner outside the sandbox.
        env["PYTHONPATH"] = ""
        env[SANDBOX_TOOL_RUNNER] = "1"
        return env

    def _lease_sandbox(self) -> Any:
        """Lease the sandbox that should execute this tool call."""
        if _is_sandbox_lease_provider(self._sandbox):
            return self._sandbox.lease(concurrency_safe=self.metadata.concurrency_safe)
        return _StaticSandboxLease(self._sandbox)

    async def _ensure_dependencies(self, sandbox: Sandbox | None = None) -> None:
        """Ensure dependencies are installed in the sandbox."""
        if sandbox is None:
            sandbox = resolve_primary_sandbox(self._sandbox)
        await SandboxDependencyManager.ensure_requirements(sandbox, self._requirements)

    def _resolve_execution_target(self) -> Any:
        """Unwrap supported bound tools without serializing application objects."""
        if isinstance(self._target, FunctionTool):
            bound = extract_bound_method_target(self._target)
            if bound is not None:
                return bound[0]
            raise RuntimeError(
                f"FunctionTool '{self._target.name}' uses an unsupported callable."
            )
        return self._target

    def _build_runtime_request(self, args: Mapping[str, Any]) -> dict[str, Any]:
        from ..command_executor import CommandExecutorTool
        from ..javascript_executor import JavaScriptExecutorTool
        from ..python_executor import PythonExecutorTool

        target = self._reconstruction_target
        if isinstance(target, CommandExecutorTool):
            kind = "command"
        elif isinstance(target, PythonExecutorTool):
            kind = "python"
        elif isinstance(target, JavaScriptExecutorTool):
            kind = "javascript"
        else:
            raise RuntimeError(
                "Tool is not supported by the isolated execution runtime. "
                "Backend execution and platform source mounts are not permitted."
            )
        parsed_args = target.args_type().model_validate(dict(args)).model_dump()
        request: dict[str, Any] = {
            "kind": kind,
            "arguments": parsed_args,
            "working_directory": target._get_working_directory(),
        }
        if isinstance(target, PythonExecutorTool):
            request["workspace_variables"] = target._get_workspace_env() or {}
        return request

    async def _execute_mcp_in_sandbox(
        self,
        connection: Mapping[str, Any],
        tool_name: str,
        arguments: Mapping[str, Any],
        meta: Mapping[str, Any],
    ) -> Any:
        return await self._execute_runtime_request(
            {
                "kind": "mcp_call",
                "connection": sandbox_connection(connection),
                "tool_name": tool_name,
                "arguments": dict(arguments),
                "meta": dict(meta),
            }
        )

    async def get_sandbox_for_test(self) -> Sandbox:
        """Get the sandbox for exec test"""
        async with self._lease_sandbox() as sandbox:
            await self._ensure_dependencies(sandbox)
            return cast(Sandbox, sandbox)

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Synchronous execution (calls async version via asyncio.run)"""
        return asyncio.run(self.run_json_async(args))

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute the tool in the sandbox, then register its files here."""
        result = await self._run_json_in_sandbox(args)
        from ..command_executor import CommandExecutorTool, _dump_result

        # Shell output metadata was already registered and validated on the
        # host. Do not strip its validation or stage its bytes a second time.
        if isinstance(self._reconstruction_target, CommandExecutorTool):
            return _dump_result(result)
        # Guest metadata is not evidence about persisted host bytes. Only the
        # host rebuild below can attach an authoritative validation report.
        pending = [result]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                value.pop("validation", None)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        try:
            result = await self._register_sandbox_outputs(result)
        except Exception:
            # A successful sandbox run must survive a registration failure.
            logger.warning(
                "Host-side registration failed for %s; returning sandbox metadata",
                self._target.name,
                exc_info=True,
            )
        return self._normalize_executor_result(result)

    def _normalize_executor_result(self, result: Any) -> Any:
        from ..javascript_executor import (
            JavaScriptExecutorResult,
            JavaScriptExecutorTool,
        )
        from ..python_executor import PythonExecutorResult, PythonExecutorTool

        if not isinstance(result, dict):
            return result
        target = self._reconstruction_target
        if not isinstance(target, (PythonExecutorTool, JavaScriptExecutorTool)):
            return result
        result.setdefault("output", "")
        refs = result.get("file_refs") or []
        result["artifacts"] = [build_inline_artifact(ref) for ref in refs]
        if isinstance(target, JavaScriptExecutorTool):
            if result.get("success") and result.get("generated_files"):
                output = (
                    result.get("output") or "Code executed successfully (no output)"
                )
                result["output"] = (
                    f"{output}\n\nGenerated files: {', '.join(result['generated_files'])}"
                )
            return JavaScriptExecutorResult(**result).model_dump()
        return PythonExecutorResult(**result).model_dump()

    @staticmethod
    def _resolved_ref_key(ref: Mapping[str, Any]) -> str | None:
        """Merge key for every ref, registrable or not.

        Deliberately not _host_output_key: that one decides what the host may
        persist, so it rejects symlinks and paths outside the output tree.
        Applying those checks here would drop guest-only refs from the merge.
        """
        raw = ref.get("file_path")
        if not raw:
            return None
        try:
            return str(Path(str(raw)).resolve())
        except OSError:
            return None

    @staticmethod
    def _host_output_key(workspace: Any, ref: Mapping[str, Any]) -> str | None:
        """Resolved key for a ref the host is willing to register.

        A guest-supplied path is not authorization to persist whatever it
        points at: sandboxed code can drop a symlink into the output tree.
        Only a regular, non-symlink file whose resolved path stays under this
        workspace's output directory is accepted.
        """
        raw = ref.get("file_path")
        if not raw:
            return None
        candidate = Path(str(raw))
        try:
            if candidate.is_symlink() or not candidate.is_file():
                return None
            resolved = candidate.resolve()
            output_root = Path(workspace.output_dir).resolve()
        except OSError:
            return None
        if not resolved.is_relative_to(output_root):
            logger.warning(
                "Refusing to register sandbox output %s outside %s",
                resolved,
                output_root,
            )
            return None
        return str(resolved)

    @staticmethod
    def _reregister_on_host(workspace: Any, paths: list[str]) -> dict[str, Any]:
        """Re-stage sandbox outputs on the host, then describe what persisted.

        ``register_file`` is called explicitly: a regenerated file already
        carries an id, and ``build_workspace_file_ref`` would short-circuit on
        it and leave the previous generation's bytes as the served version.
        Only paths that registered are described, because that same
        short-circuit would otherwise hand a failed path its stale id back.
        """
        registered: list[str] = []
        for path in paths:
            try:
                workspace.register_file(path)
            except Exception:
                logger.warning(
                    "Host-side re-registration failed for %s", path, exc_info=True
                )
                continue
            registered.append(path)
        if not registered:
            return {"generated_files": [], "file_refs": [], "artifacts": []}
        return build_generated_file_metadata(workspace=workspace, file_paths=registered)

    async def _register_sandbox_outputs(self, result: Any) -> Any:
        """Mint usable file_ids for sandbox-produced files.

        The sandbox has no database credentials, so a file_id minted in there
        names no real record. Refs the host cannot re-register keep the sandbox
        entry: a dropped artifact reads as "nothing was generated", which is
        what makes an agent overwrite a real file to obtain a usable id.
        """
        workspace = getattr(self._reconstruction_target, "_workspace", None)
        if workspace is None:
            logger.debug(
                "%s exposes no _workspace; sandbox outputs stay unregistered",
                self._target.name,
            )
            return result
        if not isinstance(result, dict):
            return result
        original_refs = [
            ref for ref in result.get("file_refs") or [] if isinstance(ref, dict)
        ]
        if not original_refs:
            return result

        # Merge on resolved paths: a guest mount point deliberately keeps the
        # unresolved spelling, while FileRefs carry the resolved one.
        keys = [self._host_output_key(workspace, ref) for ref in original_refs]
        paths = [key for key in keys if key]
        if not paths:
            return result

        rebuilt = await asyncio.to_thread(
            self._reregister_on_host,
            workspace,
            paths,
        )
        rebuilt_by_path = {
            key: ref
            for ref in rebuilt["file_refs"]
            if (key := self._resolved_ref_key(ref))
        }
        if not rebuilt_by_path:
            return result

        # The sandboxed Python, JS and command executors keep artifacts 1:1
        # with file_refs, so the rebuild below is lossless for them and only
        # for them.
        merged = [
            rebuilt_by_path.get(key or "", ref) for key, ref in zip(keys, original_refs)
        ]
        result["file_refs"] = merged
        result["artifacts"] = [build_inline_artifact(ref) for ref in merged]
        result["generated_files"] = [
            str(ref["filename"]) for ref in merged if ref.get("filename")
        ]
        return result

    async def _run_json_in_sandbox(self, args: Mapping[str, Any]) -> Any:
        if self._mcp_target is not None:
            return await self._mcp_target.run_json_async(args)
        request = self._build_runtime_request(args)
        if request["kind"] == "command":
            from ..command_executor import _register_changed_files, _snapshot_artifacts

            workspace = getattr(self._reconstruction_target, "_workspace", None)
            root = request["working_directory"]
            before = (
                await asyncio.to_thread(_snapshot_artifacts, root, workspace)
                if workspace is not None and root
                else {}
            )
            result = await self._execute_runtime_request(request)
            # Only the host's snapshot may describe shell-generated files.
            for field in (
                "generated_files",
                "file_refs",
                "artifacts",
                "registration_note",
                "validation",
            ):
                result.pop(field, None)
            if workspace is not None and root:
                after = await asyncio.to_thread(_snapshot_artifacts, root, workspace)
                changed = {
                    path for path, state in after.items() if before.get(path) != state
                }
                # Reuse the host's bulk-output limits and file validation. The
                # guest receives no workspace object or registration credentials.
                result.update(
                    await asyncio.to_thread(
                        _register_changed_files, workspace, changed, root
                    )
                )
            return result
        return await self._execute_runtime_request(request)

    async def _execute_runtime_request(self, request: Mapping[str, Any]) -> Any:
        """Run primitive execution data with private inputs and bounded cleanup."""
        # Keep every per-call payload in a private sandbox-side file. In
        # particular, MCP transport requests contain the connection env, and
        # command-line base64 would expose those secrets through /proc and
        # process listings to other code running as the same sandbox user.
        call_id = uuid.uuid4().hex
        result_file = f"/tmp/xagent_result_{call_id}.json"
        args_file = f"/tmp/xagent_args_{call_id}.json"

        try:
            async with self._lease_sandbox() as sandbox:
                # Ensure dependencies are installed
                await self._ensure_dependencies(sandbox)
                try:
                    await _write_sandbox_file_cancellation_safe(
                        sandbox,
                        json.dumps(dict(request), ensure_ascii=False),
                        args_file,
                        overwrite=False,
                    )
                    command = build_runtime_command(
                        request_file=args_file, result_file=result_file
                    )
                    timeout_seconds = (
                        self._exec_timeout() if self._exec_timeout else None
                    )
                    if timeout_seconds is not None:
                        # Bound inside the sandbox: `timeout` kills the runner's
                        # whole process group, which cancelling this await would not.
                        # SIGKILL rather than TERM-then-KILL: once the runner dies
                        # `timeout` exits, so a delayed KILL would never reach a
                        # child that ignores SIGTERM.
                        command = [
                            "timeout",
                            "-s",
                            "KILL",
                            str(timeout_seconds),
                            *command,
                        ]
                    started = time.monotonic()
                    result = await sandbox.exec(
                        command[0], *command[1:], env=self._build_execution_env()
                    )

                    if (
                        timeout_seconds is not None
                        and result.exit_code in _SIGKILL_EXIT_CODES
                        and time.monotonic() - started >= timeout_seconds
                    ):
                        raise RuntimeError(
                            f"Tool execution timed out after {timeout_seconds} seconds "
                            "and its sandbox process was terminated. Split the work "
                            "into smaller steps or avoid unbounded operations such as "
                            "recursive scans of '/'."
                        )

                    # Check execution result
                    if result.exit_code != 0:
                        error_msg = (
                            result.stderr or result.error_message or "Unknown error"
                        )
                        logger.error(f"Tool execution failed: {error_msg}")
                        raise RuntimeError(f"Tool execution failed: {error_msg}")

                    # Read output from result file
                    output = ""
                    read_result = await sandbox.exec("cat", result_file)
                    if read_result.exit_code != 0:
                        logger.error(
                            f"Failed to read result file: {read_result.stderr}"
                        )
                        raise RuntimeError(
                            f"Failed to read result file: {read_result.stderr}"
                        )

                    output = read_result.stdout.strip()

                    # Handle empty output
                    if not output:
                        return None

                    return json.loads(output)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse tool output from {result_file}. Raw output:\n{output}"
                    )
                    raise RuntimeError(f"Failed to parse tool output: {e}")
                finally:
                    # Always clean up private inputs, including when writing the
                    # payload, launching the process, or waiting for it
                    # fails or is cancelled. Do this before releasing the worker
                    # lease so another call cannot observe stale credentials.
                    try:
                        await sandbox.exec(
                            "rm",
                            "-f",
                            result_file,
                            args_file,
                            result_file + ".runtime.pyz",
                        )
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"Error executing tool in sandbox: {e}", exc_info=True)
            raise


async def create_sandboxed_tool(
    tool: AbstractBaseTool,
    sandbox: "Sandbox | _SandboxLeaseProviderLike",
) -> SandboxedToolWrapper:
    """
    Create sandboxed tool instance

    Args:
        tool: Tool to wrap
        sandbox: Created sandbox instance, or a lease provider -- retained
            as-is (never resolved to its primary here) so each per-call
            invocation can still take its own worker-slot lease

    Returns:
        Sandboxed tool wrapper
    """

    # Create wrapper
    wrapper = SandboxedToolWrapper(
        target_tool=tool,
        sandbox=sandbox,
    )

    return wrapper


async def create_workspace_in_sandbox(
    sandbox: Sandbox,
    workspace: TaskWorkspace,
) -> None:
    """Create workspace directories inside an independently-backed sandbox.

    This is the fallback for raw/standalone sandbox integrations. Web sandbox
    lease providers expose their host-mount coverage to ``ToolFactory`` and
    skip this exec when the host-created workspace is already visible.

    Args:
        sandbox: Sandbox instance
        workspace: TaskWorkspace instance
    """
    dirs = workspace.get_allowed_dirs()
    if not dirs:
        return

    await sandbox.exec("mkdir", "-p", *dirs)
