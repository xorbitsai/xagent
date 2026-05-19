"""
Generic sandboxed tool wrapper

Execute tool's run_json_sync/async methods in sandbox environment.
"""

import asyncio
import base64
import inspect
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Type

import cloudpickle  # type: ignore[import-untyped]
from pydantic import BaseModel

from ......sandbox.base import Sandbox
from .....workspace import TaskWorkspace
from ..base import AbstractBaseTool, ToolMetadata
from ..function import FunctionTool
from .sandbox_config import (
    extract_bound_method_target,
    resolve_sandbox_config,
)

logger = logging.getLogger(__name__)

# Base path where project source code is mounted inside the sandbox
_SANDBOX_SRC_ROOT = "/app/src"
# Coupled to the mounted package layout inside the sandbox. Update this path if
# ``tool_runner.py`` moves.
_SANDBOX_TOOL_RUNNER_PATH = (
    f"{_SANDBOX_SRC_ROOT}/xagent/core/tools/adapters/vibe/sandboxed_tool/tool_runner.py"
)

_SANDBOX_BASE_DEPENDENCIES = [
    "pydantic>=2.0.0",
    "pydantic-settings",
    "cloudpickle>=3.0.0",
]


def _extract_init_params(instance: Any) -> dict[str, Any]:
    """Extract ``__init__`` parameter values from a tool or method-owner instance.

    Uses ``inspect.signature`` to get parameter names from the class
    ``__init__``, then looks up corresponding attribute values on the
    instance using the naming convention: ``_name`` or ``name``.

    *instance* is typed as ``Any`` because for the ``kind="method"``
    path the caller passes the bound-method owner, which can be any
    class instance — not necessarily an :class:`AbstractBaseTool`
    subclass.

    Args:
        instance: Tool instance or bound-method owner to extract init params from.

    Returns:
        Dict mapping parameter name to its value.
        Empty dict if the class has no init params (beyond *self*).
    """
    sig = inspect.signature(instance.__class__.__init__)

    params: dict[str, Any] = {}
    instance_dict = getattr(instance, "__dict__", {})
    for name in sig.parameters:
        if name == "self":
            continue
        # Look up attribute: _name or name
        found = False
        for attr_name in (f"_{name}", name):
            if attr_name in instance_dict:
                params[name] = instance_dict[attr_name]
                found = True
                break
        if not found:
            logger.warning(
                f"Init param '{name}' not found on {instance.__class__.__name__} "
                f"(tried '_{name}' and '{name}'), skipping"
            )

    return params


def _class_import_path(cls: type[Any]) -> str:
    """Return stable import path for a top-level class."""
    return f"{cls.__module__}:{cls.__name__}"


def _serialize_init_params(params: dict[str, Any]) -> str | None:
    """Serialize init params dict to base64-encoded pickle string.

    Args:
        params: Dict of parameter name -> value

    Returns:
        base64-encoded pickle string, or None if params is empty.

    Raises:
        RuntimeError: If any parameter value is not serializable.
    """
    if not params:
        return None

    try:
        data = cloudpickle.dumps(params)
    except Exception:
        for param_name, value in params.items():
            try:
                cloudpickle.dumps(value)
            except Exception as e:
                raise RuntimeError(
                    f"Init parameter '{param_name}' (type: {type(value).__name__}) "
                    f"is not serializable: {e}. "
                    f"This tool cannot run in sandbox with non-serializable init params."
                ) from e
        raise

    return base64.b64encode(data).decode("ascii")


class SandboxedToolWrapper(AbstractBaseTool):
    """
    Generic sandboxed tool wrapper

    Wrap any AbstractBaseTool as a sandboxed execution version.
    Execute tool logic in isolated environment by mounting the entire xagent library to the sandbox.
    """

    # Per-sandbox dependency tracking: sandbox.name -> installed flag
    _sandbox_deps_installed: dict[str, bool] = {}
    _sandbox_deps_locks: dict[str, asyncio.Lock] = {}
    _locks_lock = asyncio.Lock()  # Protects _sandbox_deps_locks creation

    def __init__(
        self,
        target_tool: AbstractBaseTool,
        sandbox: Sandbox,
    ):
        """
        Initialize sandboxed tool wrapper

        Args:
            target_tool: Target tool to wrap
            sandbox: Sandbox instance
        """
        self._target = target_tool
        self._sandbox = sandbox
        self._sandbox_key = sandbox.name

        sandbox_config = resolve_sandbox_config(target_tool)
        if sandbox_config is None or not sandbox_config.enabled:
            raise RuntimeError(
                f"Tool '{target_tool.name}' is not configured for sandbox runtime."
            )

        # base dependencies + tool dependencies
        self._requirements = _SANDBOX_BASE_DEPENDENCIES + list(sandbox_config.packages)
        self._env_vars = list(sandbox_config.env_vars)

        # Proxy target tool attributes
        self._visibility = getattr(target_tool, "_visibility", None)
        self._allow_users = getattr(target_tool, "_allow_users", None)

        self._execution_spec, reconstruction_target = self._resolve_execution_spec()

        # Extract and serialize init params for sandbox reconstruction
        init_params = _extract_init_params(reconstruction_target)
        self._init_params_b64 = _serialize_init_params(init_params)

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

    def _get_target_workspace(self) -> Optional[TaskWorkspace]:
        """Return the target tool's bound workspace, if any.

        Sandboxed executor tools (JS/Python/command) hold a TaskWorkspace on
        ``_workspace``; for FunctionTool wrappers we check the bound method's
        owner instance instead. Returns None when there is no usable workspace.
        """
        candidates: list[Any] = [self._target]
        if isinstance(self._target, FunctionTool):
            method_target = extract_bound_method_target(self._target)
            if method_target is not None:
                candidates.append(method_target[0])

        for candidate in candidates:
            workspace = getattr(candidate, "_workspace", None) or getattr(
                candidate, "workspace", None
            )
            if isinstance(workspace, TaskWorkspace):
                return workspace
        return None

    def _build_execution_env(self) -> dict[str, str]:
        """Build per-exec environment variables (scoped to this process, not the sandbox)."""
        env = {"PYTHONPATH": _SANDBOX_SRC_ROOT}

        for env_var in self._env_vars:
            value = os.getenv(env_var)
            if value is not None:
                env[env_var] = value
            else:
                logger.warning(f"Environment variable {env_var} not found in host")

        return env

    async def _ensure_dependencies(self) -> None:
        """Ensure dependencies are installed in the sandbox.

        Uses per-sandbox asyncio.Lock to avoid blocking unrelated sandboxes.
        """
        if SandboxedToolWrapper._sandbox_deps_installed.get(self._sandbox_key, False):
            return

        # Get or create per-sandbox lock
        if self._sandbox_key not in SandboxedToolWrapper._sandbox_deps_locks:
            async with SandboxedToolWrapper._locks_lock:
                if self._sandbox_key not in SandboxedToolWrapper._sandbox_deps_locks:
                    SandboxedToolWrapper._sandbox_deps_locks[self._sandbox_key] = (
                        asyncio.Lock()
                    )
        lock = SandboxedToolWrapper._sandbox_deps_locks[self._sandbox_key]

        async with lock:
            # Double-check after acquiring lock
            if SandboxedToolWrapper._sandbox_deps_installed.get(
                self._sandbox_key, False
            ):
                return

            if not self._requirements:
                SandboxedToolWrapper._sandbox_deps_installed[self._sandbox_key] = True
                return

            try:
                requirements_txt = "\n".join(self._requirements)
                await self._sandbox.write_file(
                    content=requirements_txt,
                    remote_path="/tmp/requirements.txt",
                    overwrite=True,
                )

                try:
                    result = await asyncio.wait_for(
                        self._sandbox.exec(
                            "pip",
                            "install",
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

                SandboxedToolWrapper._sandbox_deps_installed[self._sandbox_key] = True

            except Exception as e:
                logger.error(f"Error installing dependencies: {e}")
                raise

    def _resolve_execution_spec(self) -> tuple[dict[str, str], Any]:
        """
        Resolve how to execute the tool in sandbox.

        Returns:
            Execution spec and the instance whose init params should be serialized.
        """
        if isinstance(self._target, FunctionTool):
            function_target = extract_bound_method_target(self._target)
            if function_target is not None:
                instance, method_name = function_target
                return (
                    {
                        "kind": "method",
                        "tool_class": _class_import_path(instance.__class__),
                        "method_name": method_name,
                    },
                    instance,
                )

            raise RuntimeError(
                f"FunctionTool '{self._target.name}' uses a closure or unsupported "
                "callable form that cannot be reconstructed in sandbox automatically."
            )

        return (
            {
                "kind": "tool",
                "tool_class": _class_import_path(self._target.__class__),
            },
            self._target,
        )

    def _build_execution_command(
        self, args: Mapping[str, Any], result_file: str
    ) -> list[str]:
        """Build the sandbox command used to execute a tool runner."""
        args_json = json.dumps(dict(args), ensure_ascii=False)
        args_b64 = base64.b64encode(args_json.encode("utf-8")).decode("ascii")
        execution_spec_json = json.dumps(self._execution_spec, ensure_ascii=False)
        execution_spec_b64 = base64.b64encode(
            execution_spec_json.encode("utf-8")
        ).decode("ascii")

        command = [
            "python",
            _SANDBOX_TOOL_RUNNER_PATH,
            "--execution-spec-b64",
            execution_spec_b64,
            "--args-b64",
            args_b64,
            "--result-file",
            result_file,
        ]
        if self._init_params_b64 is not None:
            command.extend(["--init-params-b64", self._init_params_b64])
        return command

    async def get_sandbox_for_test(self) -> Sandbox:
        """Get the sandbox for exec test"""
        await self._ensure_dependencies()
        return self._sandbox

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Synchronous execution (calls async version via asyncio.run)"""
        return asyncio.run(self.run_json_async(args))

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute tool asynchronously in sandbox"""

        # Generate unique result file name
        result_file = f"/tmp/xagent_result_{uuid.uuid4().hex}.json"

        try:
            # Ensure dependencies are installed
            await self._ensure_dependencies()

            # Execute script in sandbox
            logger.debug(f"Executing tool {self._target.name} in sandbox")
            command = self._build_execution_command(args, result_file)

            # Files written by the sandbox process appear on the host via the
            # mounted workspace volume, but the sandbox cannot reach the host
            # database, so file registration done inside the sandbox is silently
            # dropped. Wrap the exec on the host side so new workspace files get
            # registered with valid file_ids.
            workspace = self._get_target_workspace()
            if workspace is not None:
                with workspace.auto_register_files():
                    result = await self._sandbox.exec(
                        command[0], *command[1:], env=self._build_execution_env()
                    )
            else:
                result = await self._sandbox.exec(
                    command[0], *command[1:], env=self._build_execution_env()
                )

            # Check execution result
            if result.exit_code != 0:
                error_msg = result.stderr or result.error_message or "Unknown error"
                logger.error(f"Tool execution failed: {error_msg}")
                raise RuntimeError(f"Tool execution failed: {error_msg}")

            # Read output from result file
            output = ""
            try:
                read_result = await self._sandbox.exec("cat", result_file)
                if read_result.exit_code != 0:
                    logger.error(f"Failed to read result file: {read_result.stderr}")
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

        except Exception as e:
            logger.error(f"Error executing tool in sandbox: {e}", exc_info=True)
            raise
        finally:
            # Clean up result file
            try:
                await self._sandbox.exec("rm", "-f", result_file)
            except Exception:
                pass


async def create_sandboxed_tool(
    tool: AbstractBaseTool,
    sandbox: Sandbox,
) -> SandboxedToolWrapper:
    """
    Create sandboxed tool instance

    Args:
        tool: Tool to wrap
        sandbox: Created sandbox instance

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
    """Create workspace directories inside the sandbox.

    Args:
        sandbox: Sandbox instance
        workspace: TaskWorkspace instance
    """
    dirs = workspace.get_allowed_dirs()
    if not dirs:
        return

    await sandbox.exec("mkdir", "-p", *dirs)


def _get_project_root() -> Path:
    """Find project root by traversing up to locate pyproject.toml + src/xagent."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (
            parent / "src" / "xagent"
        ).exists():
            return parent
    raise RuntimeError("Could not find project root")


def build_code_mount_volumes() -> list[tuple[str, str, str]]:
    """Build read-only volume mounts for src/ and tests/ directories.

    Returns:
        List of (host_path, guest_path, mode) tuples.
    """
    project_root = _get_project_root()
    volumes: list[tuple[str, str, str]] = []

    src_dir = project_root / "src"
    volumes.append((str(src_dir.resolve()), _SANDBOX_SRC_ROOT, "ro"))

    tests_dir = project_root / "tests"
    if tests_dir.exists():
        volumes.append((str(tests_dir.resolve()), "/app/tests", "ro"))

    return volumes
