"""
Generic sandboxed tool wrapper

Execute tool's run_json_sync/async methods in sandbox environment by uploading the entire xagent library to the sandbox.
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
from .sandboxed_tool_config import get_sandbox_tool_config

logger = logging.getLogger(__name__)


def _extract_init_params(tool: AbstractBaseTool) -> dict[str, Any]:
    """Extract __init__ parameter values from a tool instance.

    Uses inspect.signature to get parameter names, then looks up
    corresponding attribute values on the tool instance using the
    naming convention: _name or name.

    Args:
        tool: Tool instance to extract init params from

    Returns:
        Dict mapping parameter name to its value.
        Empty dict if tool has no init params (beyond self).
    """
    sig = inspect.signature(tool.__class__.__init__)

    params: dict[str, Any] = {}
    instance_dict = getattr(tool, "__dict__", {})
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
                f"Init param '{name}' not found on {tool.__class__.__name__} "
                f"(tried '_{name}' and '{name}'), skipping"
            )

    return params


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

        # Load dependencies and environment variables from config module
        base_requirements = [
            "pydantic>=2.0.0",
            "pydantic-settings",
            "cloudpickle>=3.0.0",
        ]
        tool_config = get_sandbox_tool_config(target_tool.name)
        self._requirements = base_requirements + tool_config.packages
        self._env_vars = tool_config.env_vars

        # Proxy target tool attributes
        self._visibility = getattr(target_tool, "_visibility", None)
        self._allow_users = getattr(target_tool, "_allow_users", None)

        # Extract and serialize init params for sandbox reconstruction
        init_params = _extract_init_params(target_tool)
        self._init_params_b64 = _serialize_init_params(init_params)

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

    def _generate_env_setup(self) -> str:
        """
        Generate environment variable setup code

        Read configured environment variables from host and generate Python code
        to set these variables in the sandbox

        Returns:
            Python code for environment variable setup
        """
        if not self._env_vars:
            return ""

        env_lines = []
        for env_var in self._env_vars:
            # Read environment variable from host
            value = os.getenv(env_var)
            if value is not None:
                # Use json.dumps for safe string encoding (handles \n, quotes, etc.)
                env_lines.append(f"os.environ['{env_var}'] = {json.dumps(value)}")
            else:
                # Environment variable not found, log warning but don't interrupt execution
                logger.warning(f"Environment variable {env_var} not found in host")

        return "\n".join(env_lines)

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

    def _resolve_execution_strategy(self) -> str:
        """
        Resolve how to execute the tool in sandbox.

        Returns:
            tool_class import path (e.g. "module.path:ClassName")
        """
        from .sandboxed_tool_config import get_sandbox_tool_config

        config = get_sandbox_tool_config(self._target.name)

        if config.tool_class:
            return config.tool_class

        raise RuntimeError(
            f"Cannot determine execution strategy for tool '{self._target.name}'. "
            f"Configure 'tool_class' in sandboxed_tool_config.yml"
        )

    def _generate_execution_script(
        self, args: Mapping[str, Any], result_file: str
    ) -> str:
        """
        Generate Python script to execute tool in sandbox.

        Uses tool_class strategy: Class().run_json_sync(args) or run_json_async(args)
        When init params exists, use Class(**params).

        Args:
            args: Tool arguments
            result_file: Result output file path

        Returns:
            Python execution script
        """
        import_path = self._resolve_execution_strategy()
        module_path, name = import_path.rsplit(":", 1)

        # Serialize arguments
        args_json = json.dumps(dict(args), ensure_ascii=False)
        args_b64 = base64.b64encode(args_json.encode("utf-8")).decode("ascii")

        # Generate environment variable setup code
        env_setup = self._generate_env_setup()

        if self._init_params_b64 is not None:
            execution_code = f"""
import cloudpickle

# Deserialize init params
init_params_b64 = '{self._init_params_b64}'
init_params = cloudpickle.loads(base64.b64decode(init_params_b64))

# Import and reconstruct tool class with params
from {module_path} import {name}
tool = {name}(**init_params)

# Execute via tool's run method
import inspect
if inspect.iscoroutinefunction(tool.run_json_async):
    import asyncio
    result = asyncio.run(tool.run_json_async(args))
else:
    result = tool.run_json_sync(args)
"""
        else:
            execution_code = f"""
# Import and reconstruct tool class
from {module_path} import {name}
tool = {name}()

# Execute via tool's run method
import inspect
if inspect.iscoroutinefunction(tool.run_json_async):
    import asyncio
    result = asyncio.run(tool.run_json_async(args))
else:
    result = tool.run_json_sync(args)
"""

        script = f"""
import base64
import json
import os
import sys

# Set environment variables
{env_setup}

# Add xagent to Python path
sys.path.insert(0, '/app/src')

# Decode and parse arguments
args_b64 = '{args_b64}'
args_json = base64.b64decode(args_b64).decode('utf-8')
args = json.loads(args_json)
{execution_code}
# Write result to file
with open('{result_file}', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)
"""
        return script

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

            # Generate execution script
            script = self._generate_execution_script(args, result_file)

            # Write script to sandbox
            script_path = f"/tmp/tool_execution_{uuid.uuid4().hex}.py"
            await self._sandbox.write_file(
                content=script,
                remote_path=script_path,
                overwrite=True,
            )

            # Execute script in sandbox
            logger.debug(f"Executing tool {self._target.name} in sandbox")
            result = await self._sandbox.exec("python", script_path)

            # Clean up script path
            try:
                await self._sandbox.exec("rm", "-f", script_path)
            except Exception:
                pass

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


def _calculate_tar_hash(tar_path: str) -> str:
    """
    Calculate SHA256 hash of TAR file

    Args:
        tar_path: TAR file path

    Returns:
        Hash string (first 16 characters of SHA256)
    """
    import hashlib

    hasher = hashlib.sha256()
    with open(tar_path, "rb") as f:
        # Read in chunks to avoid excessive memory usage for large files
        while chunk := f.read(8192):
            hasher.update(chunk)

    # Return first 16 characters (sufficient for version comparison)
    return hasher.hexdigest()[:16]


async def upload_code_to_sandbox(
    sandbox: Any, force_upload: bool = False, contain_tests: bool = False
) -> None:
    """
    Package and upload xagent code to sandbox to enable tool execution in the sandbox.

    Args:
        sandbox: Sandbox instance
        force_upload: Whether to force upload (ignore version check)
        contain_tests: Whether to include tests directory in the tar package
    """
    import tarfile
    import tempfile

    # Auto-detect xagent root directory
    current_file = Path(__file__).resolve()

    # Traverse up to find project root marker
    xagent_root = None
    for parent in list(current_file.parents):
        # Check if project root marker file exists
        if (parent / "pyproject.toml").exists():
            # Verify src/xagent directory exists
            if (parent / "src" / "xagent").exists():
                xagent_root = str(parent)
                logger.debug(f"Auto-detected xagent root: {xagent_root}")
                break

    if xagent_root is None:
        raise RuntimeError("Could not auto-detect xagent root directory")

    root_path = Path(xagent_root)

    # Collect directories to package: always include src, optionally include tests
    dirs_to_package: list[tuple[Path, str]] = [
        (root_path / "src", "src"),
    ]
    if contain_tests:
        tests_dir = root_path / "tests"
        if tests_dir.exists():
            dirs_to_package.append((tests_dir, "tests"))
            logger.debug("Including tests directory in upload package")
        else:
            logger.warning("contain_tests=True but tests directory not found, skipping")

    # Create temporary TAR file
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = tmp.name

    try:
        # Package code (without compression to ensure reproducible hash)
        with tarfile.open(tar_path, "w") as tar:
            for source_dir, prefix in dirs_to_package:
                for root, dirs, files in os.walk(source_dir):
                    # Skip unnecessary directories
                    dirs[:] = [
                        d
                        for d in dirs
                        if d
                        not in {
                            "__pycache__",
                            ".git",
                            ".venv",
                            "node_modules",
                            ".mypy_cache",
                            ".pytest_cache",
                            ".ruff_cache",
                        }
                    ]

                    for file in files:
                        # Skip unnecessary files
                        if file.endswith((".pyc", ".pyo", ".pyd", ".so", ".DS_Store")):
                            continue

                        file_path = Path(root) / file
                        # Calculate relative path with correct prefix
                        # For src: src/xagent/... ; For tests: tests/...
                        rel_path = file_path.relative_to(source_dir)
                        arcname = f"{prefix}/{rel_path}"

                        # Create TarInfo and set fixed timestamp (ensure reproducible hash)
                        tarinfo = tar.gettarinfo(str(file_path), arcname=arcname)
                        tarinfo.mtime = 0  # Fixed timestamp to 1970-01-01
                        tarinfo.uid = 0
                        tarinfo.gid = 0
                        tarinfo.uname = ""
                        tarinfo.gname = ""

                        with open(file_path, "rb") as f:
                            tar.addfile(tarinfo, f)

        # Calculate TAR file hash
        current_version = _calculate_tar_hash(tar_path)
        tar_size = os.path.getsize(tar_path) / 1024 / 1024
        logger.debug(f"TAR package created: {tar_size:.2f} MB, hash: {current_version}")

        # Check version in sandbox (version file in /app directory)
        if not force_upload:
            version_check = await sandbox.exec("cat", "/app/.xagent_version")
            if version_check.exit_code == 0:
                sandbox_version = version_check.stdout.strip()
                if sandbox_version == current_version:
                    logger.info("Code already up-to-date in sandbox, skipping upload")
                    return
                logger.debug(
                    f"Sandbox version mismatch: {sandbox_version} != {current_version}"
                )

        logger.info("Uploading code to sandbox...")

        # upload to tmp
        await sandbox.upload_file(tar_path, "/tmp/xagent_code.tar", overwrite=True)

        logger.debug("Extracting TAR in sandbox...")

        temp_dir = f"/tmp/xagent_src_{uuid.uuid4().hex[:8]}"
        await sandbox.exec("mkdir", "-p", temp_dir)

        # extract to tmp dir
        result = await sandbox.exec(
            "tar", "-xf", "/tmp/xagent_code.tar", "-C", temp_dir
        )

        if result.exit_code != 0:
            error_msg = result.stderr or "Unknown error"
            logger.error(f"Failed to extract TAR: {error_msg}")
            raise RuntimeError(f"Failed to extract TAR: {error_msg}")

        # Write version marker file to /app directory
        await sandbox.write_file(
            content=current_version,
            remote_path="/app/.xagent_version",
            overwrite=True,
        )

        # Code update
        await sandbox.exec("mkdir", "-p", "/app")
        await sandbox.exec("rm", "-rf", "/app/src")
        result = await sandbox.exec("mv", f"{temp_dir}/src", "/app/src")

        if result.exit_code != 0:
            error_msg = result.stderr or "Unknown error"
            logger.error(f"Failed to move src directory: {error_msg}")
            raise RuntimeError(f"Failed to move src directory: {error_msg}")

        # Move tests directory if included
        if contain_tests:
            await sandbox.exec("rm", "-rf", "/app/tests")
            mv_tests = await sandbox.exec("mv", f"{temp_dir}/tests", "/app/tests")
            if mv_tests.exit_code != 0:
                logger.warning(
                    f"Failed to move tests directory: {mv_tests.stderr or 'Unknown error'}"
                )

        # Clean up temp extraction dir
        await sandbox.exec("rm", "-rf", temp_dir)

        logger.info(f"Code uploaded successfully (version: {current_version})")

        # clean up tmp file
        await sandbox.exec("rm", "-f", "/tmp/xagent_code.tar")

    finally:
        # clean up tmp file
        if os.path.exists(tar_path):
            os.unlink(tar_path)
