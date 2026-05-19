"""
Process service using xoscar with sub-pool creation.

Creates a main pool at startup, then creates sub-pools (with one worker) for
each execution request using append_sub_pool, destroys them after completion.
Sub-pools are created on-demand and destroyed when done.
"""

import asyncio
import logging
import os
import sys
import uuid
from typing import Any, Optional

import xoscar as xo

from .base import (
    BaseService,
    ExecutionResult,
    ServiceInfo,
    ServiceStatus,
)

logger = logging.getLogger(__name__)


class ProcessService(BaseService):
    """Process service using xoscar with sub-pool creation.

    Creates a main pool at startup.
    For each execution request, appends a sub-pool with one worker.
    After execution, the sub-pool is killed.
    """

    def __init__(self, address: str = "localhost:12345", n_workers: int = 0):
        super().__init__()
        self._address = address
        self._n_workers = n_workers
        self._max_concurrency = n_workers if n_workers > 0 else (os.cpu_count() or 1)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._lock = asyncio.Lock()
        self._active_actors: dict[str, Any] = {}
        self._pool: Any = None

    @property
    def service_name(self) -> str:
        return "process"

    async def start(self) -> None:
        """Start process service."""
        self._status = ServiceStatus.STARTING
        try:
            # Initialize xoscar router
            from xoscar.backends import router as xo_router

            default_router = xo_router.Router.get_instance_or_empty()
            xo_router.Router.set_instance(default_router)
            logger.info("xoscar router initialized")

            # Create main pool that will hold sub-pools
            self._pool = await xo.create_actor_pool(
                address=self._address,
                n_process=0,  # Main pool doesn't need workers, only sub-pools
            )

            self._status = ServiceStatus.RUNNING
            logger.info(
                f"ProcessService started successfully with main pool at {self._address}"
            )
        except Exception as e:
            self._status = ServiceStatus.ERROR
            logger.error(f"Failed to start ProcessService: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """Stop process service."""
        self._status = ServiceStatus.STOPPING
        try:
            logger.info("Stopping ProcessService")

            # Stop main pool (this will also stop all sub-pools)
            if self._pool:
                await self._pool.stop()
                self._pool = None
                logger.info("Main pool stopped")

            self._status = ServiceStatus.STOPPED
            logger.info("ProcessService stopped successfully")
        except Exception as e:
            self._status = ServiceStatus.ERROR
            logger.error(f"Failed to stop ProcessService: {e}", exc_info=True)
            raise

    async def health_check(self) -> bool:
        """Health check."""
        return self._status == ServiceStatus.RUNNING

    def get_info(self) -> ServiceInfo:
        """Get service information."""
        return ServiceInfo(
            name=self.service_name,
            status=self._status,
            resource_info={
                "type": "dynamic",
                "address": self._address,
                "n_workers": self._n_workers,
                "max_concurrency": self._max_concurrency,
                "active_actors": len(self._active_actors),
            },
            metrics={},
        )

    def _ensure_running(self) -> None:
        if self._status != ServiceStatus.RUNNING or self._pool is None:
            raise RuntimeError("ProcessService not started")

    def _sub_pool_env(self) -> dict[str, str]:
        """Build environment for child pools so xoscar can import caller modules."""
        cwd = os.getcwd()
        path_entries = []
        for path in sys.path:
            abs_path = os.path.abspath(path or cwd)
            if abs_path == cwd or abs_path.startswith(cwd + os.sep):
                path_entries.append(abs_path)

        env = {}
        existing_pythonpath = os.environ.get("PYTHONPATH")
        if existing_pythonpath:
            path_entries.extend(existing_pythonpath.split(os.pathsep))
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path_entries))
        return env

    async def _execute_in_actor(
        self,
        task_id: str,
        actor_cls: Any,
        timeout: int,
        **execute_kwargs: Any,
    ) -> ExecutionResult:
        """Execute an actor call in a temporary sub-pool."""
        self._ensure_running()

        async with self._semaphore:
            return await self._execute_in_sub_pool(
                task_id,
                actor_cls,
                timeout,
                **execute_kwargs,
            )

    async def _execute_in_sub_pool(
        self,
        task_id: str,
        actor_cls: Any,
        timeout: int,
        **execute_kwargs: Any,
    ) -> ExecutionResult:
        sub_pool_address = None
        actor_ref = None
        timed_out = False
        startup_timeout = min(float(timeout), 30.0)
        timeout_stage = "sub-pool startup"
        timeout_seconds = startup_timeout

        try:
            sub_pool_address = await asyncio.wait_for(
                self._pool.append_sub_pool(
                    label=task_id,
                    env=self._sub_pool_env(),
                ),
                timeout=startup_timeout,
            )
            logger.debug(f"Appended sub-pool {sub_pool_address} for {task_id}")

            timeout_stage = "actor creation"
            actor_ref = await asyncio.wait_for(
                xo.create_actor(actor_cls, address=sub_pool_address),
                timeout=startup_timeout,
            )
            async with self._lock:
                self._active_actors[task_id] = actor_ref

            logger.debug(f"Created actor {task_id}")
            actor = await xo.actor_ref(actor_ref)

            timeout_stage = "actor execution"
            timeout_seconds = float(timeout)
            result_dict = await asyncio.wait_for(
                actor.execute(**execute_kwargs, timeout=timeout),
                timeout=timeout,
            )
            return ExecutionResult.from_dict(result_dict)

        except asyncio.TimeoutError:
            timed_out = True
            return ExecutionResult(
                success=False,
                output="",
                error=(
                    f"Execution timed out during {timeout_stage} "
                    f"after {timeout_seconds:g} seconds"
                ),
                return_code=-1,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Execution failed: {str(e)}",
                return_code=-1,
            )
        finally:
            if actor_ref is not None:
                if not timed_out:
                    try:
                        await xo.destroy_actor(actor_ref)
                        logger.debug(f"Destroyed actor {task_id}")
                    except Exception as e:
                        logger.error(f"Failed to destroy actor {task_id}: {e}")
                async with self._lock:
                    self._active_actors.pop(task_id, None)

            if sub_pool_address and self._pool is not None:
                try:
                    await self._pool.remove_sub_pool(sub_pool_address, force=timed_out)
                    logger.debug(f"Removed sub-pool {sub_pool_address}")
                except Exception as e:
                    logger.error(f"Failed to remove sub-pool {sub_pool_address}: {e}")

    async def execute_python(
        self,
        code: str,
        workspace: Optional[str] = None,
        timeout: int = 300,
    ) -> ExecutionResult:
        """Execute Python code in a dynamic actor.

        Creates a sub-pool, creates an actor, executes the code,
        then destroys both the actor and sub-pool.
        """
        task_id = f"python_{uuid.uuid4().hex[:8]}"
        from ..actors.python_executor_actor import PythonExecutorActor

        return await self._execute_in_actor(
            task_id,
            PythonExecutorActor,
            timeout,
            code=code,
            workspace=workspace,
        )

    async def execute_tool(
        self,
        tool: Any,
        args: dict,
        timeout: int = 300,
    ) -> ExecutionResult:
        """Execute any tool in a dynamic actor.

        Creates a sub-pool, creates an actor, executes the tool,
        then destroys both the actor and sub-pool.
        """
        task_id = f"tool_{uuid.uuid4().hex[:8]}"
        from ..actors.tool_executor_actor import ToolExecutorActor

        return await self._execute_in_actor(
            task_id,
            ToolExecutorActor,
            timeout,
            tool=tool,
            args=args,
        )

    async def execute_command(
        self,
        command: str,
        workspace: Optional[str] = None,
        timeout: int = 300,
    ) -> ExecutionResult:
        """Execute a shell command in a dynamic actor."""
        from ..actors.command_executor_actor import CommandExecutorActor

        task_id = f"command_{uuid.uuid4().hex[:8]}"
        return await self._execute_in_actor(
            task_id,
            CommandExecutorActor,
            timeout,
            command=command,
            workspace=workspace,
        )

    async def execute_javascript(
        self,
        code: str,
        workspace: Optional[str] = None,
        timeout: int = 300,
    ) -> ExecutionResult:
        """Execute JavaScript code in a dynamic actor."""
        from ..actors.javascript_executor_actor import JavaScriptExecutorActor

        task_id = f"javascript_{uuid.uuid4().hex[:8]}"
        return await self._execute_in_actor(
            task_id,
            JavaScriptExecutorActor,
            timeout,
            code=code,
            workspace=workspace,
        )
