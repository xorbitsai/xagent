"""Shell command executor actor."""

import asyncio
from typing import Optional

from ...tools.core.command_executor import CommandExecutorCore
from .base_executor_actor import BaseExecutorActor


class CommandExecutorActor(BaseExecutorActor):
    """Execute shell commands in an isolated process."""

    async def execute(
        self,
        command: str,
        workspace: Optional[str] = None,
        timeout: int = 300,
    ) -> dict:
        async def _execute() -> dict:
            executor = CommandExecutorCore(working_directory=workspace)
            return await asyncio.to_thread(
                executor.execute_command,
                command,
                timeout=timeout,
            )

        return await self._execute_async_with_tracking(_execute)
