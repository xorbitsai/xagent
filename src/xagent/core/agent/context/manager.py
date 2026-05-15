from __future__ import annotations

import logging
import threading
from typing import Any

from .execution import ExecutionContext

logger = logging.getLogger(__name__)


class ContextManager:
    """Singleton registry for active execution contexts."""

    _instance: "ContextManager" | None = None
    _instance_lock = threading.Lock()
    _contexts: dict[str, ExecutionContext]
    _lock: threading.RLock

    def __new__(cls) -> "ContextManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._contexts = {}
                cls._instance._lock = threading.RLock()
        return cls._instance

    def create_context(
        self,
        execution_id: str,
        user_id: str | None = None,
        session_id: str | None = None,
        system_prompt: str | None = None,
        *,
        workspace_id: str | None = None,
        workspace_path: str | None = None,
        cwd: str | None = None,
        workspace_state: dict[str, Any] | None = None,
        memory_session_id: str | None = None,
        memory_snapshot: dict[str, Any] | None = None,
    ) -> ExecutionContext:
        context = ExecutionContext(
            execution_id=execution_id,
            user_id=user_id,
            session_id=session_id,
            system_prompt=system_prompt,
        )
        if any(
            value is not None
            for value in (workspace_id, workspace_path, cwd, workspace_state)
        ):
            context.attach_workspace(
                workspace_id=workspace_id,
                workspace_path=workspace_path,
                cwd=cwd,
                state=workspace_state,
            )
        if memory_session_id or memory_snapshot is not None:
            context.attach_memory_session(
                session_id=memory_session_id,
                snapshot=memory_snapshot,
            )
        with self._lock:
            if execution_id in self._contexts:
                logger.warning("Replacing existing execution context %s", execution_id)
            self._contexts[execution_id] = context
        return context

    def get_context(self, execution_id: str) -> ExecutionContext | None:
        with self._lock:
            return self._contexts.get(execution_id)

    def set_context(self, context: ExecutionContext) -> ExecutionContext:
        with self._lock:
            if context.execution_id in self._contexts:
                logger.warning(
                    "Replacing existing execution context %s",
                    context.execution_id,
                )
            self._contexts[context.execution_id] = context
        return context

    def remove_context(self, execution_id: str) -> None:
        with self._lock:
            self._contexts.pop(execution_id, None)

    def list_active_contexts(
        self, user_id: str | None = None
    ) -> list[ExecutionContext]:
        with self._lock:
            contexts = list(self._contexts.values())
        if user_id:
            return [ctx for ctx in contexts if ctx.user_id == user_id]
        return contexts
