from __future__ import annotations

import logging
import threading
from typing import Any

from .execution import TOOL_EVIDENCE_REMOVED_METADATA_KEY, ExecutionContext

logger = logging.getLogger(__name__)


class ContextManager:
    """Singleton registry for active execution contexts."""

    _instance: "ContextManager" | None = None
    _instance_lock = threading.Lock()
    _contexts: dict[str, ExecutionContext]
    # execution_id -> [reader_count, epoch]; present only while readers exist.
    _cold_starts: dict[str, list[int]]
    _lock: threading.RLock

    def __new__(cls) -> "ContextManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._contexts = {}
                cls._instance._cold_starts = {}
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
        # Stamped on every context this build creates. An absent key therefore
        # means one of two things, and both read as unknown: a payload written by
        # a build that did not track this, or a marker ``from_dict`` dropped because
        # the payload named no writer. See tool_evidence_state and
        # EVIDENCE_MARKER_WRITER_FIELD.
        context.metadata[TOOL_EVIDENCE_REMOVED_METADATA_KEY] = False
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

    def begin_cold_start(self, execution_id: str) -> int:
        """Register a checkpoint reader; the returned token feeds ``end_cold_start``."""
        with self._lock:
            entry = self._cold_starts.setdefault(execution_id, [0, 0])
            entry[0] += 1
            return entry[1]

    def end_cold_start(
        self,
        execution_id: str,
        token: int,
        context: ExecutionContext | None,
    ) -> ExecutionContext | None:
        """Publish a cold-started context unless it may already be stale.

        Concurrent readers share whichever context is cached first. A reader
        whose token predates an eviction gets ``None``: its checkpoint read may
        have raced the evicted context's final write, so it must read again.
        """
        with self._lock:
            entry = self._cold_starts.get(execution_id)
            epoch = entry[1] if entry is not None else token
            if entry is not None:
                entry[0] -= 1
                if entry[0] <= 0:
                    del self._cold_starts[execution_id]
            cached = self._contexts.get(execution_id)
            if cached is not None:
                return cached
            if context is None or epoch != token:
                return None
            self._contexts[execution_id] = context
            return context

    def discard_context(self, execution_id: str, expected: ExecutionContext) -> bool:
        """Evict ``expected`` only if it is still the cached context."""
        with self._lock:
            if self._contexts.get(execution_id) is not expected:
                return False
            del self._contexts[execution_id]
            self._note_eviction(execution_id)
            return True

    def remove_context(self, execution_id: str) -> None:
        with self._lock:
            if self._contexts.pop(execution_id, None) is not None:
                self._note_eviction(execution_id)

    def _note_eviction(self, execution_id: str) -> None:
        # Caller holds the lock.
        entry = self._cold_starts.get(execution_id)
        if entry is not None:
            entry[1] += 1

    def list_active_contexts(
        self, user_id: str | None = None
    ) -> list[ExecutionContext]:
        with self._lock:
            contexts = list(self._contexts.values())
        if user_id:
            return [ctx for ctx in contexts if ctx.user_id == user_id]
        return contexts
