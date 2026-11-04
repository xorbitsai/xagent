"""Real-time progress broadcasting for live updates."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Protocol, Set

from ..core.schemas import DocumentProcessingStatus, ProgressUpdateEvent, TaskProgress

logger = logging.getLogger(__name__)


class WebSocketConnection(Protocol):
    """Protocol for WebSocket-like connections."""

    async def send_text(self, data: str) -> None:
        """Send text data through the connection."""
        ...

    def is_connected(self) -> bool:
        """Check if connection is still active."""
        ...


class ProgressBroadcaster:
    """Broadcasts progress updates to connected clients in real-time."""

    def __init__(self) -> None:
        self._connections: Dict[str, Set[WebSocketConnection]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, task_id: str, connection: WebSocketConnection) -> None:
        """Connect a WebSocket to receive progress updates for a task.

        Args:
            task_id: Task identifier to subscribe to
            connection: WebSocket connection
        """
        async with self._lock:
            if task_id not in self._connections:
                self._connections[task_id] = set()
            self._connections[task_id].add(connection)

            logger.info("WebSocket connected for task %s", task_id)

    async def disconnect(self, task_id: str, connection: WebSocketConnection) -> None:
        """Disconnect a WebSocket from task updates.

        Args:
            task_id: Task identifier
            connection: WebSocket connection to remove
        """
        async with self._lock:
            if task_id in self._connections:
                self._connections[task_id].discard(connection)
                if not self._connections[task_id]:
                    del self._connections[task_id]

                logger.info("WebSocket disconnected from task %s", task_id)

    async def broadcast_progress(self, task_progress: TaskProgress) -> None:
        """Broadcast progress update to all subscribers of the task.

        Args:
            task_progress: Current task progress
        """
        import time

        event = ProgressUpdateEvent(
            task_id=task_progress.task_id,
            task_type=task_progress.task_type,
            status=task_progress.status,
            current_step=task_progress.current_step,
            overall_progress=task_progress.overall_progress,
            timestamp=time.time(),
            data={
                "start_time": task_progress.start_time,
                "end_time": task_progress.end_time,
                "metadata": task_progress.metadata,
            },
        )

        await self._broadcast_event(task_progress.task_id, event)

    async def broadcast_event(
        self, task_id: str, event_type: str, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast a custom event to task subscribers.

        Args:
            task_id: Task identifier
            event_type: Type of event
            data: Event data
        """
        import time

        # Create a minimal event for custom types
        # Note: We need a status and progress even for custom events to satisfy the schema
        # Ideally we'd have a separate schema for generic events, but reusing ProgressUpdateEvent works
        event = ProgressUpdateEvent(
            event_type=event_type,
            task_id=task_id,
            task_type="custom",  # placeholder
            status=DocumentProcessingStatus.RUNNING,  # placeholder
            overall_progress=0.0,
            timestamp=time.time(),
            data=data or {},
        )

        await self._broadcast_event(task_id, event)

    async def _broadcast_event(self, task_id: str, event: ProgressUpdateEvent) -> None:
        """Internal method to broadcast event to task subscribers."""
        async with self._lock:
            connections = self._connections.get(task_id, set()).copy()

        if not connections:
            return

        # Convert event to JSON using Pydantic's model_dump_json
        try:
            message = event.model_dump_json()
        except Exception as e:
            logger.error("Failed to serialize progress event: %s", e)
            return

        # Send to all connections
        disconnected = []
        for connection in connections:
            try:
                if connection.is_connected():
                    await connection.send_text(message)
                else:
                    disconnected.append(connection)
            except Exception as e:
                logger.warning("Failed to send progress update to connection: %s", e)
                disconnected.append(connection)

        # Clean up disconnected connections
        if disconnected:
            async with self._lock:
                task_connections = self._connections.get(task_id)
                if task_connections:
                    for conn in disconnected:
                        task_connections.discard(conn)
                    if not task_connections:
                        del self._connections[task_id]

    async def get_connection_count(self, task_id: str) -> int:
        """Get number of active connections for a task.

        Args:
            task_id: Task identifier

        Returns:
            Number of active connections
        """
        async with self._lock:
            return len(self._connections.get(task_id, set()))

    async def cleanup_task(self, task_id: str) -> None:
        """Clean up all connections for a completed task.

        Args:
            task_id: Task identifier
        """
        async with self._lock:
            if task_id in self._connections:
                # Close all connections
                connections = self._connections[task_id]
                for connection in connections:
                    try:
                        # Note: In real WebSocket implementation, you might want to
                        # send a completion message before closing
                        pass
                    except Exception:
                        pass

                del self._connections[task_id]
                logger.info("Cleaned up connections for completed task %s", task_id)

    async def _reset(self) -> None:
        """Reset the broadcaster state (FOR TESTING ONLY)."""
        async with self._lock:
            self._connections.clear()


# Global broadcaster instance
progress_broadcaster = ProgressBroadcaster()
