"""Progress monitoring WebSocket API endpoints."""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)

from ...core.tools.core.RAG_tools.core.schemas import ProgressUpdateEvent
from ...core.tools.core.RAG_tools.progress import (
    get_progress_manager,
    progress_broadcaster,
)
from ..auth_dependencies import get_current_user
from ..models.user import User
from ..services.db_runtime import drain_async_task_cancellation_safe
from ..services.websocket_writer import (
    activate_websocket_writer,
    retire_websocket_writer,
    send_websocket_text,
)
from .websocket_auth import (
    _WebSocketAuthenticationTerminated,
    get_authenticated_user,
)

logger = logging.getLogger(__name__)

# Create router for progress WebSocket endpoints
progress_ws_router = APIRouter()

# Global progress manager instance (via singleton accessor)
# progress_manager = get_progress_manager()
# broadcaster = progress_broadcaster


@progress_ws_router.websocket("/ws/progress/{task_id}")
async def progress_websocket_endpoint(
    websocket: WebSocket,
    task_id: str,
    token: str = Query(..., description="Authentication token"),
) -> None:
    """WebSocket endpoint for real-time progress monitoring."""
    try:
        # Verify token and get user
        try:
            user = await get_authenticated_user(websocket, token)
        except _WebSocketAuthenticationTerminated:
            return

        if not user:
            # Token validation failed but didn't raise exception
            await websocket.close(code=1008)  # Policy Violation
            return

        # Check if task exists and belongs to the user
        task_progress = get_progress_manager().get_task_progress(task_id)
        if (
            task_progress
            and task_progress.user_id
            and str(task_progress.user_id) != str(user.id)
        ):
            logger.warning(
                f"User {user.id} attempted to access task {task_id} owned by {task_progress.user_id}"
            )
            await websocket.close(code=1008)  # Policy Violation
            return

        await websocket.accept()
        activate_websocket_writer(websocket)
        logger.info(
            f"Progress WebSocket connected for task {task_id} (user: {user.id})"
        )

        # Create a simple WebSocket connection adapter for the broadcaster
        class WebSocketAdapter:
            def __init__(self, ws: WebSocket):
                self.ws = ws

            async def send_text(self, data: str) -> None:
                await send_websocket_text(self.ws, data)

            def is_connected(self) -> bool:
                # WebSocket doesn't have a direct is_connected method
                # We'll assume it's connected until disconnect
                return True

        adapter = WebSocketAdapter(websocket)

        try:
            # Connect to progress broadcaster
            await progress_broadcaster.connect(task_id, adapter)

            # Send current task status if available
            task_progress = get_progress_manager().get_task_progress(task_id)
            if task_progress:
                # Send initial status using proper JSON serialization
                event = ProgressUpdateEvent(
                    task_id=task_progress.task_id,
                    task_type=task_progress.task_type,
                    status=task_progress.status,
                    current_step=task_progress.current_step,
                    overall_progress=task_progress.overall_progress or 0.0,
                    timestamp=task_progress.start_time or 0,
                    data={
                        "start_time": task_progress.start_time,
                        "end_time": task_progress.end_time,
                        "metadata": task_progress.metadata,
                    },
                    event_type="progress_update",
                )
                await adapter.send_text(event.model_dump_json())

            try:
                # Keep connection alive and listen for client messages
                while True:
                    # Wait for client messages (though we mainly send updates)
                    data = await websocket.receive_text()
                    logger.debug(f"Received message from progress client: {data}")

                    # For now, we don't handle client messages
                    # Could be extended to handle commands like "cancel_task"

            except WebSocketDisconnect:
                logger.info(f"Progress WebSocket disconnected for task {task_id}")
            except Exception as e:
                logger.error(f"Progress WebSocket error for task {task_id}: {e}")
        finally:
            retire_websocket_writer(websocket)
            cleanup = asyncio.create_task(
                progress_broadcaster.disconnect(task_id, adapter)
            )
            await drain_async_task_cancellation_safe(cleanup)

    except WebSocketDisconnect:
        logger.info(f"Progress WebSocket disconnected for task {task_id}")
    except Exception as e:
        logger.error(f"Progress WebSocket error: {e}")
        try:
            await websocket.close(code=1011)  # Internal error
        except Exception:
            pass  # Ignore cleanup errors


@progress_ws_router.get("/api/progress/{task_id}")
async def get_task_progress(
    task_id: str,
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get current progress status for a task."""
    try:
        task_progress = get_progress_manager().get_task_progress(task_id)

        if not task_progress:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Ownership check
        if task_progress.user_id and str(task_progress.user_id) != str(user.id):
            raise HTTPException(
                status_code=403, detail="Not authorized to access this task"
            )

        return {
            "task_id": task_progress.task_id,
            "task_type": task_progress.task_type,
            "status": task_progress.status.value,
            "current_step": task_progress.current_step,
            "overall_progress": task_progress.overall_progress or 0.0,
            "start_time": task_progress.start_time,
            "end_time": task_progress.end_time,
            "metadata": task_progress.metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task progress for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@progress_ws_router.get("/api/progress")
async def list_active_tasks(
    task_type: Optional[str] = Query(None, description="Filter by task type"),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """List all active progress tasks for the current user."""
    try:
        active_tasks = get_progress_manager().get_active_tasks(user_id=int(user.id))

        # Filter by task type if specified
        if task_type:
            active_tasks = {
                task_id: task
                for task_id, task in active_tasks.items()
                if task.task_type == task_type
            }

        return {
            "tasks": [
                {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": task.status.value,
                    "current_step": task.current_step,
                    "overall_progress": task.overall_progress or 0.0,
                    "start_time": task.start_time,
                    "end_time": task.end_time,
                    "metadata": task.metadata,
                }
                for task in active_tasks.values()
            ],
            "count": len(active_tasks),
        }

    except Exception as e:
        logger.error(f"Error listing active tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
