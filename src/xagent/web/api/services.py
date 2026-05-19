"""
API endpoints for execution service management.

Provides endpoints to check service status and health.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ...core.execution.service.manager import get_process_service
from ..auth_dependencies import get_current_user
from ..models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/process-service", tags=["Process Service"])


@router.get("/status")
async def get_service_status(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get ProcessService status.

    Returns:
        Dictionary with service status information
    """
    process_service = get_process_service()
    if not process_service:
        return {
            "enabled": False,
            "status": "not_initialized",
            "message": "Process isolation is not enabled or failed to initialize",
        }

    try:
        info = process_service.get_info()
        return {
            "enabled": True,
            "status": info.status.value,
            "resource_info": info.resource_info,
            "metrics": info.metrics,
        }
    except Exception as e:
        logger.error(f"Failed to get service status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check(
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Check ProcessService health.

    Returns:
        Dictionary with health check result
    """
    process_service = get_process_service()
    if not process_service:
        return {
            "healthy": False,
            "message": "ProcessService not initialized",
        }

    try:
        is_healthy = await process_service.health_check()
        return {
            "healthy": is_healthy,
            "message": "Service is healthy" if is_healthy else "Service is unhealthy",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "healthy": False,
            "message": f"Health check failed: {str(e)}",
        }
