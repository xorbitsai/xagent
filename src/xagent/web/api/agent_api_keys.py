"""Centralized multi-key admin endpoints for agent SDK API keys.

Unlike the legacy single-key endpoints at ``/api/agents/{agent_id}/api-key``
(still available, see ``api/agents.py``), these let a caller list, create,
pause/resume, regenerate, and delete keys across *all* of their agents --
an agent may hold any number of simultaneously-active keys. All routes are
JWT-gated and scoped to the caller's own agents via
``AgentApiKeyService``'s ownership joins.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.user import User
from ..schemas.agent_api_key import (
    AgentApiKeyCreateRequest,
    AgentApiKeyListItem,
    AgentApiKeyStats,
    APIKeyGenerateResponse,
)
from ..services.api_keys import AgentApiKeyService, KeyRotationConflict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent-api-keys", tags=["agent-api-keys"])


@router.get("", response_model=List[AgentApiKeyListItem])
async def list_agent_api_keys(
    agent_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[AgentApiKeyListItem]:
    """List API keys across every agent the caller owns.

    Optionally filtered to a single ``agent_id`` (used by the "Manage API
    Key" jump-link from an agent's card/deploy dialog).
    """
    return AgentApiKeyService(db).list_keys_for_user(
        int(current_user.id), agent_id=agent_id
    )


@router.get("/stats", response_model=AgentApiKeyStats)
async def get_agent_api_key_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentApiKeyStats:
    """Aggregate counters for the API Keys page's stat cards."""
    return AgentApiKeyService(db).get_stats_for_user(int(current_user.id))


@router.post("", response_model=APIKeyGenerateResponse)
async def create_agent_api_key(
    request: AgentApiKeyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> APIKeyGenerateResponse:
    """Add a new key for an agent without touching its other keys.

    Raises 404 if ``agent_id`` doesn't exist or isn't owned by the caller
    (same ownership-hiding rationale as ``_get_owned_agent_or_404``).
    """
    try:
        result = AgentApiKeyService(db).create_key(
            int(current_user.id), request.agent_id, request.label
        )
    except KeyRotationConflict as e:
        logger.warning(
            f"Concurrent API key creation race for agent {request.agent_id}: {e}"
        )
        raise HTTPException(status_code=409, detail="rotation_conflict")
    except Exception as e:
        logger.error(f"Failed to create API key for agent {request.agent_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

    if result is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    item, full_key = result
    return APIKeyGenerateResponse(
        full_key=full_key, key_prefix=item.key_prefix, created_at=item.created_at
    )


@router.post("/{key_id}/pause", response_model=AgentApiKeyListItem)
async def pause_agent_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentApiKeyListItem:
    item = AgentApiKeyService(db).pause_key(int(current_user.id), key_id)
    if item is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return item


@router.post("/{key_id}/resume", response_model=AgentApiKeyListItem)
async def resume_agent_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentApiKeyListItem:
    item = AgentApiKeyService(db).resume_key(int(current_user.id), key_id)
    if item is None:
        raise HTTPException(status_code=404, detail="API key not found")
    return item


@router.post("/{key_id}/regenerate", response_model=APIKeyGenerateResponse)
async def regenerate_agent_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> APIKeyGenerateResponse:
    """Issue a new secret for an existing key, keeping its id/label/status."""
    try:
        result = AgentApiKeyService(db).regenerate_key(int(current_user.id), key_id)
    except KeyRotationConflict as e:
        logger.warning(f"Concurrent API key regeneration race for key {key_id}: {e}")
        raise HTTPException(status_code=409, detail="rotation_conflict")
    except Exception as e:
        logger.error(f"Failed to regenerate API key {key_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")

    if result is None:
        raise HTTPException(status_code=404, detail="API key not found")

    item, full_key = result
    return APIKeyGenerateResponse(
        full_key=full_key, key_prefix=item.key_prefix, created_at=item.created_at
    )


@router.delete("/{key_id}")
async def delete_agent_api_key(
    key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    deleted = AgentApiKeyService(db).delete_key(int(current_user.id), key_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"deleted": True}
