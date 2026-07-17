"""
Skills API Endpoints

Provides REST API endpoints for managing and using skills in the web application.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..auth_dependencies import get_current_user
from ..models.user import User

logger = logging.getLogger(__name__)


# ===== Pydantic Models =====


class SkillInfo(BaseModel):
    """Skill brief information"""

    name: str = Field(..., description="Skill name")
    description: str = Field(..., description="Skill description")
    when_to_use: str = Field(..., description="When to use this skill")
    tags: list[str] = Field(default_factory=list, description="Skill tags")


class SkillDetail(SkillInfo):
    """Skill complete information"""

    content: str = Field(..., description="Complete SKILL.md content")
    execution_flow: str = Field(..., description="Execution flow")
    files: list[str] = Field(
        default_factory=list, description="Files in skill directory"
    )
    path: str = Field(..., description="Skill directory path")


class ReloadResponse(BaseModel):
    """Skills reload response"""

    message: str = Field(..., description="Status message")
    count: int = Field(..., description="Number of skills loaded")


# ===== Router =====

router = APIRouter(prefix="/api/skills", tags=["skills"])


def _skill_context(
    current_user: User, request: Request, db: object | None = None
) -> Any:
    from ...skills.library import SkillScopeContext

    return SkillScopeContext(
        user=current_user,
        user_id=int(current_user.id) if current_user.id is not None else None,
        db=db,
        request=request,
    )


async def _request_skill_manager(request: Request, current_user: User) -> Any:
    from ...skills.utils import create_skill_manager
    from ..models.database import get_session_local

    db = get_session_local()()
    manager: Any = create_skill_manager(
        context=_skill_context(current_user, request, db)
    )
    manager._scope_db_session = db  # noqa: SLF001 - closed by route helper
    await manager.ensure_initialized()
    return manager


def _close_skill_manager(manager: object) -> None:
    db = getattr(manager, "_scope_db_session", None)
    if db is not None:
        db.close()


# ===== Endpoints =====


@router.get("/", response_model=list[SkillInfo])
async def list_skills(
    request: Request, current_user: User = Depends(get_current_user)
) -> list[SkillInfo]:
    """
    List all available skills

    Returns:
        List of available skills with basic information
    """
    skill_manager = await _request_skill_manager(request, current_user)
    try:
        skills = await skill_manager.list_skills()
    finally:
        _close_skill_manager(skill_manager)
    # Convert to SkillInfo type
    from typing import cast

    return cast(list[SkillInfo], skills)


@router.get("/{skill_name}", response_model=SkillDetail)
async def get_skill(
    skill_name: str,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> SkillDetail:
    """
    Get single skill detail (including template)

    Args:
        skill_name: Name of the skill to retrieve

    Returns:
        Detailed skill information including template

    Raises:
        HTTPException: If skill not found
    """
    skill_manager = await _request_skill_manager(request, current_user)
    try:
        skill = await skill_manager.get_skill(skill_name)
    finally:
        _close_skill_manager(skill_manager)

    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")

    return SkillDetail(
        name=skill["name"],
        description=skill.get("description", ""),
        when_to_use=skill.get("when_to_use", ""),
        tags=skill.get("tags", []),
        content=skill.get("content", ""),
        execution_flow=skill.get("execution_flow", ""),
        files=skill.get("files", []),
        path=skill["path"],
    )


@router.post("/reload", response_model=ReloadResponse)
async def reload_skills(
    request: Request, current_user: User = Depends(get_current_user)
) -> ReloadResponse:
    """
    Manually reload all skills

    Rescans the skills directory and reloads all skills.

    Returns:
        Reload status with skill count
    """
    skill_manager = await _request_skill_manager(request, current_user)
    try:
        await skill_manager.reload()
        count = len(await skill_manager.list_skills())
    finally:
        _close_skill_manager(skill_manager)

    return ReloadResponse(message="Skills reloaded", count=count)
