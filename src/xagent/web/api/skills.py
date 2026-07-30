"""
Skills API Endpoints

Provides REST API endpoints for managing and using skills in the web application.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...skills.library import SkillScopeContext
from ..models.database import get_db
from ..services.skill_runtime import (
    get_skill_runtime_scope,
    handoff_skill_runtime_session,
)

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


async def _request_skill_manager(
    context: SkillScopeContext,
    db: Session,
) -> Any:
    from ...skills.utils import create_skill_manager

    handoff_skill_runtime_session(db)
    manager: Any = create_skill_manager(context=context)
    await manager.ensure_initialized()
    return manager


# ===== Endpoints =====


@router.get("/", response_model=list[SkillInfo])
async def list_skills(
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Session = Depends(get_db),
) -> list[SkillInfo]:
    """
    List all available skills

    Returns:
        List of available skills with basic information
    """
    skill_manager = await _request_skill_manager(context, db)
    skills = await skill_manager.list_skills()
    # Convert to SkillInfo type
    from typing import cast

    return cast(list[SkillInfo], skills)


@router.get("/{skill_name}", response_model=SkillDetail)
async def get_skill(
    skill_name: str,
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Session = Depends(get_db),
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
    skill_manager = await _request_skill_manager(context, db)
    skill = await skill_manager.get_skill(skill_name)

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
    context: SkillScopeContext = Depends(get_skill_runtime_scope),
    db: Session = Depends(get_db),
) -> ReloadResponse:
    """
    Manually reload all skills

    Rescans the skills directory and reloads all skills.

    Returns:
        Reload status with skill count
    """
    skill_manager = await _request_skill_manager(context, db)
    await skill_manager.reload()
    count = len(await skill_manager.list_skills())

    return ReloadResponse(message="Skills reloaded", count=count)
