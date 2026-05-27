"""
Skills API Endpoints

Provides REST API endpoints for managing and using skills in the web application.
"""

import logging
from importlib import import_module
from typing import Optional

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


class RecallRequest(BaseModel):
    """Skill recall request"""

    task: str = Field(..., description="User task description")
    llm_id: str = Field(..., description="LLM ID to use for selection")


class RecallResponse(BaseModel):
    """Skill recall response"""

    skill: Optional[SkillDetail] = Field(None, description="Selected skill or null")
    reasoning: str = Field(..., description="Reasoning for selection")


class ReloadResponse(BaseModel):
    """Skills reload response"""

    message: str = Field(..., description="Status message")
    count: int = Field(..., description="Number of skills loaded")


# ===== Router =====

router = APIRouter(prefix="/api/skills", tags=["skills"])


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
    skill_manager = request.app.state.skill_manager
    skills = await skill_manager.list_skills()
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
    skill_manager = request.app.state.skill_manager
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


@router.post("/recall", response_model=RecallResponse)
async def recall_skill(
    request_data: RecallRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
) -> RecallResponse:
    """
    Select appropriate skill based on task

    This is the core interface: Vibe Planner calls it to get relevant skill

    Args:
        request_data: Recall request with task and llm_id

    Returns:
        Selected skill or null if no relevant skill found
    """
    skill_manager = request.app.state.skill_manager

    # Get LLM from model_manager
    get_model_manager = getattr(
        import_module("xagent.core.model.manager"),
        "get_model_manager",
    )
    model_manager = get_model_manager()
    llm = model_manager.get_llm(request_data.llm_id)

    skill = await skill_manager.select_skill(task=request_data.task, llm=llm)

    if not skill:
        return RecallResponse(skill=None, reasoning="No relevant skill found")

    return RecallResponse(
        skill=SkillDetail(
            name=skill["name"],
            description=skill.get("description", ""),
            when_to_use=skill.get("when_to_use", ""),
            tags=skill.get("tags", []),
            content=skill.get("content", ""),
            execution_flow=skill.get("execution_flow", ""),
            files=skill.get("files", []),
            path=skill["path"],
        ),
        reasoning=f"Selected '{skill['name']}' based on task relevance",
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
    skill_manager = request.app.state.skill_manager
    await skill_manager.reload()

    return ReloadResponse(
        message="Skills reloaded", count=len(await skill_manager.list_skills())
    )
