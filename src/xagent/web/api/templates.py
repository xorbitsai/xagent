"""
Templates API Endpoints

Provides REST API endpoints for managing and using agent templates.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.template_stats import TemplateStats
from ..models.user import User

logger = logging.getLogger(__name__)


# ===== Helper Functions =====


def get_localized_value(
    values: Any, lang: Optional[str] = None, default: Any = None
) -> Any:
    """
    Get localized values based on language preference

    Args:
        values: The values (can be a dict {en: "...", zh: "..."} or direct string/list)
        lang: Language code, if None attempts to fallback to English
        default: Default value

    Returns:
        Localized values
    """
    if values is None:
        return default

    if isinstance(values, dict):
        if lang and lang in values:
            return values[lang]
        return values.get("en", default)

    # If not a dictionary, return the original value directly
    return values


# ===== Pydantic Models =====


class AgentConfig(BaseModel):
    """Agent configuration from template"""

    instructions: str = Field(..., description="System prompt/instructions")
    skills: list[str] = Field(default_factory=list, description="List of skill names")
    tool_categories: list[str] = Field(
        default_factory=list, description="List of tool categories"
    )
    execution_mode: str = Field(
        default="react", description="Execution mode: simple, react, or graph"
    )


class ConnectionInfo(BaseModel):
    """Information about a connection (e.g. MCP app)"""

    name: str = Field(..., description="Name of the connection")
    logo: Optional[str] = Field(default=None, description="URL to the logo image")


class TemplateInfo(BaseModel):
    """Template brief information"""

    id: str = Field(..., description="Template unique identifier")
    name: str = Field(..., description="Template name")
    category: str = Field(..., description="Template category")
    featured: bool = Field(
        default=False, description="Whether the template is featured"
    )
    description: str = Field(..., description="Template description")
    features: list[str] = Field(default_factory=list, description="Template features")
    connections: list[ConnectionInfo] = Field(
        default_factory=list, description="App connections"
    )
    setup_time: str = Field(default="5 min setup", description="Setup time")
    tags: list[str] = Field(default_factory=list, description="Template tags")
    author: str = Field(..., description="Template author")
    version: str = Field(..., description="Template version")
    views: int = Field(default=0, description="Number of views")
    likes: int = Field(default=0, description="Number of likes")
    used_count: int = Field(default=0, description="Number of times used")


class TemplateDetail(TemplateInfo):
    """Detailed template response including agent configuration"""

    agent_config: dict[str, Any] = Field(..., description="Agent configuration")


class LikeResponse(BaseModel):
    """Like/unlike response"""

    liked: bool = Field(..., description="Whether the template is liked")
    likes: int = Field(..., description="Total number of likes")


# ===== Router =====

router = APIRouter(prefix="/api/templates", tags=["templates"])


# ===== Helper Functions =====


def get_or_create_template_stats(db: Session, template_id: str) -> TemplateStats:
    """Get or create template stats record"""
    stats = (
        db.query(TemplateStats).filter(TemplateStats.template_id == template_id).first()
    )
    if not stats:
        stats = TemplateStats(template_id=template_id)
        db.add(stats)
        db.commit()
        db.refresh(stats)
    return stats


# ===== Endpoints =====


@router.get("/", response_model=list[TemplateInfo])
async def list_templates(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    lang: Optional[str] = Query(None, description="Language code (e.g., 'en', 'zh')"),
) -> list[TemplateInfo]:
    """
    List all available templates (including statistics)

    Args:
        lang: Optional language code for localized descriptions

    Returns:
        List of available templates with statistics
    """
    template_manager = request.app.state.template_manager
    templates = await template_manager.list_templates()

    # Get statistics from database
    result = []
    for template in templates:
        template_id = template["id"]
        stats = get_or_create_template_stats(db, template_id)

        # Get localized values
        description = get_localized_value(template.get("descriptions", {}), lang, "")
        features = get_localized_value(template.get("features", {}), lang, [])
        setup_time = get_localized_value(
            template.get("setup_time", {}), lang, "5 min setup"
        )
        connections = template.get("connections", [])
        tags = get_localized_value(template.get("tags", {}), lang, [])

        result.append(
            TemplateInfo(
                id=template["id"],
                name=template["name"],
                category=template.get("category", ""),
                featured=bool(template.get("featured", False)),
                description=description,
                features=features,
                connections=connections,
                setup_time=setup_time,
                tags=tags,
                author=template.get("author", ""),
                version=template.get("version", ""),
                views=stats.views,
                likes=stats.likes,
                used_count=stats.used_count,
            )
        )

    return result


@router.get("/{template_id}", response_model=TemplateDetail)
async def get_template(
    template_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    lang: Optional[str] = Query(None, description="Language code (e.g., 'en', 'zh')"),
) -> TemplateDetail:
    """
    Get details of a single template (including agent_config)

    Args:
        template_id: ID of the template to retrieve
        lang: Optional language code for localized descriptions

    Returns:
        Detailed template information with agent configuration

    Raises:
        HTTPException: If template not found
    """
    template_manager = request.app.state.template_manager
    template = await template_manager.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Get statistics from database
    stats = get_or_create_template_stats(db, template_id)

    # Increment view count
    stats.views += 1
    db.commit()

    # Get localized values
    description = get_localized_value(template.get("descriptions", {}), lang, "")
    features = get_localized_value(template.get("features", {}), lang, [])
    setup_time = get_localized_value(
        template.get("setup_time", {}), lang, "5 min setup"
    )
    connections = template.get("connections", [])
    tags = get_localized_value(template.get("tags", {}), lang, [])

    return TemplateDetail(
        id=template["id"],
        name=template["name"],
        category=template.get("category", ""),
        featured=bool(template.get("featured", False)),
        description=description,
        features=features,
        connections=connections,
        setup_time=setup_time,
        tags=tags,
        author=template.get("author", ""),
        version=template.get("version", ""),
        views=stats.views,
        likes=stats.likes,
        used_count=stats.used_count,
        agent_config={
            "instructions": template["agent_config"].get("instructions", ""),
            "skills": template["agent_config"].get("skills", []),
            "tool_categories": template["agent_config"].get("tool_categories", []),
            "execution_mode": template["agent_config"].get("execution_mode", "react"),
        },
    )


@router.post("/{template_id}/like", response_model=LikeResponse)
async def like_template(
    template_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> LikeResponse:
    """
    Like or unlike a template

    Args:
        template_id: ID of the template to like/unlike

    Returns:
        Current like status and total likes

    Raises:
        HTTPException: If template not found
    """
    template_manager = request.app.state.template_manager
    template = await template_manager.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    stats = get_or_create_template_stats(db, template_id)

    # Simple toggle like (in production, track user-specific likes)
    stats.likes += 1
    db.commit()

    return LikeResponse(liked=True, likes=stats.likes)


@router.post("/{template_id}/use")
async def use_template(
    template_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """
    Use a template to create an agent (records usage count)

    Args:
        template_id: ID of the template to use

    Returns:
        Success message

    Raises:
        HTTPException: If template not found
    """
    template_manager = request.app.state.template_manager
    template = await template_manager.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Increment used count
    stats = get_or_create_template_stats(db, template_id)
    stats.used_count += 1
    db.commit()

    return {
        "message": "Template usage recorded",
        "template_id": template_id,
        "used_count": stats.used_count,
    }
