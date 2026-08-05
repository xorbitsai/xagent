"""
Templates API Endpoints

Provides REST API endpoints for managing and using agent templates.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.template_stats import TemplateStats, UserTemplateRelation
from ..models.user import User
from ..services.workforce_creator import create_workforce_from_template

logger = logging.getLogger(__name__)

TEMPLATE_RELATION_LIKE = "like"


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
        default="balanced",
        description="Execution mode: flash, balanced, think, or auto",
    )


class ConnectionInfo(BaseModel):
    """Information about a connection (e.g. MCP app)"""

    name: str = Field(..., description="Name of the connection")
    logo: Optional[str] = Field(default=None, description="URL to the logo image")


class SamplePrompt(BaseModel):
    """A quick-access sample prompt shown on a template card"""

    title: str = Field(..., description="Short label shown on the template card")
    prompt: str = Field(..., description="Prompt text to fill into the chat input")
    highlights: list[str] = Field(
        default_factory=list,
        description="Substrings within prompt to visually highlight as placeholders",
    )


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
    sample_prompts: list[SamplePrompt] = Field(
        default_factory=list, description="Quick-access sample prompts"
    )
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
    is_liked: bool = Field(
        default=False, description="Whether the current user liked this template"
    )
    type: str = Field(
        default="agent",
        description="'agent' for a single-agent template, 'workforce' for a "
        "manager + worker-agents template",
    )
    manager_name: Optional[str] = Field(
        default=None,
        description="Display name of the manager agent a 'workforce'-type "
        "template creates. None for 'agent'-type templates.",
    )
    worker_names: list[str] = Field(
        default_factory=list,
        description="Display names of the worker agents a 'workforce'-type "
        "template creates. Empty for 'agent'-type templates.",
    )


class TemplateDetail(TemplateInfo):
    """Detailed template response including agent configuration"""

    agent_config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Agent configuration for an 'agent'-type template. None "
        "for a 'workforce'-type template, which is configured via "
        "`workforce_config` instead.",
    )
    workforce_config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Manager + worker agent configuration for a 'workforce'-type "
        "template. None for 'agent'-type templates.",
    )


class LikeResponse(BaseModel):
    """Like/unlike response"""

    liked: bool = Field(..., description="Whether the template is liked")
    likes: int = Field(..., description="Total number of likes")


class UseAsWorkforceResponse(BaseModel):
    """Response for POST /{template_id}/use-as-workforce"""

    message: str = Field(..., description="Human-readable confirmation")
    template_id: str = Field(..., description="ID of the template that was used")
    workforce_id: int = Field(..., description="ID of the newly created workforce")


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


def get_or_create_template_stats_map(
    db: Session, template_ids: list[str]
) -> dict[str, TemplateStats]:
    """Get or create template stats records for a list of template IDs."""
    unique_template_ids = list(dict.fromkeys(template_ids))
    if not unique_template_ids:
        return {}

    stats_list = (
        db.query(TemplateStats)
        .filter(TemplateStats.template_id.in_(unique_template_ids))
        .all()
    )
    stats_by_template_id = {stats.template_id: stats for stats in stats_list}
    missing_template_ids = [
        template_id
        for template_id in unique_template_ids
        if template_id not in stats_by_template_id
    ]

    if missing_template_ids:
        new_stats = [
            TemplateStats(template_id=template_id)
            for template_id in missing_template_ids
        ]
        db.add_all(new_stats)
        db.commit()
        for stats in new_stats:
            db.refresh(stats)
            stats_by_template_id[stats.template_id] = stats

    return stats_by_template_id


def get_liked_template_ids(
    db: Session, user_id: int, template_ids: list[str]
) -> set[str]:
    """Return template IDs liked by the given user."""
    if not template_ids:
        return set()

    rows = (
        db.query(UserTemplateRelation.template_id)
        .filter(
            UserTemplateRelation.user_id == user_id,
            UserTemplateRelation.template_id.in_(template_ids),
            UserTemplateRelation.relation_type == TEMPLATE_RELATION_LIKE,
            UserTemplateRelation.is_active.is_(True),
        )
        .all()
    )
    return {row[0] for row in rows}


def is_template_liked(db: Session, user_id: int, template_id: str) -> bool:
    """Return whether the current user has an active like relation."""
    return (
        db.query(UserTemplateRelation.id)
        .filter(
            UserTemplateRelation.user_id == user_id,
            UserTemplateRelation.template_id == template_id,
            UserTemplateRelation.relation_type == TEMPLATE_RELATION_LIKE,
            UserTemplateRelation.is_active.is_(True),
        )
        .first()
        is not None
    )


def increment_template_likes(db: Session, template_id: str) -> None:
    """Increment template likes atomically in the database."""
    db.query(TemplateStats).filter(TemplateStats.template_id == template_id).update(
        {TemplateStats.likes: TemplateStats.likes + 1},
        synchronize_session=False,
    )


def get_workforce_agent_names(
    template: dict[str, Any],
) -> tuple[Optional[str], list[str]]:
    """(manager_name, worker_names) for a workforce-type template. Split
    into two fields rather than one flat list so "which one is the manager"
    is structural, not "index 0 by convention" - and a worker with no name
    would previously be silently dropped from a single list rather than
    visibly changing the count (PR #1127 review; template loading now
    rejects a missing agents[].name outright, but this stays defensive).
    Derived entirely from static YAML fields (no per-agent template
    lookups), since this runs for every template on every
    `GET /api/templates/` page load.
    """
    workforce_config = template.get("workforce_config")
    if not isinstance(workforce_config, dict):
        return None, []

    manager = workforce_config.get("manager")
    manager_name = (
        str(manager["name"])
        if isinstance(manager, dict) and manager.get("name")
        else None
    )
    worker_names = [
        str(agent["name"])
        for agent in workforce_config.get("agents") or []
        if isinstance(agent, dict) and agent.get("name")
    ]
    return manager_name, worker_names


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
    template_ids = [template["id"] for template in templates]
    current_user_id = int(current_user.id)
    liked_template_ids = get_liked_template_ids(db, current_user_id, template_ids)
    stats_by_template_id = get_or_create_template_stats_map(db, template_ids)

    # Get statistics from database
    result = []
    for template in templates:
        template_id = template["id"]
        stats = stats_by_template_id[template_id]

        # Get localized values
        description = get_localized_value(template.get("descriptions", {}), lang, "")
        features = get_localized_value(template.get("features", {}), lang, [])
        sample_prompts = get_localized_value(
            template.get("sample_prompts", {}), lang, []
        )
        setup_time = get_localized_value(
            template.get("setup_time", {}), lang, "5 min setup"
        )
        connections = template.get("connections", [])
        tags = get_localized_value(template.get("tags", {}), lang, [])
        manager_name, worker_names = get_workforce_agent_names(template)

        result.append(
            TemplateInfo(
                id=template["id"],
                name=template["name"],
                category=template.get("category", ""),
                featured=bool(template.get("featured", False)),
                description=description,
                features=features,
                sample_prompts=sample_prompts,
                connections=connections,
                setup_time=setup_time,
                tags=tags,
                author=template.get("author", ""),
                version=template.get("version", ""),
                views=stats.views,
                likes=stats.likes,
                used_count=stats.used_count,
                is_liked=template_id in liked_template_ids,
                type=template.get("type", "agent"),
                manager_name=manager_name,
                worker_names=worker_names,
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
    sample_prompts = get_localized_value(template.get("sample_prompts", {}), lang, [])
    setup_time = get_localized_value(
        template.get("setup_time", {}), lang, "5 min setup"
    )
    connections = template.get("connections", [])
    tags = get_localized_value(template.get("tags", {}), lang, [])
    manager_name, worker_names = get_workforce_agent_names(template)

    return TemplateDetail(
        id=template["id"],
        name=template["name"],
        category=template.get("category", ""),
        featured=bool(template.get("featured", False)),
        description=description,
        features=features,
        sample_prompts=sample_prompts,
        connections=connections,
        setup_time=setup_time,
        tags=tags,
        author=template.get("author", ""),
        version=template.get("version", ""),
        views=stats.views,
        likes=stats.likes,
        used_count=stats.used_count,
        is_liked=is_template_liked(db, int(current_user.id), template_id),
        type=template.get("type", "agent"),
        manager_name=manager_name,
        worker_names=worker_names,
        agent_config=(
            {
                "instructions": template["agent_config"].get("instructions", ""),
                "skills": template["agent_config"].get("skills", []),
                "tool_categories": template["agent_config"].get("tool_categories", []),
                "execution_mode": template["agent_config"].get(
                    "execution_mode", "balanced"
                ),
            }
            if template.get("type", "agent") == "agent"
            else None
        ),
        workforce_config=template.get("workforce_config"),
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
    current_user_id = int(current_user.id)

    relation = (
        db.query(UserTemplateRelation)
        .filter(
            UserTemplateRelation.user_id == current_user_id,
            UserTemplateRelation.template_id == template_id,
            UserTemplateRelation.relation_type == TEMPLATE_RELATION_LIKE,
        )
        .first()
    )

    if relation and relation.is_active:
        return LikeResponse(liked=True, likes=stats.likes)

    try:
        if relation:
            relation.is_active = True
        else:
            relation = UserTemplateRelation(
                user_id=current_user_id,
                template_id=template_id,
                relation_type=TEMPLATE_RELATION_LIKE,
                is_active=True,
            )
            db.add(relation)

        increment_template_likes(db, template_id)
        db.commit()
        db.refresh(stats)
    except IntegrityError:
        db.rollback()
        logger.info(
            "Duplicate like relation detected for user_id=%s template_id=%s",
            current_user_id,
            template_id,
        )
        stats = get_or_create_template_stats(db, template_id)

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


@router.post("/{template_id}/use-as-workforce", response_model=UseAsWorkforceResponse)
async def use_template_as_workforce(
    template_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    lang: Optional[str] = Query(None, description="Language code (e.g., 'en', 'zh')"),
) -> UseAsWorkforceResponse:
    """
    Instantiate a 'workforce'-type template: creates the manager agent plus
    one worker agent per template.workforce_config.agents entry (reusing an
    existing quick-access instance per-worker-template if the user already
    has one), assembles them into a new Workforce, and records usage.

    Args:
        template_id: ID of the workforce template to instantiate
        lang: Optional language code for the new Workforce's description

    Returns:
        The new workforce's id, for the frontend to navigate to its canvas

    Raises:
        HTTPException: If template not found or is not a workforce template
    """
    template_manager = request.app.state.template_manager
    template = await template_manager.get_template(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    if template.get("type") != "workforce":
        raise HTTPException(
            status_code=400, detail="Template is not a workforce template"
        )

    workforce = await create_workforce_from_template(
        db, current_user, template_manager, template, lang=lang
    )

    # The workforce (and its manager/worker agents) is already committed at
    # this point. This is purely an analytics counter, so a failure here
    # must be logged, not surfaced as a request failure - the caller would
    # otherwise see a 500 for an operation that actually already succeeded,
    # with no way to discover the workforce it already created.
    try:
        stats = get_or_create_template_stats(db, template_id)
        stats.used_count += 1
        db.commit()
    except Exception:
        db.rollback()
        logger.exception(
            "Failed to record used_count for template_id=%s after "
            "successfully creating workforce_id=%s",
            template_id,
            workforce.id,
        )

    return UseAsWorkforceResponse(
        message="Workforce created from template",
        template_id=template_id,
        workforce_id=workforce.id,
    )
