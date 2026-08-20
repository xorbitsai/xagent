"""
Templates API Endpoints

Provides REST API endpoints for managing and using agent templates.
"""

import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.agent import Agent, AgentOrigin
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


class PersonaInfo(BaseModel):
    """AI Team Marketplace card content for a template: the display name,
    avatar, and the chat-opening intro/questions a Hire flow seeds. Absent
    for templates with no marketplace persona (e.g. a workforce-type
    template, whose card is rendered from `agent_count`/`workforce_config`
    instead).
    """

    name: str = Field(..., description="Display name, e.g. 'Maya'")
    role: str = Field(
        ..., description="Localized job-title line, e.g. 'Social Media Content Manager'"
    )
    avatar: Optional[str] = Field(
        default=None,
        description="Path to the avatar image, e.g. '/marketplace/avatars/maya.png' - "
        "an app-relative path resolved against the frontend origin, not a "
        "standalone URL usable from any context",
    )
    intro: str = Field(
        default="",
        description="Localized opening message sent after the agent is hired",
    )
    kickoff_questions: list[str] = Field(
        default_factory=list,
        description="Localized setup questions the *agent* asks the *user* "
        "right after the intro (unlike sample_prompts / "
        "Agent.suggested_prompts, which are prompts for the user to send)",
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
    persona: Optional[PersonaInfo] = Field(
        default=None, description="AI Team Marketplace card content, if any"
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
    type: Literal["agent", "workforce"] = Field(
        default="agent",
        description="'agent' for a single-agent template, 'workforce' for a "
        "manager + worker-agents template",
    )
    agent_count: int = Field(
        default=0,
        description="Total number of agents (manager + workers) a "
        "'workforce'-type template creates. 0 for 'agent'-type templates.",
    )
    hired: bool = Field(
        default=False,
        description="Whether the current user already has a quick-access "
        "agent instance of this template (see POST /from-template/resolve "
        "on the agents API)",
    )
    hired_agent_id: Optional[int] = Field(
        default=None,
        description="ID of the current user's quick-access agent instance "
        "of this template, if `hired` is true",
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


def build_persona_info(
    template: dict[str, Any], lang: Optional[str] = None
) -> Optional[PersonaInfo]:
    """Resolve a template's raw (locale-keyed) `persona` dict into the
    localized `PersonaInfo` a marketplace card/detail page renders. `None`
    when the template has no persona - see `PersonaInfo`'s docstring.
    """
    persona = template.get("persona")
    if not persona:
        return None

    return PersonaInfo(
        name=persona["name"],
        # get_localized_value already returns `default` for a None input,
        # so a bare .get(key) (no {}/[] fallback) is enough here.
        role=get_localized_value(persona.get("role"), lang, ""),
        avatar=persona.get("avatar"),
        intro=get_localized_value(persona.get("intro"), lang, ""),
        kickoff_questions=get_localized_value(
            persona.get("kickoff_questions"), lang, []
        ),
    )


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


def get_hired_agent_map(
    db: Session, user_id: int, template_ids: list[str]
) -> dict[str, int]:
    """Map each template_id the user has a quick-access agent for to that
    agent's id, scoped to `AgentOrigin.TEMPLATE_QUICK_ACCESS` (the "Hire"
    flow's resolve-created agent) so a workforce-builder instance created
    from the same template under a user-chosen name doesn't count - the
    same origin scoping `resolve_agent_from_template`'s reuse query uses.
    At most one row per (user_id, template_id) can exist here:
    `AGENT_TEMPLATE_QUICK_ACCESS_UNIQUE_INDEX` enforces that pair unique at
    the database level for this origin, so there is no tie to break.

    The contract is deliberately "hired == the agent Hire would return",
    i.e. this query's filters stay identical to
    `_resolve_agent_from_template_sync`'s reuse query (that query also
    orders by id and takes the first row - a tie-break the unique index
    above makes moot, so it is not duplicated here), which has two
    intended consequences (PR #1498 round-2 review, M1/N1):

    - Instantiating a workforce also creates its worker agents with this
      origin plus each worker's own template_id
      (`workforce_creator._get_or_create_quick_access_worker_agent`), so
      the member templates report hired afterward. That is accurate, not
      a false positive: clicking Hire on such a template resolves to
      exactly that worker agent - reporting "not hired" would promise a
      fresh agent the resolve flow will never create.
    - No `status` filter: the resolve flow deliberately returns a found
      agent as-is whatever its status - DRAFT and ARCHIVED alike - rather
      than auto-publishing it (see the B3 comment in
      `_resolve_agent_from_template_sync`), so any quick-access agent
      is still "what Hire returns" and counts as hired. A consumer that
      needs publishability should read the agent's own status rather
      than have this map silently desync from the resolve query.
    """
    if not template_ids:
        return {}

    rows = (
        db.query(Agent.template_id, Agent.id)
        .filter(
            Agent.user_id == user_id,
            Agent.template_id.in_(template_ids),
            Agent.origin == AgentOrigin.TEMPLATE_QUICK_ACCESS.value,
        )
        .all()
    )
    return {template_id: agent_id for template_id, agent_id in rows}


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


def increment_template_used_count(db: Session, template_id: str) -> None:
    """Increment template used_count atomically in the database - the same
    UPDATE-in-place pattern as `increment_template_likes` above. The
    previous `stats.used_count += 1` ORM read-modify-write lost increments
    under concurrent successes on the same template (two concurrent
    `use-as-workforce` calls could both read used_count=0 and both write 1,
    losing one), and its `TemplateStats` row-creation race was swallowed by
    a broad `except IntegrityError` with no compensating re-read. Callers
    must `db.commit()` then `db.refresh(stats)` to see the post-increment
    value, matching `increment_template_likes`'s existing callers.
    """
    db.query(TemplateStats).filter(TemplateStats.template_id == template_id).update(
        {TemplateStats.used_count: TemplateStats.used_count + 1},
        synchronize_session=False,
    )


def get_workforce_agent_count(template: dict[str, Any]) -> int:
    """Total agents (1 manager + N workers) a workforce-type template
    creates - the card badge's only need, computed server-side rather than
    shipping name lists the frontend would just count (this replaced a
    manager_name/worker_names pair whose only consumer read
    lengths/presence). 0 for agent-type templates.
    Derived entirely from static YAML fields (no per-agent template
    lookups), since this runs for every template on every
    `GET /api/templates/` page load.
    """
    if template.get("type") != "workforce":
        # An agent-type template carrying a stray workforce_config block
        # (unvalidated for that type) must not advertise a count.
        return 0
    workforce_config = template.get("workforce_config")
    if not isinstance(workforce_config, dict):
        return 0
    workers = workforce_config.get("agents")
    worker_count = len(workers) if isinstance(workers, list) else 0
    return 1 + worker_count


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
    hired_agent_id_by_template_id = get_hired_agent_map(
        db, current_user_id, template_ids
    )

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
        hired_agent_id = hired_agent_id_by_template_id.get(template_id)

        result.append(
            TemplateInfo(
                id=template["id"],
                name=template["name"],
                category=template.get("category", ""),
                featured=bool(template.get("featured", False)),
                description=description,
                features=features,
                sample_prompts=sample_prompts,
                persona=build_persona_info(template, lang),
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
                agent_count=get_workforce_agent_count(template),
                hired=hired_agent_id is not None,
                hired_agent_id=hired_agent_id,
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
    current_user_id = int(current_user.id)
    hired_agent_id = get_hired_agent_map(db, current_user_id, [template_id]).get(
        template_id
    )

    return TemplateDetail(
        id=template["id"],
        name=template["name"],
        category=template.get("category", ""),
        featured=bool(template.get("featured", False)),
        description=description,
        features=features,
        sample_prompts=sample_prompts,
        persona=build_persona_info(template, lang),
        connections=connections,
        setup_time=setup_time,
        tags=tags,
        author=template.get("author", ""),
        version=template.get("version", ""),
        views=stats.views,
        likes=stats.likes,
        used_count=stats.used_count,
        is_liked=is_template_liked(db, current_user_id, template_id),
        type=template.get("type", "agent"),
        agent_count=get_workforce_agent_count(template),
        hired=hired_agent_id is not None,
        hired_agent_id=hired_agent_id,
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
    if template.get("type") == "workforce":
        # Every other agent-creation surface (the service-layer raises,
        # the v1 400, the home/task/builder filters) already refuses a
        # workforce template - this legacy endpoint was left ungated,
        # returning 200 "Template usage recorded" while creating nothing.
        # It predates the workforce type and frontend code only calls
        # /use-as-workforce for one, but external API consumers could
        # still hit it directly.
        raise HTTPException(
            status_code=400,
            detail="This template creates a workforce, not a single agent; "
            "use POST /{template_id}/use-as-workforce instead",
        )

    stats = get_or_create_template_stats(db, template_id)
    increment_template_used_count(db, template_id)
    db.commit()
    db.refresh(stats)

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
        get_or_create_template_stats(db, template_id)
        increment_template_used_count(db, template_id)
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
