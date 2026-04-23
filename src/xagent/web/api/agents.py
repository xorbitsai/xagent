"""Agent Builder API endpoints for creating and managing custom AI agents."""

import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...config import get_uploads_dir
from ...core.agent.service import AgentService
from ...core.memory.in_memory import InMemoryMemoryStore
from ...core.utils.type_check import ensure_list
from ..auth_dependencies import get_current_user
from ..models.agent import Agent, AgentStatus
from ..models.database import get_db
from ..models.model import Model as DBModel
from ..models.user import User
from ..services.llm_utils import UserAwareModelStorage
from ..tools.config import WebToolConfig
from ..user_isolated_memory import UserContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


# ===== Pydantic Models =====


class AgentCreateRequest(BaseModel):
    """Request model for creating a new agent."""

    name: str = Field(..., min_length=1, max_length=200, description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    instructions: Optional[str] = Field(None, description="System instructions/prompt")
    execution_mode: Optional[str] = Field(
        "balanced", description="Execution mode: flash, balanced, or think"
    )
    models: Optional[dict] = Field(
        None, description="Model config: {general, small_fast, visual, compact}"
    )
    knowledge_bases: List[str] = Field(
        default_factory=list, description="Knowledge base names"
    )
    skills: List[str] = Field(default_factory=list, description="Skill names")
    tool_categories: List[str] = Field(
        default_factory=list, description="Tool category names"
    )
    suggested_prompts: List[str] = Field(
        default_factory=list, description="Suggested prompt examples for users"
    )
    logo_base64: Optional[str] = Field(
        None, description="Logo image as base64 data URL"
    )


class AgentUpdateRequest(BaseModel):
    """Request model for updating an agent."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    instructions: Optional[str] = None
    execution_mode: Optional[str] = Field(
        None, description="Execution mode: flash, balanced, or think"
    )
    models: Optional[dict] = None
    knowledge_bases: Optional[List[str]] = None
    skills: Optional[List[str]] = None
    tool_categories: Optional[List[str]] = None
    suggested_prompts: Optional[List[str]] = Field(
        None, description="Suggested prompt examples for users"
    )
    logo_base64: Optional[str] = None
    widget_enabled: Optional[bool] = None
    allowed_domains: Optional[List[str]] = None


class AgentResponse(BaseModel):
    """Response model for agent data."""

    id: int
    user_id: int
    name: str
    description: Optional[str]
    instructions: Optional[str]
    execution_mode: str
    models: Optional[dict]
    knowledge_bases: List[str]
    skills: List[str]
    tool_categories: List[str]
    suggested_prompts: List[str]
    logo_url: Optional[str]
    status: str
    published_at: Optional[str]
    created_at: str
    updated_at: str
    widget_enabled: bool
    allowed_domains: List[str]


class AgentListItem(BaseModel):
    """Simplified agent model for list views."""

    id: int
    name: str
    description: Optional[str]
    logo_url: Optional[str]
    status: str
    created_at: str
    updated_at: str
    widget_enabled: bool
    allowed_domains: List[str]


class PublishResponse(BaseModel):
    """Response model for publish/unpublish operations."""

    message: str
    agent: AgentResponse


class OptimizeInstructionsRequest(BaseModel):
    """Request model for optimizing agent instructions."""

    instructions: str = Field(..., description="Draft instructions to optimize")
    model_id: Optional[int] = Field(
        None, description="Model ID to use for optimization"
    )


KNOWLEDGE_TOOL_CATEGORY = "knowledge"

KB_PRIORITY_PROMPT = (
    "\n\n[Knowledge Base Instructions]\n"
    "You have access to the following knowledge base(s). "
    "When answering user questions, you MUST first search the knowledge base(s) "
    "using the available knowledge tools before relying on your own knowledge. "
    "Always prioritize information retrieved from the knowledge base(s) over "
    "your built-in knowledge. If the knowledge base does not contain relevant "
    "information, you may then use your own knowledge to answer, but clearly "
    "indicate that the answer is not from the knowledge base."
)


def enhance_system_prompt_with_kb(
    system_prompt: Optional[str], knowledge_bases: Optional[List[str]]
) -> Optional[str]:
    """Append knowledge-base priority instructions when KBs are configured."""
    if not knowledge_bases:
        return system_prompt

    kb_list = ", ".join(knowledge_bases)
    kb_prompt = (
        f"\n\nAvailable knowledge bases: {kb_list}. "
        "These knowledge bases are already selected. "
        "Do not call list_knowledge_bases to discover them; "
        "use knowledge_search directly for answers."
    )

    if system_prompt:
        return system_prompt + kb_prompt
    return kb_prompt.lstrip("\n")


# ===== Helper Functions =====


def _validate_knowledge_base_tools(
    knowledge_bases: List[str], tool_categories: List[str]
) -> None:
    """Raise HTTPException if knowledge bases are selected without the knowledge tool category."""
    if knowledge_bases and KNOWLEDGE_TOOL_CATEGORY not in tool_categories:
        raise HTTPException(
            status_code=400,
            detail="Knowledge bases are selected but the Knowledge tool category is not enabled. Please enable the Knowledge tools before saving.",
        )


def _save_logo(base64_data: Optional[str], agent_id: int) -> Optional[str]:
    """Save logo image and return URL."""
    if not base64_data:
        return None

    try:
        import base64

        # Parse data URL
        if not base64_data.startswith("data:image"):
            logger.warning(f"Invalid image data URL for agent {agent_id}")
            return None

        # Extract the base64 part
        header, encoded = base64_data.split(",", 1)
        image_data = base64.b64decode(encoded)

        # Determine file extension from data URL
        if "png" in header:
            ext = "png"
        elif "jpeg" in header or "jpg" in header:
            ext = "jpg"
        elif "webp" in header:
            ext = "webp"
        else:
            ext = "png"

        # Create uploads directory if needed
        upload_dir = get_uploads_dir() / "agent_logos"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        filename = f"agent_{agent_id}.{ext}"
        filepath = upload_dir / filename
        with open(filepath, "wb") as f:
            f.write(image_data)

        # Return URL
        return f"/uploads/agent_logos/{filename}"

    except Exception as e:
        logger.error(f"Failed to save logo for agent {agent_id}: {e}")
        return None


def _delete_logo(logo_url: str) -> None:
    """Delete logo file."""
    try:
        if logo_url and logo_url.startswith("/"):
            filepath = logo_url.lstrip("/")
            if os.path.exists(filepath):
                os.remove(filepath)
    except Exception as e:
        logger.error(f"Failed to delete logo {logo_url}: {e}")


def _agent_to_response(agent: Agent, db: Session) -> AgentResponse:
    """Convert Agent model to response."""
    return AgentResponse(
        id=agent.id,
        user_id=agent.user_id,
        name=agent.name,
        description=agent.description,
        instructions=agent.instructions,
        execution_mode=agent.execution_mode or "graph",
        models=agent.models,
        knowledge_bases=ensure_list(agent.knowledge_bases) or [],
        skills=ensure_list(agent.skills) or [],
        tool_categories=ensure_list(agent.tool_categories) or [],
        suggested_prompts=ensure_list(agent.suggested_prompts) or [],
        logo_url=agent.logo_url,
        status=agent.status.value,
        published_at=agent.published_at.isoformat() if agent.published_at else None,
        created_at=agent.created_at.isoformat(),
        updated_at=agent.updated_at.isoformat(),
        widget_enabled=agent.widget_enabled,
        allowed_domains=ensure_list(agent.allowed_domains) or [],
    )


# ===== Endpoints =====


@router.post("/optimize-instructions")
async def optimize_instructions(
    request: OptimizeInstructionsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Optimize agent instructions using an LLM."""
    try:
        # Get model storage
        model_storage = UserAwareModelStorage(db)
        user_id = int(current_user.id)

        # Get LLM (use provided model_id or default)
        llm = None
        if request.model_id:
            llm = model_storage.get_llm_by_id(str(request.model_id), user_id)

        if not llm:
            # Get default LLM
            default_llm, _, _, _ = model_storage.get_configured_defaults(user_id)
            llm = default_llm

        if not llm:
            # Fallback to system default if user has no default
            default_llm, _, _, _ = model_storage.get_configured_defaults(None)
            llm = default_llm

        if not llm:
            raise HTTPException(
                status_code=400, detail="No LLM available for optimization"
            )

        # Construct prompt
        system_prompt = (
            "You are an expert agent builder and prompt engineer. "
            "Your task is to refine and optimize the user's draft instructions for an AI agent. "
            "The output should be clear, structured, and effective for an LLM to follow. "
            "Do not include any conversational filler. Just output the optimized instructions."
        )

        user_prompt = f"Draft instructions:\n{request.instructions}\n\nPlease optimize these instructions."

        # Call LLM
        response = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        if isinstance(response, dict) and "content" in response:
            content = response["content"]
        else:
            content = response if isinstance(response, str) else str(response)

        return {"optimized_instructions": content}

    except Exception as e:
        logger.error(f"Failed to optimize instructions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=AgentResponse)
async def create_agent(
    agent_data: AgentCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentResponse:
    """Create a new custom agent."""
    try:
        # Check for duplicate name
        existing = (
            db.query(Agent)
            .filter(Agent.user_id == current_user.id, Agent.name == agent_data.name)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=400, detail="Agent with this name already exists"
            )

        _validate_knowledge_base_tools(
            agent_data.knowledge_bases, agent_data.tool_categories
        )

        # Create agent
        agent = Agent(
            user_id=current_user.id,
            name=agent_data.name,
            description=agent_data.description,
            instructions=agent_data.instructions,
            execution_mode=agent_data.execution_mode or "graph",
            models=agent_data.models,
            knowledge_bases=agent_data.knowledge_bases,
            skills=agent_data.skills,
            tool_categories=agent_data.tool_categories,
            suggested_prompts=agent_data.suggested_prompts,
            status=AgentStatus.DRAFT,
            widget_enabled=True,
            allowed_domains=[],
        )

        db.add(agent)
        db.commit()
        db.refresh(agent)

        # Save logo if provided
        if agent_data.logo_base64:
            logo_url = _save_logo(agent_data.logo_base64, agent.id)  # type: ignore[arg-type]
            if logo_url:
                agent.logo_url = logo_url  # type: ignore[assignment]
                db.commit()
                db.refresh(agent)

        logger.info(f"Created agent {agent.id} for user {current_user.id}")
        return _agent_to_response(agent, db)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[AgentListItem])
async def list_agents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[AgentListItem]:
    """List all agents for the current user."""
    try:
        agents = (
            db.query(Agent)
            .filter(Agent.user_id == current_user.id)
            .order_by(Agent.created_at.desc())
            .all()
        )

        return [
            AgentListItem(
                id=agent.id,
                name=agent.name,
                description=agent.description,
                logo_url=agent.logo_url,
                status=agent.status.value,
                created_at=agent.created_at.isoformat(),
                updated_at=agent.updated_at.isoformat()
                if agent.updated_at
                else agent.created_at.isoformat(),
                widget_enabled=agent.widget_enabled,
                allowed_domains=agent.allowed_domains or [],
            )
            for agent in agents
        ]

    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentResponse:
    """Get agent details."""
    try:
        agent = (
            db.query(Agent)
            .filter(Agent.id == agent_id, Agent.user_id == current_user.id)
            .first()
        )

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return _agent_to_response(agent, db)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: int,
    agent_data: AgentUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentResponse:
    """Update an existing agent."""
    try:
        agent = (
            db.query(Agent)
            .filter(Agent.id == agent_id, Agent.user_id == current_user.id)
            .first()
        )

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Validate knowledge base + tool category consistency
        effective_kb = (
            agent_data.knowledge_bases
            if agent_data.knowledge_bases is not None
            else (agent.knowledge_bases or [])
        )
        effective_tools = (
            agent_data.tool_categories
            if agent_data.tool_categories is not None
            else (agent.tool_categories or [])
        )
        _validate_knowledge_base_tools(effective_kb, effective_tools)  # type: ignore[arg-type]

        # Update fields
        if agent_data.name is not None:
            # Check for duplicate name (excluding current agent)
            existing = (
                db.query(Agent)
                .filter(
                    Agent.user_id == current_user.id,
                    Agent.name == agent_data.name,
                    Agent.id != agent_id,
                )
                .first()
            )
            if existing:
                raise HTTPException(
                    status_code=400, detail="Agent with this name already exists"
                )
            agent.name = agent_data.name  # type: ignore[assignment]

        if agent_data.description is not None:
            agent.description = agent_data.description  # type: ignore[assignment]
        if agent_data.instructions is not None:
            agent.instructions = agent_data.instructions  # type: ignore[assignment]
        if agent_data.models is not None:
            agent.models = agent_data.models  # type: ignore[assignment]
        if agent_data.knowledge_bases is not None:
            agent.knowledge_bases = agent_data.knowledge_bases  # type: ignore[assignment]
        if agent_data.skills is not None:
            agent.skills = agent_data.skills  # type: ignore[assignment]
        if agent_data.tool_categories is not None:
            agent.tool_categories = agent_data.tool_categories  # type: ignore[assignment]
        if agent_data.execution_mode is not None:
            agent.execution_mode = agent_data.execution_mode  # type: ignore[assignment]
        if agent_data.suggested_prompts is not None:
            agent.suggested_prompts = agent_data.suggested_prompts  # type: ignore[assignment]
        if agent_data.widget_enabled is not None:
            agent.widget_enabled = agent_data.widget_enabled  # type: ignore[assignment]
        if agent_data.allowed_domains is not None:
            agent.allowed_domains = agent_data.allowed_domains  # type: ignore[assignment]

        # Handle logo
        if agent_data.logo_base64 is not None:
            # Delete old logo
            if agent.logo_url:
                _delete_logo(agent.logo_url)  # type: ignore[arg-type]

            # Save new logo
            logo_url = _save_logo(agent_data.logo_base64, agent.id)  # type: ignore[arg-type]
            agent.logo_url = logo_url  # type: ignore[assignment]

        db.commit()
        db.refresh(agent)

        logger.info(f"Updated agent {agent_id} for user {current_user.id}")
        return _agent_to_response(agent, db)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent {agent_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Delete an agent."""
    try:
        agent = (
            db.query(Agent)
            .filter(Agent.id == agent_id, Agent.user_id == current_user.id)
            .first()
        )

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Delete logo if exists
        if agent.logo_url:
            _delete_logo(agent.logo_url)  # type: ignore[arg-type]

        db.delete(agent)
        db.commit()

        logger.info(f"Deleted agent {agent_id} for user {current_user.id}")
        return {"message": "Agent deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/publish", response_model=PublishResponse)
async def publish_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PublishResponse:
    """Publish an agent (make it publicly accessible)."""
    try:
        agent = (
            db.query(Agent)
            .filter(Agent.id == agent_id, Agent.user_id == current_user.id)
            .first()
        )

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.status == AgentStatus.PUBLISHED:
            return PublishResponse(
                message="Agent is already published",
                agent=_agent_to_response(agent, db),
            )

        agent.status = AgentStatus.PUBLISHED
        agent.published_at = datetime.now()  # type: ignore[assignment]
        db.commit()
        db.refresh(agent)

        logger.info(f"Published agent {agent_id} for user {current_user.id}")
        return PublishResponse(
            message="Agent published successfully", agent=_agent_to_response(agent, db)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to publish agent {agent_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/unpublish", response_model=PublishResponse)
async def unpublish_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PublishResponse:
    """Unpublish an agent (revert to draft status)."""
    try:
        agent = (
            db.query(Agent)
            .filter(Agent.id == agent_id, Agent.user_id == current_user.id)
            .first()
        )

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.status != AgentStatus.PUBLISHED:
            return PublishResponse(
                message="Agent is not published", agent=_agent_to_response(agent, db)
            )

        agent.status = AgentStatus.DRAFT
        agent.published_at = None  # type: ignore[assignment]
        db.commit()
        db.refresh(agent)

        logger.info(f"Unpublished agent {agent_id} for user {current_user.id}")
        return PublishResponse(
            message="Agent unpublished successfully",
            agent=_agent_to_response(agent, db),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unpublish agent {agent_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/logo", response_model=dict)
async def upload_agent_logo(
    agent_id: int,
    logo_base64: str = Body(..., description="Logo image as base64 data URL"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Upload or update agent logo."""
    try:
        agent = (
            db.query(Agent)
            .filter(Agent.id == agent_id, Agent.user_id == current_user.id)
            .first()
        )

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Delete old logo
        if agent.logo_url:
            _delete_logo(agent.logo_url)  # type: ignore[arg-type]

        # Save new logo
        logo_url = _save_logo(logo_base64, agent.id)  # type: ignore[arg-type]
        if not logo_url:
            raise HTTPException(status_code=400, detail="Failed to save logo")

        agent.logo_url = logo_url  # type: ignore[assignment]
        db.commit()
        db.refresh(agent)

        logger.info(f"Updated logo for agent {agent_id}")
        return {"logo_url": logo_url}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload logo for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Preview Models =====


class AgentPreviewRequest(BaseModel):
    """Request model for agent preview."""

    instructions: Optional[str] = Field(None, description="System instructions/prompt")
    execution_mode: Optional[str] = Field(
        "balanced", description="Execution mode: flash, balanced, or think"
    )
    models: Optional[dict] = Field(
        None, description="Model config: {general, small_fast, visual, compact}"
    )
    knowledge_bases: List[str] = Field(
        default_factory=list, description="Knowledge base names"
    )
    skills: List[str] = Field(default_factory=list, description="Skill names")
    tool_categories: List[str] = Field(
        default_factory=list, description="Tool category names"
    )
    message: str = Field(..., description="User message to preview")


class AgentPreviewResponse(BaseModel):
    """Response model for agent preview."""

    response: str
    status: str


# ===== Preview Endpoint =====


@router.post("/preview", response_model=AgentPreviewResponse)
async def preview_agent(
    request: AgentPreviewRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentPreviewResponse:
    """Preview agent response without saving to database."""
    try:
        # Resolve LLMs from model IDs
        default_llm = None
        fast_llm = None
        vision_llm = None
        compact_llm = None

        if request.models:
            model_config = request.models
            storage = UserAwareModelStorage(db)

            if model_config.get("general"):
                general_model = (
                    db.query(DBModel)
                    .filter(DBModel.id == model_config["general"])
                    .first()
                )
                if general_model:
                    default_llm = storage.get_llm_by_name_with_access(
                        str(general_model.model_id), int(current_user.id)
                    )
            if model_config.get("small_fast"):
                fast_model = (
                    db.query(DBModel)
                    .filter(DBModel.id == model_config["small_fast"])
                    .first()
                )
                if fast_model:
                    fast_llm = storage.get_llm_by_name_with_access(
                        str(fast_model.model_id), int(current_user.id)
                    )
            if model_config.get("visual"):
                visual_model = (
                    db.query(DBModel)
                    .filter(DBModel.id == model_config["visual"])
                    .first()
                )
                if visual_model:
                    vision_llm = storage.get_llm_by_name_with_access(
                        str(visual_model.model_id), int(current_user.id)
                    )
            if model_config.get("compact"):
                compact_model = (
                    db.query(DBModel)
                    .filter(DBModel.id == model_config["compact"])
                    .first()
                )
                if compact_model:
                    compact_llm = storage.get_llm_by_name_with_access(
                        str(compact_model.model_id), int(current_user.id)
                    )

        if not default_llm:
            raise HTTPException(
                status_code=400, detail="General model is required for preview"
            )

        # Create tool config with allowed collections, skills, and tools
        # WebToolConfig expects db and request, pass a minimal dict-like request object
        class MinimalRequest:
            def __init__(self, user_id: int) -> None:
                self.user = type("obj", (), {"id": user_id})()

        # Generate unique task_id for each preview to avoid workspace conflicts
        preview_task_id = f"preview_{uuid.uuid4().hex[:8]}"

        tool_config = WebToolConfig(
            db=db,
            request=MinimalRequest(int(current_user.id)),
            llm=default_llm,
            user_id=int(current_user.id),
            is_admin=bool(current_user.is_admin),
            allowed_collections=request.knowledge_bases
            if request.knowledge_bases is not None
            else None,
            allowed_skills=request.skills if request.skills is not None else None,
            task_id=preview_task_id,
            workspace_base_dir=str(get_uploads_dir() / "preview"),
        )

        # Determine execution mode (default to "think")
        execution_mode = request.execution_mode or "think"

        # Map execution mode to use_dag_pattern
        # flash: SingleCall pattern (quick tasks)
        # balanced: ReAct pattern (everyday tasks)
        # think: DAG/Graph plan-execute pattern (complex tasks)
        if execution_mode == "think":
            use_dag_pattern = True
        elif execution_mode == "balanced":
            use_dag_pattern = False
        elif execution_mode == "flash":
            use_dag_pattern = False  # SingleCall doesn't use DAG
        else:  # fallback to balanced (react)
            use_dag_pattern = False

        # Create agent service (no tracer - no database logging for preview)
        memory = InMemoryMemoryStore()
        agent_service = AgentService(
            name="preview_agent",
            llm=default_llm,
            fast_llm=fast_llm,
            vision_llm=vision_llm,
            compact_llm=compact_llm,
            memory=memory,
            tool_config=tool_config,
            use_dag_pattern=use_dag_pattern,
            id=preview_task_id,
            enable_workspace=True,  # Both patterns support workspace
            workspace_base_dir=str(get_uploads_dir() / "preview"),
            task_id=preview_task_id,  # Add task_id for proper tool initialization
            tracer=None,  # No tracer for preview - don't log to database
        )

        # Execute task with system prompt in context
        execution_context = {}
        if request.instructions:
            execution_context["system_prompt"] = request.instructions
        execution_context["system_prompt"] = enhance_system_prompt_with_kb(  # type: ignore[assignment]
            execution_context.get("system_prompt"),
            request.knowledge_bases if request.knowledge_bases is not None else None,
        )

        with UserContext(int(current_user.id)):
            result = await agent_service.execute_task(
                task=request.message,
                context=execution_context if execution_context else None,
                task_id=preview_task_id,
            )

        return AgentPreviewResponse(
            response=result.get("output", "No response generated"),
            status=result.get("status", "unknown"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to preview agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))
