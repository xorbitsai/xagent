"""Agent Builder API endpoints for creating and managing custom AI agents."""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import get_agent_pattern_for_execution_mode, get_uploads_dir
from ...core.agent.service import AgentService
from ...core.memory.in_memory import InMemoryMemoryStore
from ...core.tools.core.document_search import find_missing_knowledge_bases
from ...core.tracing import create_agent_tracer
from ...core.utils.api_key import generate_api_key
from ..auth_dependencies import get_current_user
from ..models.agent import Agent
from ..models.agent_api_key import AgentApiKey
from ..models.database import get_db
from ..models.model import Model as DBModel
from ..models.user import User
from ..schemas.agent_api_key import (
    APIKeyGenerateResponse,
    APIKeyMetadataResponse,
    APIKeyRevokeResponse,
)
from ..services.agent_store import AgentStore
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
        "balanced", description="Execution mode: flash, balanced, think, or auto"
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
        None, description="Execution mode: flash, balanced, think, or auto"
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
        "use knowledge_search directly for answers. "
        "For specific how-to or factual questions, start with one targeted "
        "knowledge_search, inspect all returned results as one evidence set, "
        "and answer from that evidence when it is relevant. Search again only "
        "when the returned results as a group are missing the information "
        "needed to answer the current question."
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


async def _validate_knowledge_bases_exist(
    knowledge_bases: List[str], current_user: User
) -> None:
    """Raise HTTPException if any selected knowledge base is not visible to the user."""
    missing = await find_missing_knowledge_bases(
        knowledge_bases,
        user_id=int(current_user.id),
        is_admin=bool(current_user.is_admin),
    )
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                "Knowledge base(s) not found or not visible to this user: "
                + ", ".join(missing)
            ),
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


def _get_owned_agent_or_404(agent_id: int, current_user: User, db: Session) -> Agent:
    """Resolve an agent_id against the caller's ownership, raising 404 otherwise.

    Why 404 instead of 403 when ownership doesn't match:
        Returning 403 ("forbidden") would leak that an agent with this id
        exists, just owned by somebody else. The /v1/* surface design (and
        general best practice for multi-tenant resources) is to fold
        "missing" and "not yours" into the same 404 response so callers
        cannot enumerate other users' agent ids.

    Args:
        agent_id: Path parameter from the route.
        current_user: Authenticated user from ``Depends(get_current_user)``.
        db: SQLAlchemy session.

    Returns:
        The :class:`Agent` row, guaranteed to belong to ``current_user``.

    Raises:
        HTTPException 404: agent does not exist, or exists but belongs to
            another user.
    """
    agent = (
        db.query(Agent)
        .filter(Agent.id == agent_id, Agent.user_id == current_user.id)
        .first()
    )
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def _mask_key(key_prefix: str) -> str:
    """Render the display form ``xag_<prefix>_••••••••`` for read-only views.

    The bullet count is fixed at eight on purpose. The actual secret is 32
    characters; reflecting the real length in the UI would leak length
    metadata in screenshots and screenshares. Eight bullets is short
    enough to render compactly and long enough to read as "redacted".

    Args:
        key_prefix: The public-safe lookup handle (6 chars).

    Returns:
        Display string suitable for the web UI's "API Key" card.
    """
    return f"xag_{key_prefix}_••••••••"


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
        store = AgentStore(db)
        user_id = int(current_user.id)
        # Check for duplicate name
        if store.agent_name_exists(user_id, agent_data.name):
            raise HTTPException(
                status_code=400, detail="Agent with this name already exists"
            )

        _validate_knowledge_base_tools(
            agent_data.knowledge_bases, agent_data.tool_categories
        )
        await _validate_knowledge_bases_exist(agent_data.knowledge_bases, current_user)

        agent = store.create_agent(
            user_id=user_id,
            name=agent_data.name,
            description=agent_data.description,
            instructions=agent_data.instructions,
            execution_mode=agent_data.execution_mode or "graph",
            models=agent_data.models,
            knowledge_bases=agent_data.knowledge_bases,
            skills=agent_data.skills,
            tool_categories=agent_data.tool_categories,
            suggested_prompts=agent_data.suggested_prompts,
        )

        # Save logo if provided
        if agent_data.logo_base64:
            logo_url = _save_logo(agent_data.logo_base64, agent.id)  # type: ignore[arg-type]
            if logo_url:
                agent = (
                    store.update_agent_fields(
                        user_id, int(agent.id), {"logo_url": logo_url}
                    )
                    or agent
                )

        logger.info(f"Created agent {agent.id} for user {current_user.id}")
        return AgentResponse.model_validate(store.agent_to_response_dict(agent))

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
        items = AgentStore(db).list_agent_items(int(current_user.id))
        return [AgentListItem.model_validate(item) for item in items]

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
        response = AgentStore(db).get_agent_response(int(current_user.id), agent_id)
        if response is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return AgentResponse.model_validate(response)

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
        store = AgentStore(db)
        user_id = int(current_user.id)
        agent = store.get_owned_agent(user_id, agent_id)

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
        await _validate_knowledge_bases_exist(effective_kb, current_user)  # type: ignore[arg-type]

        # Update fields
        updates: dict[str, object] = {}
        if agent_data.name is not None:
            # Check for duplicate name (excluding current agent)
            if store.agent_name_exists(
                user_id, agent_data.name, exclude_agent_id=agent_id
            ):
                raise HTTPException(
                    status_code=400, detail="Agent with this name already exists"
                )
            updates["name"] = agent_data.name

        if agent_data.description is not None:
            updates["description"] = agent_data.description
        if agent_data.instructions is not None:
            updates["instructions"] = agent_data.instructions
        if agent_data.models is not None:
            updates["models"] = agent_data.models
        if agent_data.knowledge_bases is not None:
            updates["knowledge_bases"] = agent_data.knowledge_bases
        if agent_data.skills is not None:
            updates["skills"] = agent_data.skills
        if agent_data.tool_categories is not None:
            updates["tool_categories"] = agent_data.tool_categories
        if agent_data.execution_mode is not None:
            updates["execution_mode"] = agent_data.execution_mode
        if agent_data.suggested_prompts is not None:
            updates["suggested_prompts"] = agent_data.suggested_prompts
        if agent_data.widget_enabled is not None:
            updates["widget_enabled"] = agent_data.widget_enabled
        if agent_data.allowed_domains is not None:
            updates["allowed_domains"] = agent_data.allowed_domains

        # Handle logo
        if agent_data.logo_base64 is not None:
            # Delete old logo
            if agent.logo_url:
                _delete_logo(agent.logo_url)  # type: ignore[arg-type]

            # Save new logo
            logo_url = _save_logo(agent_data.logo_base64, agent.id)  # type: ignore[arg-type]
            updates["logo_url"] = logo_url

        if updates:
            agent = store.update_agent_fields(user_id, agent_id, updates) or agent

        logger.info(f"Updated agent {agent_id} for user {current_user.id}")
        return AgentResponse.model_validate(store.agent_to_response_dict(agent))

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
        store = AgentStore(db)
        user_id = int(current_user.id)
        agent = store.get_owned_agent(user_id, agent_id)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Delete logo if exists
        if agent.logo_url:
            _delete_logo(agent.logo_url)  # type: ignore[arg-type]

        store.delete_agent(user_id, agent_id)
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
        store = AgentStore(db)
        agent = store.get_owned_agent(int(current_user.id), agent_id)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.status.value == "published":
            return PublishResponse(
                message="Agent is already published",
                agent=AgentResponse.model_validate(store.agent_to_response_dict(agent)),
            )

        agent = store.publish_agent(int(current_user.id), agent_id) or agent

        logger.info(f"Published agent {agent_id} for user {current_user.id}")
        return PublishResponse(
            message="Agent published successfully",
            agent=AgentResponse.model_validate(store.agent_to_response_dict(agent)),
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
        store = AgentStore(db)
        agent = store.get_owned_agent(int(current_user.id), agent_id)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.status.value != "published":
            return PublishResponse(
                message="Agent is not published",
                agent=AgentResponse.model_validate(store.agent_to_response_dict(agent)),
            )

        agent = store.unpublish_agent(int(current_user.id), agent_id) or agent

        logger.info(f"Unpublished agent {agent_id} for user {current_user.id}")
        return PublishResponse(
            message="Agent unpublished successfully",
            agent=AgentResponse.model_validate(store.agent_to_response_dict(agent)),
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
        store = AgentStore(db)
        user_id = int(current_user.id)
        agent = store.get_owned_agent(user_id, agent_id)

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Delete old logo
        if agent.logo_url:
            _delete_logo(agent.logo_url)  # type: ignore[arg-type]

        # Save new logo
        logo_url = _save_logo(logo_base64, agent.id)  # type: ignore[arg-type]
        if not logo_url:
            raise HTTPException(status_code=400, detail="Failed to save logo")

        store.update_agent_fields(user_id, agent_id, {"logo_url": logo_url})

        logger.info(f"Updated logo for agent {agent_id}")
        return {"logo_url": logo_url}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload logo for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== API Key Endpoints =====
#
# Three sibling endpoints (POST/GET/DELETE) at /api/agents/{agent_id}/api-key
# let the agent owner manage the SDK key. All three share JWT auth via
# ``get_current_user`` and gate ownership through ``_get_owned_agent_or_404``;
# the unsuccessful-ownership path returns 404 (not 403) so the existence of
# another user's agent is not leaked. See the SDK design doc §5 for the
# product-level contract and §10 for the security rationale.


@router.post("/{agent_id}/api-key", response_model=APIKeyGenerateResponse)
async def generate_agent_api_key(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> APIKeyGenerateResponse:
    """Generate or rotate the SDK API key for an agent.

    If an active (non-revoked) key already exists for the agent, this
    endpoint revokes it and inserts a new active row in a single
    transaction. The new ``full_key`` is returned exactly once in the
    response; the plaintext secret is never persisted server-side, only
    its bcrypt hash.

    Args:
        agent_id: Path parameter; the target agent's primary key.
        current_user: Resolved from the ``Authorization: Bearer <JWT>``
            header by ``get_current_user``.
        db: SQLAlchemy session injected by FastAPI.

    Returns:
        :class:`APIKeyGenerateResponse` containing ``full_key`` (one-shot
        plaintext), ``key_prefix``, and ``created_at``.

    Raises:
        HTTPException 401: missing or invalid JWT.
        HTTPException 404: agent does not exist or does not belong to the
            caller (deliberate to avoid leaking agent existence).
        HTTPException 500: any unexpected error; transaction rolled back.
            The most plausible internal failure is a partial-unique index
            violation from a concurrent POST race, which the DB enforces.

    Notes:
        - Transactional shape mirrors ``auth.setup_admin`` and
          ``custom_api.create_custom_api`` -- we collect all writes in the
          session and commit once. There is no ``SELECT ... FOR UPDATE``;
          concurrent rotations are caught by the
          ``uq_agent_api_keys_agent_active`` partial unique index and
          surfaced as a 500. Two clients racing to rotate the same key is
          a corner case; a 500 is acceptable.
        - Logs include the ``key_prefix`` only -- never the ``full_key``,
          the secret half, or the bcrypt hash.
    """
    try:
        # Ownership gate. Raises 404 on miss; never reveals "exists but
        # not yours" vs "does not exist".
        _get_owned_agent_or_404(agent_id, current_user, db)

        # Revoke any existing active key for this agent. We touch
        # ``updated_at`` so audit queries can see the rotation moment on
        # the old row as well as the new row.
        now = datetime.now(timezone.utc)
        existing = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
            )
            .first()
        )
        if existing is not None:
            existing.revoked_at = now  # type: ignore[assignment]
            existing.updated_at = now  # type: ignore[assignment]

        # Generate a fresh prefix+secret+hash. ``generate_api_key`` does
        # its own prefix-collision probe against ``agent_api_keys`` so
        # we don't have to.
        full_key, key_prefix, key_hash = generate_api_key(db)

        new_row = AgentApiKey(
            agent_id=agent_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
        )
        db.add(new_row)

        # Single commit: revoke + insert are atomic. If a concurrent
        # POST snuck in between our SELECT and INSERT, the partial
        # unique index raises IntegrityError here and the outer except
        # rolls back. The losing client sees 500, which the UI can retry.
        db.commit()
        db.refresh(new_row)

        # ``key_prefix`` is the only safe field to log. Do NOT log
        # full_key / secret / key_hash even in DEBUG.
        logger.info(
            f"Generated API key for agent {agent_id} "
            f"(prefix={key_prefix}, rotated={existing is not None})"
        )

        return APIKeyGenerateResponse(
            full_key=full_key,
            key_prefix=key_prefix,
            created_at=new_row.created_at,
        )

    except HTTPException:
        raise
    except IntegrityError as e:
        # Partial unique constraint hit -- another POST won the race
        # between our SELECT and our COMMIT. Surface this as 409 rather
        # than a generic 500 so the client can retry without alarm.
        # Internal SQL message stays in the log only.
        db.rollback()
        logger.warning(f"Concurrent API key rotation race for agent {agent_id}: {e}")
        raise HTTPException(status_code=409, detail="rotation_conflict")
    except Exception as e:
        # Sanitize: do NOT echo str(e) to the client -- it could leak
        # internal table names, SQL error wording, or storage backend
        # identity. Full diagnostic stays in the server log.
        logger.error(f"Failed to generate API key for agent {agent_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{agent_id}/api-key", response_model=APIKeyMetadataResponse)
async def get_agent_api_key(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> APIKeyMetadataResponse:
    """Return metadata for the agent's currently active API key.

    Returns the public-safe prefix and a display-only ``masked_key``.
    The plaintext secret is unrecoverable by design -- if the owner has
    lost it, they must POST to rotate.

    Args:
        agent_id: Path parameter.
        current_user: Resolved from JWT.
        db: SQLAlchemy session.

    Returns:
        :class:`APIKeyMetadataResponse` with ``key_prefix``, ``masked_key``,
        and ``created_at``.

    Raises:
        HTTPException 401: missing or invalid JWT.
        HTTPException 404: agent missing / not owned; or owned but has no
            active key. Both shapes use the same status code so the
            caller cannot distinguish "agent doesn't exist" from "no key
            generated yet". The ``detail`` differentiates so the UI can
            render "未生成" instead of "agent not found".
    """
    try:
        _get_owned_agent_or_404(agent_id, current_user, db)

        row = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
            )
            .first()
        )
        if row is None:
            # "Has the owner generated a key yet?" answered with 404 so
            # the UI catches and renders the empty state.
            raise HTTPException(status_code=404, detail="no_active_key")

        return APIKeyMetadataResponse(
            key_prefix=row.key_prefix,
            masked_key=_mask_key(row.key_prefix),  # type: ignore[arg-type]
            created_at=row.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        # Sanitize: do NOT echo str(e) to the client (see POST handler note).
        logger.error(f"Failed to read API key for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{agent_id}/api-key", response_model=APIKeyRevokeResponse)
async def revoke_agent_api_key(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> APIKeyRevokeResponse:
    """Soft-revoke the agent's active API key.

    Idempotent: calling DELETE on an agent with no active key still
    returns HTTP 200 with ``revoked=false``. This lets clients call
    DELETE blindly without first getting it to check existence.

    Args:
        agent_id: Path parameter.
        current_user: Resolved from JWT.
        db: SQLAlchemy session.

    Returns:
        :class:`APIKeyRevokeResponse` with:
          - ``revoked=true, revoked_at=<now>`` if an active key was just revoked.
          - ``revoked=false, revoked_at=null`` if no active key existed.

    Raises:
        HTTPException 401: missing or invalid JWT.
        HTTPException 404: agent missing / not owned.

    Notes:
        Revoked rows stay in the table forever (we only flip ``revoked_at``).
        The audit trail of "when was a key created and when was it
        revoked" is the entire point of soft-delete here; hard-deleting
        would also lose the ability to answer "is this old hash one
        we issued?" during incident response.
    """
    try:
        _get_owned_agent_or_404(agent_id, current_user, db)

        now = datetime.now(timezone.utc)
        row = (
            db.query(AgentApiKey)
            .filter(
                AgentApiKey.agent_id == agent_id,
                AgentApiKey.revoked_at.is_(None),
            )
            .first()
        )
        if row is None:
            # Idempotent no-op path; same HTTP shape as "yes we revoked".
            logger.info(f"Revoke API key for agent {agent_id}: no active key (no-op)")
            return APIKeyRevokeResponse(revoked=False, revoked_at=None)

        row.revoked_at = now  # type: ignore[assignment]
        row.updated_at = now  # type: ignore[assignment]
        db.commit()
        db.refresh(row)

        logger.info(f"Revoked API key for agent {agent_id} (prefix={row.key_prefix})")
        return APIKeyRevokeResponse(revoked=True, revoked_at=row.revoked_at)

    except HTTPException:
        raise
    except Exception as e:
        # Sanitize: do NOT echo str(e) to the client (see POST handler note).
        logger.error(f"Failed to revoke API key for agent {agent_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


# ===== Preview Models =====


class AgentPreviewRequest(BaseModel):
    """Request model for agent preview."""

    instructions: Optional[str] = Field(None, description="System instructions/prompt")
    execution_mode: Optional[str] = Field(
        "balanced", description="Execution mode: flash, balanced, think, or auto"
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

        pattern = get_agent_pattern_for_execution_mode(execution_mode)

        tracer = create_agent_tracer(
            task_id=preview_task_id,
            user_id=int(current_user.id),
            trace_name=f"xagent-web-agent-preview-{preview_task_id}",
            session_id=preview_task_id,
            tags=["xagent", "web", "preview", "agent-builder"],
            metadata={
                "source": "xagent-web",
                "task_id": preview_task_id,
                "is_preview": True,
                "preview_transport": "rest",
            },
        )

        enhanced_system_prompt = enhance_system_prompt_with_kb(
            request.instructions if request.instructions else None,
            request.knowledge_bases if request.knowledge_bases is not None else None,
        )

        # Create agent service (Langfuse only, no database/websocket logging)
        memory = InMemoryMemoryStore()
        agent_service = AgentService(
            name="preview_agent",
            llm=default_llm,
            fast_llm=fast_llm,
            vision_llm=vision_llm,
            compact_llm=compact_llm,
            memory=memory,
            tool_config=tool_config,
            pattern=pattern,
            id=preview_task_id,
            enable_workspace=True,  # Both patterns support workspace
            workspace_base_dir=str(get_uploads_dir() / "preview"),
            task_id=preview_task_id,  # Add task_id for proper tool initialization
            tracer=tracer,
            system_prompt=enhanced_system_prompt,
            memory_enabled=False,
        )

        # Execute task with system prompt in context
        execution_context = {}
        if enhanced_system_prompt:
            execution_context["system_prompt"] = enhanced_system_prompt

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
