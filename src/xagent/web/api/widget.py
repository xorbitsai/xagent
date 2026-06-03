"""Web Widget API route handlers."""

from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..models.agent import Agent, is_workforce_generated_manager_agent
from ..models.database import get_db
from ..models.user import User
from ..schemas.chat import TaskCreateRequest, TaskCreateResponse
from .public_chat_access import (
    PublicChatAccessContext,
    PublicChatAuthResponse,
    build_public_chat_dependency,
    create_public_chat_access_token,
    create_public_chat_task,
    public_chat_websocket_endpoint,
    upload_public_chat_files,
)

widget_router = APIRouter(prefix="/api/widget", tags=["widget"])


class WidgetAuthRequest(BaseModel):
    guest_id: str
    agent_id: Optional[int] = None


WidgetAuthResponse = PublicChatAuthResponse


@widget_router.post("/auth", response_model=WidgetAuthResponse)
async def authenticate_widget(
    request: WidgetAuthRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> Any:
    """Authenticate widget and issue a guest token"""
    agent_id = request.agent_id
    user = None
    target_channel = None
    agent = None

    # Authenticate via agent_id directly (since web widget channel is deprecated)
    if agent_id:
        candidate_agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if candidate_agent and not is_workforce_generated_manager_agent(
            candidate_agent
        ):
            agent = candidate_agent
        if agent:
            if not agent.widget_enabled:
                raise HTTPException(
                    status_code=403, detail="Widget is disabled for this agent"
                )

            # Check allowed domains
            allowed_domains: list[str] = agent.allowed_domains or []  # type: ignore

            # Use X-Forwarded-Host if available, then Host, then Origin/Referer
            # This is important when widget is loaded via iframe
            origin = req.headers.get("origin") or req.headers.get("referer", "")

            from urllib.parse import urlparse

            origin_domain = ""

            # Try origin/referer
            if origin:
                parsed = urlparse(origin)
                origin_domain = parsed.netloc or parsed.path

            # If origin_domain is localhost:3000 but host is localhost:8001
            # Next.js might be rewriting the request. Let's use the origin_domain.

            # Check if origin matches any allowed domain
            is_allowed = False
            for domain in allowed_domains:
                if (
                    domain == "*"
                    or domain == origin_domain
                    or (origin_domain and origin_domain.endswith("." + domain))
                ):
                    is_allowed = True
                    break

            if not is_allowed:
                raise HTTPException(
                    status_code=403, detail=f"Domain not allowed: {origin_domain}"
                )

            user = db.query(User).filter(User.id == agent.user_id).first()

    if not user:
        raise HTTPException(
            status_code=401, detail="Widget owner not found or invalid agent_id"
        )

    # Get agent name if available
    agent_name = None
    agent_logo = None
    if agent:
        agent_name = agent.name
        agent_logo = agent.logo_url

    channel_id = target_channel.id if target_channel else None

    access_token = create_public_chat_access_token(
        {
            "sub": user.username,
            "user_id": user.id,
            "channel_id": channel_id,
            "guest_id": request.guest_id,
            "auth_mode": "widget",
            "widget_agent_id": int(agent.id) if agent else None,
        }
    )

    return WidgetAuthResponse(
        access_token=access_token,
        agent_id=agent_id,
        agent_name=agent_name,
        agent_logo=agent_logo,
        agent_description=agent.description if agent else None,
        suggested_prompts=agent.suggested_prompts or [] if agent else [],
    )


get_current_widget_user_dep = build_public_chat_dependency("widget")


@widget_router.post("/files/upload")
async def upload_widget_file(
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    task_type: str = Form(...),
    message: str = Form(""),
    task_id: str = Form(None),
    folder: str = Form(None),
    widget_info: PublicChatAccessContext = Depends(get_current_widget_user_dep),
    db: Session = Depends(get_db),
) -> Any:
    return await upload_public_chat_files(
        file=file,
        files=files,
        task_type=task_type,
        message=message,
        task_id=task_id,
        folder=folder,
        access_context=widget_info,
        db=db,
    )


@widget_router.post("/chat/task/create", response_model=TaskCreateResponse)
async def create_widget_task(
    request: TaskCreateRequest,
    widget_info: PublicChatAccessContext = Depends(get_current_widget_user_dep),
    db: Session = Depends(get_db),
) -> Any:
    """Create new chat task for widget guest."""
    return await create_public_chat_task(
        request=request,
        access_context=widget_info,
        db=db,
        default_channel_name="Web Widget",
    )


@widget_router.websocket("/chat/ws/{task_id}")
async def websocket_widget_chat_endpoint(
    websocket: WebSocket,
    task_id: int,
    token: str = Query(..., description="Authentication token"),
) -> None:
    """WebSocket unified endpoint for widget."""
    await public_chat_websocket_endpoint(
        websocket=websocket,
        task_id=task_id,
        token=token,
        expected_auth_mode="widget",
    )
