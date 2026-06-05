"""Public share-link routes for agent chat."""

from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..models.agent import Agent, AgentStatus, is_workforce_generated_manager_agent
from ..models.database import get_db
from ..models.user import User
from ..schemas.chat import TaskCreateRequest, TaskCreateResponse
from .public_chat_access import (
    PublicChatAuthResponse,
    ShareChatAccessContext,
    build_share_chat_dependency,
    create_public_chat_access_token,
    create_share_chat_task,
    share_chat_websocket_endpoint,
    upload_share_chat_files,
)

share_router = APIRouter(prefix="/api/share", tags=["share"])


class ShareAuthRequest(BaseModel):
    share_token: str


@share_router.post("/auth", response_model=PublicChatAuthResponse)
async def authenticate_share_link(
    request: ShareAuthRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Authenticate a public share link and issue a guest chat token."""
    agent = (
        db.query(Agent)
        .filter(
            Agent.share_token == request.share_token,
            Agent.share_enabled.is_(True),
        )
        .first()
    )
    if not agent or is_workforce_generated_manager_agent(agent):
        raise HTTPException(status_code=404, detail="Share link not found")
    if agent.status != AgentStatus.PUBLISHED:
        raise HTTPException(status_code=403, detail="Share link is unavailable")

    user = db.query(User).filter(User.id == agent.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Agent owner not found")

    access_token = create_public_chat_access_token(
        {
            "sub": user.username,
            "user_id": user.id,
            "auth_mode": "share",
            "share_agent_id": int(agent.id),
            "share_token": agent.share_token,
        }
    )
    return PublicChatAuthResponse(
        access_token=access_token,
        agent_id=int(agent.id),
        agent_name=agent.name,
        agent_logo=agent.logo_url,
        agent_description=agent.description,
        suggested_prompts=agent.suggested_prompts or [],
    )


get_current_share_user_dep = build_share_chat_dependency()


@share_router.post("/files/upload")
async def upload_share_file(
    file: UploadFile | None = File(None),
    files: list[UploadFile] | None = File(None),
    task_type: str = Form(...),
    message: str = Form(""),
    task_id: str = Form(None),
    folder: str = Form(None),
    share_info: ShareChatAccessContext = Depends(get_current_share_user_dep),
    db: Session = Depends(get_db),
) -> Any:
    return await upload_share_chat_files(
        file=file,
        files=files,
        task_type=task_type,
        message=message,
        task_id=task_id,
        folder=folder,
        access_context=share_info,
        db=db,
    )


@share_router.post("/chat/task/create", response_model=TaskCreateResponse)
async def create_share_task(
    request: TaskCreateRequest,
    share_info: ShareChatAccessContext = Depends(get_current_share_user_dep),
    db: Session = Depends(get_db),
) -> Any:
    return await create_share_chat_task(
        request=request,
        access_context=share_info,
        db=db,
        default_channel_name="Shared Agent",
    )


@share_router.websocket("/chat/ws/{task_id}")
async def websocket_share_chat_endpoint(
    websocket: WebSocket,
    task_id: int,
    token: str = Query(..., description="Authentication token"),
) -> None:
    await share_chat_websocket_endpoint(
        websocket=websocket,
        task_id=task_id,
        token=token,
    )
