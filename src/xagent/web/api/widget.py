"""Web Widget API route handlers"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from ..models.agent import Agent
from ..models.database import get_db
from ..models.task import Task, TaskStatus
from ..models.user import User
from ..models.user_channel import UserChannel
from ..schemas.chat import TaskCreateRequest, TaskCreateResponse
from ..utils.db_timezone import format_datetime_for_api
from .websocket import (
    handle_chat_message,
    handle_execute_task,
    handle_intervention,
    handle_status_request,
    manager,
)

logger = logging.getLogger(__name__)

widget_router = APIRouter(prefix="/api/widget", tags=["widget"])


class WidgetAuthRequest(BaseModel):
    guest_id: str
    agent_id: Optional[int] = None


class WidgetAuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    agent_id: Optional[int] = None
    agent_name: Optional[str] = None
    agent_logo: Optional[str] = None


def create_widget_access_token(data: Dict[str, Any]) -> str:
    """Create JWT access token for widget guest"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=30)
    to_encode.update({"exp": expire, "type": "widget"})
    encoded_jwt: str = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def get_widget_user(token: str, db: Session) -> tuple[User, int, str]:
    """Get user, channel_id, and guest_id from widget token"""
    try:
        if token.startswith("Bearer "):
            token = token[7:]

        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "widget":
            raise ValueError("Invalid token type")

        user_id = payload.get("user_id")
        channel_id = payload.get("channel_id")
        guest_id = payload.get("guest_id")

        if not user_id or not guest_id:
            raise ValueError("Invalid token payload")

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        return user, channel_id, guest_id

    except Exception as e:
        logger.error(f"Widget token validation error: {e}")
        raise HTTPException(status_code=401, detail="Invalid widget token")


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

    # Authenticate via agent_id directly (since web widget channel is deprecated)
    if agent_id:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
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
    if agent_id:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if agent:
            agent_name = agent.name
            agent_logo = agent.logo_url

    channel_id = target_channel.id if target_channel else None

    access_token = create_widget_access_token(
        {
            "sub": user.username,
            "user_id": user.id,
            "channel_id": channel_id,
            "guest_id": request.guest_id,
        }
    )

    return WidgetAuthResponse(
        access_token=access_token,
        agent_id=agent_id,
        agent_name=agent_name,
        agent_logo=agent_logo,
    )


security = HTTPBearer()


def get_current_widget_user_dep(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> tuple[User, int, str]:
    return get_widget_user(credentials.credentials, db)


@widget_router.post("/chat/task/create", response_model=TaskCreateResponse)
async def create_widget_task(
    request: TaskCreateRequest,
    widget_info: tuple[User, int, str] = Depends(get_current_widget_user_dep),
    db: Session = Depends(get_db),
) -> Any:
    """Create new chat task for widget guest"""
    user, channel_id, guest_id = widget_info

    try:
        task_description = request.description or ""

        channel = db.query(UserChannel).filter(UserChannel.id == channel_id).first()
        channel_name = channel.channel_name if channel else "Web Widget"

        agent_config = request.agent_config or {}
        agent_config["guest_id"] = guest_id

        # Determine agent_id: prioritize request, fallback to channel config
        agent_id = request.agent_id
        if agent_id is None and channel and channel.config:
            agent_id = channel.config.get("agent_id")

        task_title = request.title or task_description or "Untitled Task"
        if task_title and len(task_title) > 50:
            task_title = task_title[:50] + "..."

        task = Task(
            user_id=user.id,
            title=task_title,
            description=task_description,
            status=TaskStatus.PENDING,
            channel_id=channel_id,
            channel_name=channel_name,
            agent_id=agent_id,
            agent_config=agent_config,
        )

        db.add(task)
        db.commit()
        db.refresh(task)

        return TaskCreateResponse(
            task_id=task.id,
            title=task.title,
            status=task.status.value,
            created_at=format_datetime_for_api(task.created_at)
            if task.created_at
            else None,
            channel_id=task.channel_id,
            channel_name=task.channel_name,
        )

    except Exception as e:
        logger.error(f"Create widget task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@widget_router.websocket("/chat/ws/{task_id}")
async def websocket_widget_chat_endpoint(
    websocket: WebSocket,
    task_id: int,
    token: str = Query(..., description="Authentication token"),
) -> None:
    """WebSocket unified endpoint for widget"""
    db_gen = get_db()
    db = next(db_gen)
    try:
        user, channel_id, guest_id = get_widget_user(token, db)
    except Exception:
        await websocket.close(code=4001, reason="Authentication required")
        db.close()
        return

    try:
        task = (
            db.query(Task)
            .filter(
                Task.id == task_id,
                Task.user_id == user.id,
                Task.channel_id.is_(channel_id)
                if channel_id is None
                else Task.channel_id == channel_id,
            )
            .first()
        )
        if not task:
            await websocket.close(code=4003, reason="Task not found or access denied")
            return

        # Verify guest_id matches
        if not task.agent_config or task.agent_config.get("guest_id") != guest_id:
            await websocket.close(code=4003, reason="Access denied for this guest")
            return
    finally:
        db.close()

    await manager.connect(websocket, task_id)

    try:
        await handle_status_request(websocket, task_id, user)

        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            message_data["user_id"] = user.id
            message_data["user"] = user

            if message_data.get("type") == "chat":
                await handle_chat_message(websocket, task_id, message_data)
            elif message_data.get("type") == "execute_task":
                await handle_execute_task(websocket, task_id, message_data)
            elif message_data.get("type") == "intervention":
                await handle_intervention(websocket, task_id, message_data)

    except WebSocketDisconnect as e:
        logger.info(f"Widget WebSocket disconnected: {e}")
    except Exception as e:
        logger.error(f"Widget WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, task_id)
