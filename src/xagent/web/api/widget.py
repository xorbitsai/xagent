"""Web Widget API route handlers."""

import hashlib
import hmac
from datetime import timedelta
from typing import Any, Optional
from urllib.parse import urlparse

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
from jose import JWTError, jwt
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from ..models.agent import Agent, is_workforce_generated_manager_agent
from ..models.database import get_db
from ..models.user import User
from ..schemas.chat import LatestTaskResponse, TaskCreateRequest, TaskCreateResponse
from .auth import create_access_token
from .public_chat_access import (
    PublicChatAccessContext,
    PublicChatAuthResponse,
    build_public_chat_dependency,
    create_public_chat_access_token,
    create_public_chat_task,
    get_latest_public_chat_task,
    public_chat_websocket_endpoint,
    upload_public_chat_files,
)

widget_router = APIRouter(prefix="/api/widget", tags=["widget"])

EMBED_TICKET_TYPE = "widget_embed_ticket"
EMBED_TICKET_TTL_SECONDS = 60

# Actionable errors for the removed legacy (agent_id / bare-header) auth paths,
# so operators who miss the migration note can diagnose quickly.
WIDGET_KEY_REQUIRED_DETAIL = (
    "A widget key is required. Re-copy the embed snippet from the agent's "
    "App Widget settings."
)
WIDGET_CREDENTIAL_REQUIRED_DETAIL = (
    "Widget authentication requires a valid embed ticket or widget key. "
    "Re-copy the embed snippet from the agent's App Widget settings."
)

# Namespace for guest_ids backed by a verified end-user identity (see
# _resolve_verified_guest_id). Reserved so a client can never forge a
# "verified" guest_id for someone else's end_user_id without producing a
# valid HMAC signature for it -- request validation rejects any
# client-supplied guest_id that starts with this prefix outright.
VERIFIED_END_USER_GUEST_ID_PREFIX = "verified_end_user:"


class WidgetAuthRequest(BaseModel):
    # Opaque, unauthenticated per-browser session id (the widget's own random
    # anonymous fallback). Not used to assert a real identity, so it needs no
    # verification -- knowledge of this high-entropy value is itself the only
    # "credential" it ever carried. Reserved-prefix values are rejected below
    # since only a verified end_user_id may produce one (see
    # _resolve_verified_guest_id).
    guest_id: Optional[str] = Field(default=None, max_length=256)
    # A real end-user identity from the embedding page (data-end-user-id),
    # scoped to a specific user/tenant rather than an anonymous browser. Must
    # be accompanied by end_user_signature: an unverified identity claim is a
    # BOLA/IDOR risk (anyone could claim to be any user), unlike guest_id
    # which was never meant to assert who someone is.
    end_user_id: Optional[str] = Field(default=None, max_length=256)
    # hex-encoded HMAC-SHA256(agent.widget_end_user_secret, end_user_id),
    # computed server-side by the embedding site and never exposed to the
    # browser it's minted for.
    end_user_signature: Optional[str] = Field(default=None, max_length=128)
    # Retained for backward compatibility with older embed pages; the agent is
    # authoritatively resolved from the embed ticket or widget key, never from
    # this client-supplied id.
    agent_id: Optional[int] = None
    # A signed embed ticket is a compact JWT; cap it to reject pathological
    # payloads, matching the length limits on sibling request models.
    embed_ticket: Optional[str] = Field(default=None, max_length=4096)
    # Direct (non-embedded) visits carry the widget key instead of a ticket.
    widget_key: Optional[str] = Field(default=None, max_length=512)


class EmbedTicketRequest(BaseModel):
    # The widget key is the unguessable per-agent credential distributed in the
    # embed snippet; capped like sibling request fields to reject junk payloads.
    # Optional so a legacy (key-less) request yields an actionable 403 rather
    # than a generic 422 validation error.
    widget_key: Optional[str] = Field(default=None, max_length=512)


class EmbedTicketResponse(BaseModel):
    ticket: str
    # The agent id is not secret; returning it lets widget.js address the chat
    # iframe without embedding the widget key inside the iframe URL.
    agent_id: int


WidgetAuthResponse = PublicChatAuthResponse


def _origin_to_domain(origin: str) -> str:
    """Extract a lowercased host[:port] from an origin/referer value."""
    if not origin:
        return ""
    parsed = urlparse(origin)
    return (parsed.netloc or parsed.path).lower()


def _domain_allowed(origin_domain: str, allowed_domains: list[str]) -> bool:
    """Check a domain against the agent allowlist (case-insensitive,
    supports "*" and subdomain suffix matches)."""
    for domain in allowed_domains:
        normalized_domain = domain.strip().lower()
        if (
            normalized_domain == "*"
            or normalized_domain == origin_domain
            or (origin_domain and origin_domain.endswith("." + normalized_domain))
        ):
            return True
    return False


def _require_domain_allowed(origin_domain: str, allowed_domains: list[str]) -> None:
    """Raise 403 unless the domain passes the agent allowlist."""
    if not _domain_allowed(origin_domain, allowed_domains):
        raise HTTPException(
            status_code=403, detail=f"Domain not allowed: {origin_domain}"
        )


def _get_widget_enabled_agent(db: Session, agent_id: int) -> Agent:
    """Load a widget-enabled agent or raise the matching HTTP error."""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent is None or is_workforce_generated_manager_agent(agent):
        raise HTTPException(
            status_code=401, detail="Widget owner not found or invalid agent_id"
        )
    if not agent.widget_enabled:
        raise HTTPException(status_code=403, detail="Widget is disabled for this agent")
    return agent


def _get_widget_agent_by_key(db: Session, widget_key: str) -> Agent:
    """Resolve a widget-enabled agent from its embed key.

    All failure modes (unknown key, disabled widget, workforce-manager agent)
    collapse into a single 403 so callers cannot enumerate agents or probe
    which keys exist.
    """
    # Short-circuit blank/whitespace keys before hitting the database; a real
    # key is a URL-safe token and never matches these anyway.
    if not widget_key or not widget_key.strip():
        raise HTTPException(status_code=403, detail="Invalid widget key")
    agent = db.query(Agent).filter(Agent.widget_key == widget_key).first()
    if (
        agent is None
        or not agent.widget_key
        or is_workforce_generated_manager_agent(agent)
        or not agent.widget_enabled
    ):
        raise HTTPException(status_code=403, detail="Invalid widget key")
    return agent


@widget_router.post("/embed-ticket", response_model=EmbedTicketResponse)
async def issue_widget_embed_ticket(
    request: EmbedTicketRequest,
    req: Request,
    db: Session = Depends(get_db),
) -> Any:
    """Issue a short-lived signed embed ticket to the embedding page.

    The agent is identified by its unguessable widget key, not an enumerable
    agent id: a forged Origin header alone is worthless without the key, which
    can only be obtained from a real deployment. This endpoint is called by
    widget.js from the top-level embedding page, so the browser-enforced Origin
    header carries the real embedding site — unlike fetches from inside the
    widget iframe, whose Origin is the xagent host itself. The signed ticket is
    the only way that validated origin is trusted downstream; the widget never
    self-reports its parent's origin.

    The Origin/allowed_domains check is retained as defense-in-depth: for
    genuine browser traffic it still blocks embedding from non-allowlisted
    sites, but it is no longer the boundary a non-browser client must defeat.
    """
    if not request.widget_key or not request.widget_key.strip():
        # Legacy key-less request (e.g. an old data-agent-id snippet): fail
        # with an actionable error rather than a generic 422.
        raise HTTPException(status_code=403, detail=WIDGET_KEY_REQUIRED_DETAIL)
    agent = _get_widget_agent_by_key(db, request.widget_key)

    allowed_domains: list[str] = agent.allowed_domains or []  # type: ignore
    origin = req.headers.get("origin") or req.headers.get("referer", "")
    origin_domain = _origin_to_domain(origin)
    _require_domain_allowed(origin_domain, allowed_domains)

    # The ticket has no jti/nonce and is intentionally replayable within its
    # short TTL: it only re-certifies "this origin is allowed", which /auth
    # independently re-checks against the live allowlist on every use, and the
    # guest tokens it mints are low-privilege. Replay-within-TTL is accepted.
    ticket = create_access_token(
        {
            "type": EMBED_TICKET_TYPE,
            "agent_id": int(agent.id),
            "embed_origin": origin_domain,
        },
        expires_delta=timedelta(seconds=EMBED_TICKET_TTL_SECONDS),
    )
    return EmbedTicketResponse(ticket=ticket, agent_id=int(agent.id))


def _agent_from_embed_ticket(db: Session, embed_ticket: str) -> Agent:
    """Resolve the agent for the embedded flow from a signed embed ticket.

    The auth fetch runs inside the widget iframe, so its Origin/Referer headers
    reflect the xagent host, not the embedding site. The embedding page's origin
    is instead carried by the backend-signed ticket issued by /embed-ticket,
    where it was validated against the browser-enforced Origin header. A
    client-supplied origin value is never trusted here, and the target agent is
    taken from the ticket's claims rather than any client-supplied id.
    """
    try:
        claims = jwt.decode(embed_ticket, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid or expired embed ticket")

    ticket_agent_id = claims.get("agent_id")
    if claims.get("type") != EMBED_TICKET_TYPE or not isinstance(ticket_agent_id, int):
        raise HTTPException(status_code=403, detail="Invalid or expired embed ticket")

    agent = _get_widget_enabled_agent(db, ticket_agent_id)
    allowed_domains: list[str] = agent.allowed_domains or []  # type: ignore
    origin_domain = str(claims.get("embed_origin") or "")
    # Re-check so tickets die immediately if the allowlist shrinks.
    _require_domain_allowed(origin_domain, allowed_domains)
    return agent


def _resolve_widget_auth_agent(db: Session, request: WidgetAuthRequest) -> Agent:
    """Resolve the agent a widget guest token will be scoped to.

    Guest tokens are only issued against a credential the backend can verify:
    a signed embed ticket (embedded flow) or the widget key (direct visit). A
    bare Origin/Referer header — spoofed or genuine — authenticates nothing.
    """
    if request.embed_ticket:
        return _agent_from_embed_ticket(db, request.embed_ticket)
    if request.widget_key:
        # Direct visit (chat page opened outside an iframe): the key alone is
        # the gate. No origin allowlist applies — the allowlist governs
        # embedding sites, and a direct visit is not embedded.
        return _get_widget_agent_by_key(db, request.widget_key)
    raise HTTPException(status_code=403, detail=WIDGET_CREDENTIAL_REQUIRED_DETAIL)


def _resolve_authenticated_guest_id(agent: Agent, request: WidgetAuthRequest) -> str:
    """Resolve the guest_id to embed in the guest token.

    An end_user_id claim only becomes a guest_id if it carries a valid HMAC
    signature (proving the embedding site's own server vouched for it, not
    just whatever the browser sent) -- otherwise anyone could claim to be any
    user and read their conversation history via /tasks/latest. A bare
    guest_id is left as an opaque, unauthenticated per-browser id exactly as
    before, except it may never itself forge into the reserved "verified"
    namespace: that would let a request skip signing by simply omitting
    end_user_id and passing the target guest_id directly.
    """
    if not request.guest_id and not request.end_user_id:
        raise HTTPException(
            status_code=422, detail="Either guest_id or end_user_id is required"
        )

    if request.end_user_id:
        secret = agent.widget_end_user_secret
        if not secret:
            raise HTTPException(
                status_code=403,
                detail=(
                    "This agent has no end-user signing secret configured; "
                    "generate one from the agent's App Widget settings before "
                    "sending data-end-user-id."
                ),
            )
        expected_signature = hmac.new(
            secret.encode("utf-8"),
            request.end_user_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(
            expected_signature, request.end_user_signature or ""
        ):
            raise HTTPException(status_code=403, detail="Invalid end-user signature")
        return f"{VERIFIED_END_USER_GUEST_ID_PREFIX}{request.end_user_id}"

    guest_id = request.guest_id or ""
    if guest_id.startswith(VERIFIED_END_USER_GUEST_ID_PREFIX):
        raise HTTPException(status_code=403, detail="Invalid guest_id")
    return guest_id


@widget_router.post("/auth", response_model=WidgetAuthResponse)
async def authenticate_widget(
    request: WidgetAuthRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Authenticate widget and issue a guest token"""
    agent = _resolve_widget_auth_agent(db, request)

    user = db.query(User).filter(User.id == agent.user_id).first()
    if not user:
        raise HTTPException(
            status_code=401, detail="Widget owner not found or invalid agent_id"
        )

    guest_id = _resolve_authenticated_guest_id(agent, request)

    access_token = create_public_chat_access_token(
        {
            "sub": user.username,
            "user_id": user.id,
            "channel_id": None,
            "guest_id": guest_id,
            "auth_mode": "widget",
            "widget_agent_id": int(agent.id),
        }
    )

    return WidgetAuthResponse(
        access_token=access_token,
        agent_id=int(agent.id),
        agent_name=agent.name,
        agent_logo=agent.logo_url,
        agent_description=agent.description,
        suggested_prompts=agent.suggested_prompts or [],
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


@widget_router.get("/tasks/latest", response_model=LatestTaskResponse)
async def get_latest_widget_task(
    widget_info: PublicChatAccessContext = Depends(get_current_widget_user_dep),
    db: Session = Depends(get_db),
) -> Any:
    """Look up the guest's most recent task, so a returning guest with a
    stable end-user identity (data-end-user-id) can resume it on a new
    browser/device instead of local storage forcing a fresh conversation."""
    task = get_latest_public_chat_task(db, widget_info)
    return LatestTaskResponse(task_id=int(task.id) if task else None)


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
