"""Web Widget API route handlers."""

from dataclasses import dataclass
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
from ..models.deployment import DeploymentOwnerType
from ..models.user import User
from ..models.workforce import Workforce
from ..schemas.chat import TaskCreateRequest, TaskCreateResponse
from ..services.deployments import find_enabled_widget_deployment, get_deployment
from .auth import create_access_token
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


class WidgetAuthRequest(BaseModel):
    guest_id: str = Field(max_length=256)
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
    # The owner id is not secret; returning it lets widget.js address the chat
    # iframe without embedding the widget key inside the iframe URL. Exactly
    # one of ``agent_id`` / ``workforce_id`` is set depending on what the key
    # resolves to.
    agent_id: int | None = None
    workforce_id: int | None = None


WidgetAuthResponse = PublicChatAuthResponse


# Ticket owner discriminator. Agent tickets historically carried only
# ``agent_id``; a missing ``owner_type`` is therefore treated as an agent for
# backward compatibility with tickets minted before workforce support.
EMBED_TICKET_OWNER_AGENT = "agent"
EMBED_TICKET_OWNER_WORKFORCE = "workforce"


@dataclass(frozen=True)
class ResolvedWidgetOwner:
    """A validated widget owner (agent XOR workforce) plus the data the auth
    and ticket paths need: the embedding allowlist and the resolved key."""

    allowed_domains: list[str]
    widget_key: str
    agent: Agent | None = None
    workforce: Workforce | None = None


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


def _resolve_widget_agent_by_key(db: Session, widget_key: str) -> Agent | None:
    """Return a widget-enabled agent for the key, or ``None`` if no eligible
    agent matches (unknown key, disabled widget, or a workforce-manager agent).

    Returning ``None`` — rather than raising — lets the caller fall through to
    the workforce lookup; every real failure still collapses to a single 403
    at the caller so keys/agents cannot be enumerated.
    """
    agent = db.query(Agent).filter(Agent.widget_key == widget_key).first()
    if (
        agent is None
        or not agent.widget_key
        or is_workforce_generated_manager_agent(agent)
        or not agent.widget_enabled
    ):
        return None
    return agent


def _resolve_widget_owner_by_key(db: Session, widget_key: str) -> ResolvedWidgetOwner:
    """Resolve a widget key to its owner (agent first, then workforce).

    A widget key is unique to a single owner across both the ``agents`` table
    and the ``deployments`` table, so we try the agent path first and fall
    through to the workforce deployment. All failure modes collapse into a
    single 403 so callers cannot enumerate owners or probe which keys exist.
    """
    # Short-circuit blank/whitespace keys before hitting the database; a real
    # key is a URL-safe token and never matches these anyway.
    if not widget_key or not widget_key.strip():
        raise HTTPException(status_code=403, detail="Invalid widget key")

    agent = _resolve_widget_agent_by_key(db, widget_key)
    if agent is not None:
        return ResolvedWidgetOwner(
            allowed_domains=list(agent.allowed_domains or []),
            widget_key=widget_key,
            agent=agent,
        )

    deployment = find_enabled_widget_deployment(
        db, widget_key, DeploymentOwnerType.WORKFORCE
    )
    if deployment is not None:
        workforce = (
            db.query(Workforce).filter(Workforce.id == int(deployment.owner_id)).first()
        )
        if workforce is not None and workforce.status == "active":
            return ResolvedWidgetOwner(
                allowed_domains=list(deployment.allowed_domains or []),
                widget_key=widget_key,
                workforce=workforce,
            )

    raise HTTPException(status_code=403, detail="Invalid widget key")


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
    owner = _resolve_widget_owner_by_key(db, request.widget_key)

    origin = req.headers.get("origin") or req.headers.get("referer", "")
    origin_domain = _origin_to_domain(origin)
    _require_domain_allowed(origin_domain, owner.allowed_domains)

    # The ticket has no jti/nonce and is intentionally replayable within its
    # short TTL: it only re-certifies "this origin is allowed", which /auth
    # independently re-checks against the live allowlist on every use, and the
    # guest tokens it mints are low-privilege. Replay-within-TTL is accepted.
    claims: dict[str, Any] = {
        "type": EMBED_TICKET_TYPE,
        "embed_origin": origin_domain,
    }
    if owner.workforce is not None:
        claims["owner_type"] = EMBED_TICKET_OWNER_WORKFORCE
        claims["workforce_id"] = int(owner.workforce.id)
    else:
        assert owner.agent is not None
        claims["owner_type"] = EMBED_TICKET_OWNER_AGENT
        claims["agent_id"] = int(owner.agent.id)
    ticket = create_access_token(
        claims,
        expires_delta=timedelta(seconds=EMBED_TICKET_TTL_SECONDS),
    )
    return EmbedTicketResponse(
        ticket=ticket,
        agent_id=int(owner.agent.id) if owner.agent is not None else None,
        workforce_id=int(owner.workforce.id) if owner.workforce is not None else None,
    )


def _workforce_owner_from_ticket(
    db: Session, workforce_id: int, origin_domain: str
) -> ResolvedWidgetOwner:
    """Re-validate a workforce embed ticket against live deployment state."""
    workforce = db.query(Workforce).filter(Workforce.id == workforce_id).first()
    deployment = get_deployment(db, DeploymentOwnerType.WORKFORCE, workforce_id)
    if (
        workforce is None
        or workforce.status != "active"
        or deployment is None
        or not deployment.widget_enabled
        or not deployment.widget_key
    ):
        raise HTTPException(status_code=403, detail="Invalid or expired embed ticket")
    allowed_domains = list(deployment.allowed_domains or [])
    # Re-check so tickets die immediately if the allowlist shrinks.
    _require_domain_allowed(origin_domain, allowed_domains)
    return ResolvedWidgetOwner(
        allowed_domains=allowed_domains,
        widget_key=str(deployment.widget_key),
        workforce=workforce,
    )


def _owner_from_embed_ticket(db: Session, embed_ticket: str) -> ResolvedWidgetOwner:
    """Resolve the owner for the embedded flow from a signed embed ticket.

    The auth fetch runs inside the widget iframe, so its Origin/Referer headers
    reflect the xagent host, not the embedding site. The embedding page's origin
    is instead carried by the backend-signed ticket issued by /embed-ticket,
    where it was validated against the browser-enforced Origin header. A
    client-supplied origin value is never trusted here, and the target owner is
    taken from the ticket's claims rather than any client-supplied id.
    """
    try:
        claims = jwt.decode(embed_ticket, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid or expired embed ticket")

    if claims.get("type") != EMBED_TICKET_TYPE:
        raise HTTPException(status_code=403, detail="Invalid or expired embed ticket")

    origin_domain = str(claims.get("embed_origin") or "")
    # A missing owner_type predates workforce support and is always an agent.
    owner_type = claims.get("owner_type") or EMBED_TICKET_OWNER_AGENT

    if owner_type == EMBED_TICKET_OWNER_WORKFORCE:
        workforce_id = claims.get("workforce_id")
        if not isinstance(workforce_id, int):
            raise HTTPException(
                status_code=403, detail="Invalid or expired embed ticket"
            )
        return _workforce_owner_from_ticket(db, workforce_id, origin_domain)

    ticket_agent_id = claims.get("agent_id")
    if not isinstance(ticket_agent_id, int):
        raise HTTPException(status_code=403, detail="Invalid or expired embed ticket")
    agent = _get_widget_enabled_agent(db, ticket_agent_id)
    allowed_domains: list[str] = agent.allowed_domains or []  # type: ignore
    # Re-check so tickets die immediately if the allowlist shrinks.
    _require_domain_allowed(origin_domain, allowed_domains)
    return ResolvedWidgetOwner(
        allowed_domains=list(allowed_domains),
        widget_key=str(agent.widget_key or ""),
        agent=agent,
    )


def _resolve_widget_auth_owner(
    db: Session, request: WidgetAuthRequest
) -> ResolvedWidgetOwner:
    """Resolve the owner a widget guest token will be scoped to.

    Guest tokens are only issued against a credential the backend can verify:
    a signed embed ticket (embedded flow) or the widget key (direct visit). A
    bare Origin/Referer header — spoofed or genuine — authenticates nothing.
    """
    if request.embed_ticket:
        return _owner_from_embed_ticket(db, request.embed_ticket)
    if request.widget_key:
        # Direct visit (chat page opened outside an iframe): the key alone is
        # the gate. No origin allowlist applies — the allowlist governs
        # embedding sites, and a direct visit is not embedded.
        return _resolve_widget_owner_by_key(db, request.widget_key)
    raise HTTPException(status_code=403, detail=WIDGET_CREDENTIAL_REQUIRED_DETAIL)


@widget_router.post("/auth", response_model=WidgetAuthResponse)
async def authenticate_widget(
    request: WidgetAuthRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Authenticate widget and issue a guest token"""
    owner = _resolve_widget_auth_owner(db, request)

    if owner.workforce is not None:
        workforce = owner.workforce
        user = db.query(User).filter(User.id == int(workforce.owner_user_id)).first()
        if not user:
            raise HTTPException(status_code=401, detail="Widget owner not found")
        access_token = create_public_chat_access_token(
            {
                "sub": user.username,
                "user_id": user.id,
                "channel_id": None,
                "guest_id": request.guest_id,
                "auth_mode": "widget",
                "widget_workforce_id": int(workforce.id),
                # Carry the key so rotate/disable invalidates live guest tokens
                # on their next request (get_public_chat_user re-checks it).
                "widget_key": owner.widget_key,
            }
        )
        return WidgetAuthResponse(
            access_token=access_token,
            workforce_id=int(workforce.id),
            agent_name=workforce.name,
            agent_description=workforce.description,
            suggested_prompts=[],
        )

    agent = owner.agent
    assert agent is not None
    user = db.query(User).filter(User.id == agent.user_id).first()
    if not user:
        raise HTTPException(
            status_code=401, detail="Widget owner not found or invalid agent_id"
        )

    access_token = create_public_chat_access_token(
        {
            "sub": user.username,
            "user_id": user.id,
            "channel_id": None,
            "guest_id": request.guest_id,
            "auth_mode": "widget",
            "widget_agent_id": int(agent.id),
            # Carried so per-request revalidation can reject the token once the
            # owner rotates the key (see ensure_widget_agent_available).
            "widget_key": agent.widget_key,
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
