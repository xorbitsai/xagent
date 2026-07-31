import html
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, List
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from xagent.config import (
    get_app_base_url,
    get_slack_app_token,
    get_slack_client_id,
    get_slack_client_secret,
    get_slack_oauth_redirect_uri,
)
from xagent.web.api.auth import (
    create_access_token,
    get_current_user,
    verify_token,
)
from xagent.web.models.database import get_db
from xagent.web.models.user import User
from xagent.web.models.user_channel import (
    SlackOAuthFlowState,
    UserChannel,
    has_production_channel_encryption_key,
)
from xagent.web.schemas.user_channel import (
    UserChannelCreate,
    UserChannelResponse,
    UserChannelUpdate,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Slack workspace identity is established by the OAuth token exchange and must
# never be writable through the generic channel-update endpoint (see
# update_user_channel); a client-supplied team_id would re-route another
# tenant's Slack events into this row.
_SLACK_SERVER_MANAGED_CONFIG_KEYS = frozenset(
    {
        "installation_mode",
        "team_id",
        "bot_user_id",
        "slack_app_id",
        "enterprise_id",
        "scope",
    }
)

SLACK_OAUTH_AUTHORIZE_URL = "https://slack.com/oauth/v2/authorize"
SLACK_OAUTH_ACCESS_URL = "https://slack.com/api/oauth.v2.access"
SLACK_OAUTH_SCOPES = (
    "app_mentions:read",
    "chat:write",
    "files:read",
    "files:write",
    "im:history",
    "mpim:history",
    "channels:history",
    "groups:history",
)


async def trigger_telegram_sync() -> None:
    """Helper to safely trigger telegram bot sync in background"""
    from xagent.web.channels.telegram.bot import get_telegram_channel

    tg = get_telegram_channel()

    try:
        await tg._sync_bots_async()
        logger.info("Successfully triggered telegram sync in main event loop")
    except Exception as e:
        logger.error(f"Failed to trigger telegram sync: {e}")


async def trigger_feishu_sync() -> None:
    """Helper to safely trigger feishu bot sync in background"""
    from xagent.web.channels.feishu.bot import get_feishu_channel

    fs = get_feishu_channel()

    try:
        await fs._sync_bots_async()
        logger.info("Successfully triggered feishu sync in main event loop")
    except Exception as e:
        logger.error(f"Failed to trigger feishu sync: {e}")


async def trigger_slack_sync() -> None:
    """Safely synchronize Slack Socket Mode clients in the main event loop."""
    from xagent.web.channels.slack.bot import get_slack_channel

    slack = get_slack_channel()
    try:
        await slack._sync_bots_async()
        logger.info("Successfully triggered Slack sync in main event loop")
    except Exception as e:
        logger.error(f"Failed to trigger Slack sync: {e}")


def get_telegram_bot_name_sync(token: str) -> str:
    try:
        proxy_url = (
            os.getenv("HTTPS_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("HTTP_PROXY")
            or os.getenv("http_proxy")
        )

        with httpx.Client(proxy=proxy_url) as client:
            resp = client.get(
                f"https://api.telegram.org/bot{token}/getMe", timeout=10.0
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    return str(data["result"].get("first_name", "Telegram Bot"))
    except Exception as e:
        logger.error(f"Failed to fetch telegram bot name: {e}")
    return "Telegram Bot"


def get_feishu_bot_name_sync(app_id: str, app_secret: str) -> str:
    try:
        with httpx.Client() as client:
            url = (
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            )
            resp = client.post(
                url, json={"app_id": app_id, "app_secret": app_secret}, timeout=10.0
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 0:
                    token = data["tenant_access_token"]
                    info_url = "https://open.feishu.cn/open-apis/bot/v3/info"
                    info_resp = client.get(
                        info_url,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    )
                    if info_resp.status_code == 200:
                        info_data = info_resp.json()
                        if info_data.get("code") == 0:
                            return str(info_data["bot"].get("app_name", "Feishu Bot"))
    except Exception as e:
        logger.error(f"Failed to fetch feishu bot name: {e}")
    return "Feishu Bot"


def get_slack_bot_name_sync(bot_token: str) -> str:
    try:
        with httpx.Client() as client:
            response = client.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {bot_token}"},
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    return str(data.get("user") or "Slack Bot")
    except Exception as e:
        logger.error(f"Failed to fetch Slack bot name: {e}")
    return "Slack Bot"


def _slack_oauth_missing_config() -> list[str]:
    missing: list[str] = []
    if not get_slack_client_id():
        missing.append("XAGENT_SLACK_CLIENT_ID")
    if not get_slack_client_secret():
        missing.append("XAGENT_SLACK_CLIENT_SECRET")
    if not get_slack_app_token():
        missing.append("XAGENT_SLACK_APP_TOKEN")
    if not get_slack_oauth_redirect_uri():
        missing.append("XAGENT_SLACK_REDIRECT_URI or XAGENT_PUBLIC_API_BASE_URL")
    if not has_production_channel_encryption_key():
        # Workspace tokens are persisted through UserChannel's encrypted
        # config; without a real key the "encrypted at rest" guarantee is
        # silently false, so refuse to advertise the OAuth install flow.
        missing.append("ENCRYPTION_KEY (must not be the built-in dev default)")
    return missing


def _release_slack_oauth_claim(db: Session, *, nonce: str) -> None:
    """Un-consume a claimed OAuth nonce so the user can retry the same link.

    Only for failures that are not the user's doing (transport error, Slack
    5xx). The row stays subject to its original ``expires_at``, so releasing
    it cannot extend the window an attacker has to replay a state.
    """
    if not nonce:
        return
    try:
        db.rollback()
        db.query(SlackOAuthFlowState).filter(SlackOAuthFlowState.nonce == nonce).update(
            {SlackOAuthFlowState.consumed_at: None}, synchronize_session=False
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.warning(
            "Could not release the Slack OAuth nonce after a failed exchange",
            exc_info=True,
        )


def _frontend_origin_for_slack_oauth(request: Request) -> str:
    candidate = get_app_base_url() or request.headers.get("origin") or ""
    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return f"{request.url.scheme}://{request.url.netloc}"


def _script_safe_json(value: Any) -> str:
    """JSON-encode a value for embedding inside an inline <script> block.

    ``json.dumps`` leaves ``/`` and ``<`` intact, so attacker-controlled data
    containing ``</script>`` would close the surrounding script element early
    and execute as injected markup. Escaping ``<``, ``>`` and ``&`` keeps the
    output inert to the HTML tokenizer while remaining valid JSON.
    """
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _slack_oauth_popup_response(
    *,
    success: bool,
    target_origin: str,
    message: str,
    workspace_name: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    event_type = "slack-oauth-success" if success else "slack-oauth-error"
    heading = "Slack connected" if success else "Slack connection failed"
    payload = {
        "type": event_type,
        "message": message,
        "workspace_name": workspace_name,
    }
    return HTMLResponse(
        content=f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(heading)}</title>
  </head>
  <body>
    <main>
      <h1>{html.escape(heading)}</h1>
      <p>{html.escape(message)}</p>
      <p>You can close this window.</p>
    </main>
    <script>
      if (window.opener) {{
        window.opener.postMessage(
          {_script_safe_json(payload)},
          {_script_safe_json(target_origin)}
        );
        window.close();
      }}
    </script>
  </body>
</html>""",
        status_code=status_code,
        headers={
            "Content-Security-Policy": (
                "default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'"
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


def _unique_slack_channel_name(
    db: Session,
    *,
    user_id: int,
    workspace_name: str,
    team_id: str,
    exclude_channel_id: int | None = None,
) -> str:
    base_name = workspace_name.strip() or "Slack Workspace"
    existing_names = {
        str(channel.channel_name)
        for channel in db.query(UserChannel)
        .filter(
            UserChannel.user_id == user_id,
            UserChannel.channel_type == "slack",
        )
        .all()
        if exclude_channel_id is None or int(channel.id) != exclude_channel_id
    }
    if base_name not in existing_names:
        return base_name
    return f"{base_name} ({team_id})"


def _upsert_slack_oauth_channel(
    db: Session,
    *,
    user_id: int,
    token_data: dict[str, Any],
) -> UserChannel:
    access_token = str(token_data.get("access_token") or "")
    team = token_data.get("team")
    team_data = team if isinstance(team, dict) else {}
    team_id = str(team_data.get("id") or "")
    workspace_name = str(team_data.get("name") or "Slack Workspace")
    bot_user_id = str(token_data.get("bot_user_id") or "")
    if not access_token or not team_id:
        raise ValueError("Slack OAuth response is missing the workspace or bot token")

    existing: UserChannel | None = None
    slack_channels = (
        db.query(UserChannel)
        .filter(UserChannel.channel_type == "slack")
        # Deterministic: if duplicate rows ever claim the same workspace, the
        # oldest row wins consistently rather than by arbitrary result order.
        .order_by(UserChannel.id)
        .all()
    )
    reusable_dead_row: UserChannel | None = None
    for channel in slack_channels:
        config = channel.config
        same_workspace = (
            config.get("installation_mode") == "oauth"
            and str(config.get("team_id") or "") == team_id
        )
        same_token = config.get("bot_token") == access_token
        if not (same_workspace or same_token):
            continue
        if not bool(channel.is_active):
            # A deactivated row is the residue of an uninstall: its token is
            # dead, so it must not block anyone -- including a different
            # Xagent user -- from installing this workspace again.
            if reusable_dead_row is None and int(channel.user_id) == user_id:
                reusable_dead_row = channel
            continue
        if int(channel.user_id) != user_id:
            raise ValueError(
                "This Slack workspace is already connected to another Xagent user"
            )
        existing = channel
        break
    if existing is None:
        existing = reusable_dead_row

    existing_config = existing.config if existing is not None else {}
    enterprise = token_data.get("enterprise")
    enterprise_data = enterprise if isinstance(enterprise, dict) else {}
    if existing is not None:
        # Reinstalls keep whatever access policy the owner configured.
        allowed_users = existing_config.get("allowed_users")
    else:
        # One-click installs never pass through the config dialog, so an
        # allow-all default would silently grant every workspace member
        # access. Start with the installer only; the owner can widen it.
        authed_user = token_data.get("authed_user")
        authed_user_data = authed_user if isinstance(authed_user, dict) else {}
        installer_id = str(authed_user_data.get("id") or "")
        if not installer_id:
            # Falling back to None here would mean allow-all for the whole
            # workspace, which is the opposite of the intended default.
            raise ValueError(
                "Slack did not identify the installing user, so access could "
                "not be restricted to them. Please retry the installation."
            )
        allowed_users = [installer_id]
    config = {
        "installation_mode": "oauth",
        "bot_token": access_token,
        "team_id": team_id,
        "workspace_name": workspace_name,
        "bot_user_id": bot_user_id,
        "slack_app_id": str(token_data.get("app_id") or ""),
        "enterprise_id": str(enterprise_data.get("id") or ""),
        "scope": str(token_data.get("scope") or ""),
        "allowed_users": allowed_users,
    }

    if existing is None:
        channel = UserChannel(
            user_id=user_id,
            channel_type="slack",
            channel_name=_unique_slack_channel_name(
                db,
                user_id=user_id,
                workspace_name=workspace_name,
                team_id=team_id,
            ),
            config=config,
            is_active=True,
        )
        db.add(channel)
    else:
        channel = existing
        channel.config = config
        channel.is_active = True  # type: ignore[assignment]
        if not str(channel.channel_name).strip():
            channel.channel_name = _unique_slack_channel_name(
                db,
                user_id=user_id,
                workspace_name=workspace_name,
                team_id=team_id,
                exclude_channel_id=int(channel.id),
            )  # type: ignore[assignment]

    db.commit()
    db.refresh(channel)
    return channel


@router.post("/slack/oauth/start")
def start_slack_oauth(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """Create a signed Slack authorization URL for the current Xagent user."""
    missing = _slack_oauth_missing_config()
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Slack OAuth is not configured. Missing: {', '.join(missing)}",
        )

    client_id = get_slack_client_id()
    redirect_uri = get_slack_oauth_redirect_uri()
    if client_id is None or redirect_uri is None:
        raise HTTPException(status_code=503, detail="Slack OAuth is not configured")

    nonce = secrets.token_urlsafe(24)
    now = datetime.now(timezone.utc)
    # The signed JWT alone would be valid for its whole lifetime on every
    # replay; this ledger row makes the state single-use — the callback
    # atomically claims the nonce and rejects any second presentation.
    db.query(SlackOAuthFlowState).filter(
        SlackOAuthFlowState.expires_at < now - timedelta(days=1)
    ).delete(synchronize_session=False)
    db.add(
        SlackOAuthFlowState(
            nonce=nonce,
            user_id=int(current_user.id),
            expires_at=now + timedelta(minutes=10),
        )
    )
    db.commit()

    state = create_access_token(
        data={
            "type": "slack_oauth_state",
            "user_id": int(current_user.id),
            "nonce": nonce,
            "frontend_origin": _frontend_origin_for_slack_oauth(request),
        },
        expires_delta=timedelta(minutes=10),
    )
    params = urlencode(
        {
            "client_id": client_id,
            "scope": ",".join(SLACK_OAUTH_SCOPES),
            "redirect_uri": redirect_uri,
            "state": state,
        }
    )
    parsed_redirect_uri = urlparse(redirect_uri)
    callback_origin = f"{parsed_redirect_uri.scheme}://{parsed_redirect_uri.netloc}"
    return {
        "authorize_url": f"{SLACK_OAUTH_AUTHORIZE_URL}?{params}",
        "callback_origin": callback_origin,
    }


@router.get("/slack/oauth/callback")
async def slack_oauth_callback(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    """Exchange Slack's authorization code and create the workspace channel."""
    state = request.query_params.get("state") or ""
    state_payload = verify_token(state) if state else None
    target_origin = (
        str(state_payload.get("frontend_origin"))
        if state_payload and state_payload.get("frontend_origin")
        else _frontend_origin_for_slack_oauth(request)
    )
    if not state_payload or state_payload.get("type") != "slack_oauth_state":
        return _slack_oauth_popup_response(
            success=False,
            target_origin=target_origin,
            message="The Slack authorization request is invalid or expired.",
            status_code=400,
        )

    nonce = str(state_payload.get("nonce") or "")
    raw_state_user_id = state_payload.get("user_id")
    state_user_id = int(raw_state_user_id) if raw_state_user_id is not None else None
    # The browser that completes this callback is NOT verified to be the one
    # that called /oauth/start: an attacker can send their own fresh
    # authorize_url to a victim, and the victim's consent then binds their
    # workspace to the attacker's user_id (install CSRF). Closing that needs a
    # browser binding, which a cookie cannot provide while CORS runs with
    # allow_origins=["*"] -- the first /oauth/start response carries
    # Access-Control-Allow-Origin: * so the browser refuses to store a
    # Set-Cookie from it, breaking cross-origin installs outright. Tracked
    # separately; the claim below only makes a state single-use.
    #
    # Atomically claim the nonce so a replayed state is rejected after its
    # first presentation. The row's user_id must also match the state's, so a
    # tampered pairing cannot survive even if the JWT signature check were
    # ever loosened.
    claim_time = datetime.now(timezone.utc)
    claimed = (
        db.query(SlackOAuthFlowState)
        .filter(
            SlackOAuthFlowState.nonce == nonce,
            SlackOAuthFlowState.user_id == state_user_id,
            SlackOAuthFlowState.consumed_at.is_(None),
            SlackOAuthFlowState.expires_at > claim_time,
        )
        .update(
            {SlackOAuthFlowState.consumed_at: claim_time},
            synchronize_session=False,
        )
        if state_user_id is not None
        else 0
    )
    if claimed != 1:
        db.rollback()
        return _slack_oauth_popup_response(
            success=False,
            target_origin=target_origin,
            message="The Slack authorization request is invalid or expired.",
            status_code=400,
        )
    db.commit()

    error = request.query_params.get("error")
    if error:
        # Slack error codes are short snake_case identifiers. Anything else in
        # this unauthenticated query parameter is attacker-controlled, so
        # restrict it to that alphabet before reflecting it anywhere.
        safe_error = re.sub(r"[^A-Za-z0-9_.-]", "", error)[:64] or "unknown_error"
        return _slack_oauth_popup_response(
            success=False,
            target_origin=target_origin,
            message=f"Slack authorization was not completed: {safe_error}",
            status_code=400,
        )

    code = request.query_params.get("code")
    if not code:
        return _slack_oauth_popup_response(
            success=False,
            target_origin=target_origin,
            message="Slack did not return an authorization code.",
            status_code=400,
        )

    client_id = get_slack_client_id()
    client_secret = get_slack_client_secret()
    redirect_uri = get_slack_oauth_redirect_uri()
    if not client_id or not client_secret or not redirect_uri:
        return _slack_oauth_popup_response(
            success=False,
            target_origin=target_origin,
            message="Slack OAuth is not configured on this Xagent deployment.",
            status_code=503,
        )

    try:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    SLACK_OAUTH_ACCESS_URL,
                    data={"code": code, "redirect_uri": redirect_uri},
                    auth=httpx.BasicAuth(client_id, client_secret),
                    timeout=15.0,
                )
                response.raise_for_status()
                token_data = response.json()
        except (httpx.HTTPError, ValueError):
            # The nonce was claimed before this call so a replay cannot race
            # the exchange. A transport failure or a Slack 5xx is not the
            # user's fault though, so release the claim and let them retry the
            # same authorization instead of restarting the whole flow. Slack's
            # authorization code stays valid for its own short window.
            _release_slack_oauth_claim(db, nonce=nonce)
            raise
        if not token_data.get("ok"):
            raise ValueError(
                str(token_data.get("error") or "Slack token exchange failed")
            )

        # Unreachable: the claim above only succeeds for a non-None user_id
        # matching a live flow row. Kept as an explicit guard rather than an
        # assert, which -O would strip.
        if state_user_id is None:
            raise ValueError("Slack OAuth state is missing the Xagent user")
        user_id = state_user_id
        if db.query(User.id).filter(User.id == user_id).first() is None:
            raise ValueError(
                "The Xagent user who started authorization no longer exists"
            )

        channel = _upsert_slack_oauth_channel(
            db,
            user_id=user_id,
            token_data=token_data,
        )
        background_tasks.add_task(trigger_slack_sync)
        workspace_name = str(
            channel.config.get("workspace_name") or channel.channel_name
        )
        return _slack_oauth_popup_response(
            success=True,
            target_origin=target_origin,
            message=f"{workspace_name} is now connected to Xagent.",
            workspace_name=workspace_name,
        )
    except ValueError as exc:
        # ValueErrors on this path carry fixed, user-facing strings raised by
        # our own validation (for example a workspace already connected to a
        # different Xagent user); they never wrap library internals.
        db.rollback()
        logger.warning("Slack OAuth callback rejected: %s", exc)
        return _slack_oauth_popup_response(
            success=False,
            target_origin=target_origin,
            message=str(exc),
            status_code=400,
        )
    except Exception:
        db.rollback()
        logger.exception("Slack OAuth callback failed")
        return _slack_oauth_popup_response(
            success=False,
            target_origin=target_origin,
            message=(
                "Slack authorization failed because of an internal error. "
                "Please try again or contact your administrator."
            ),
            status_code=400,
        )


@router.get("", response_model=List[UserChannelResponse])
def get_user_channels(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Get all channels configured by the current user."""
    channels = (
        db.query(UserChannel).filter(UserChannel.user_id == current_user.id).all()
    )
    return channels


@router.post("", response_model=UserChannelResponse)
def create_user_channel(
    channel_in: UserChannelCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Create a new channel configuration."""
    if channel_in.channel_type == "slack":
        if channel_in.config.get("installation_mode") == "oauth":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Slack OAuth channels must be created through the "
                    "authorization flow"
                ),
            )
        # Same allowlist the update path enforces: workspace identity is only
        # ever written by the OAuth callback. Accepting it here would let a
        # manual row carry an attacker-chosen team_id, which is inert today
        # only because every read site gates on installation_mode first.
        forbidden = sorted(
            key
            for key in channel_in.config
            if key in _SLACK_SERVER_MANAGED_CONFIG_KEYS and key != "installation_mode"
        )
        if forbidden:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Slack config field(s) {', '.join(forbidden)} are managed "
                    "by the authorization flow and cannot be set"
                ),
            )
        # Socket Mode needs both tokens; _sync_manual_bots skips a row that is
        # missing either, so without this the row would silently never start.
        if not (
            channel_in.config.get("bot_token") and channel_in.config.get("app_token")
        ):
            raise HTTPException(
                status_code=400,
                detail="Slack manual setup requires both a bot token and an app token",
            )

    # Auto-fetch channel name if not provided
    channel_name = channel_in.channel_name
    if not channel_name or not channel_name.strip():
        if channel_in.channel_type == "telegram":
            token = channel_in.config.get("bot_token", "")
            channel_name = (
                get_telegram_bot_name_sync(token) if token else "Telegram Bot"
            )
        elif channel_in.channel_type == "feishu":
            app_id = channel_in.config.get("app_id", "")
            app_secret = channel_in.config.get("app_secret", "")
            channel_name = (
                get_feishu_bot_name_sync(app_id, app_secret)
                if app_id and app_secret
                else "Feishu Bot"
            )
        elif channel_in.channel_type == "slack":
            token = channel_in.config.get("bot_token", "")
            channel_name = get_slack_bot_name_sync(token) if token else "Slack Bot"
        else:
            channel_name = "Unknown Bot"

    # Check for duplicate name or token
    existing_channels = (
        db.query(UserChannel)
        .filter(UserChannel.channel_type == channel_in.channel_type)
        .all()
    )

    for ch in existing_channels:
        if ch.user_id == current_user.id and ch.channel_name == channel_name:
            raise HTTPException(status_code=400, detail="Channel name already exists")

        ch_token = ch.config.get("bot_token")
        in_token = channel_in.config.get("bot_token")
        if ch_token and in_token and ch_token == in_token:
            raise HTTPException(status_code=400, detail="Bot token already exists")

    channel = UserChannel(
        user_id=current_user.id,
        channel_type=channel_in.channel_type,
        channel_name=channel_name,
        config=channel_in.config,
        is_active=channel_in.is_active,
    )
    db.add(channel)
    db.commit()
    db.refresh(channel)

    # Trigger bot reload via background task
    if channel.channel_type == "telegram":
        background_tasks.add_task(trigger_telegram_sync)
    elif channel.channel_type == "feishu":
        background_tasks.add_task(trigger_feishu_sync)
    elif channel.channel_type == "slack":
        background_tasks.add_task(trigger_slack_sync)

    return channel


@router.put("/{channel_id}", response_model=UserChannelResponse)
def update_user_channel(
    channel_id: int,
    channel_in: UserChannelUpdate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Update a channel configuration."""
    channel = (
        db.query(UserChannel)
        .filter(UserChannel.id == channel_id, UserChannel.user_id == current_user.id)
        .first()
    )
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Check for duplicate name or token
    existing_channels = (
        db.query(UserChannel)
        .filter(
            UserChannel.channel_type == channel.channel_type,
            UserChannel.id != channel_id,
        )
        .all()
    )

    new_name = (
        channel_in.channel_name
        if channel_in.channel_name is not None
        else str(channel.channel_name)
    )
    current_config = channel.config
    if channel_in.config is None:
        new_config = current_config
    else:
        if channel.channel_type == "slack":
            # These fields define which Slack workspace the row routes and are
            # written exclusively by the OAuth callback from Slack's verified
            # token-exchange response. Accepting them from a client PUT would
            # let any user re-point their own channel at another tenant's
            # team_id and hijack that workspace's events.
            protected_keys = set(_SLACK_SERVER_MANAGED_CONFIG_KEYS)
            if current_config.get("installation_mode") == "oauth":
                # OAuth rows also own their credentials; manual rows may
                # legitimately rotate tokens through this endpoint.
                protected_keys |= {"bot_token", "app_secret", "app_token"}
            for key, value in channel_in.config.items():
                if key not in protected_keys:
                    continue
                current_value = current_config.get(key)
                if value == current_value:
                    continue
                # Rows created before installation_mode existed may backfill
                # the explicit "manual" marker; that is not a mode change.
                if (
                    key == "installation_mode"
                    and not current_value
                    and value == "manual"
                ):
                    continue
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Slack config field '{key}' is managed by the "
                        "authorization flow and cannot be modified"
                    ),
                )
        new_config = dict(current_config)
        for key, value in channel_in.config.items():
            if key in {"bot_token", "app_secret", "app_token"} and not value:
                continue
            new_config[key] = value

    if not new_name or not new_name.strip():
        if channel.channel_type == "telegram":
            token = new_config.get("bot_token", "") if new_config else ""
            new_name = get_telegram_bot_name_sync(token) if token else "Telegram Bot"
        elif channel.channel_type == "feishu":
            app_id = new_config.get("app_id", "") if new_config else ""
            app_secret = new_config.get("app_secret", "") if new_config else ""
            new_name = (
                get_feishu_bot_name_sync(app_id, app_secret)
                if app_id and app_secret
                else "Feishu Bot"
            )
        elif channel.channel_type == "slack":
            token = new_config.get("bot_token", "") if new_config else ""
            new_name = get_slack_bot_name_sync(token) if token else "Slack Bot"
        else:
            new_name = "Unknown Bot"

    for ch in existing_channels:
        if ch.user_id == current_user.id and ch.channel_name == new_name:
            raise HTTPException(status_code=400, detail="Channel name already exists")

        ch_token = ch.config.get("bot_token")
        in_token = new_config.get("bot_token") if new_config else None
        if ch_token and in_token and ch_token == in_token:
            raise HTTPException(status_code=400, detail="Bot token already exists")

    update_data = channel_in.model_dump(exclude_unset=True)
    update_data.pop("config", None)
    for field, value in update_data.items():
        setattr(channel, field, value)

    if channel_in.config is not None:
        channel.config = new_config
    channel.channel_name = new_name  # type: ignore[assignment]

    db.commit()
    db.refresh(channel)

    if channel.channel_type == "telegram":
        background_tasks.add_task(trigger_telegram_sync)
    elif channel.channel_type == "feishu":
        background_tasks.add_task(trigger_feishu_sync)
    elif channel.channel_type == "slack":
        background_tasks.add_task(trigger_slack_sync)

    return channel


@router.delete("/{channel_id}")
def delete_user_channel(
    channel_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Delete a channel configuration."""
    channel = (
        db.query(UserChannel)
        .filter(UserChannel.id == channel_id, UserChannel.user_id == current_user.id)
        .first()
    )
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    channel_type = channel.channel_type

    db.delete(channel)
    db.commit()

    if channel_type == "telegram":
        background_tasks.add_task(trigger_telegram_sync)
    elif channel_type == "feishu":
        background_tasks.add_task(trigger_feishu_sync)
    elif channel_type == "slack":
        background_tasks.add_task(trigger_slack_sync)

    return {"status": "success"}
