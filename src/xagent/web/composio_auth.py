"""Connect Facebook through Composio-hosted auth (#3532).

While our Meta App Review is pending, Toby's Facebook connector can use
Composio's own Meta App. ``XAGENT_FACEBOOK_AUTH_BACKEND=composio`` selects this
flow for new Facebook connects only; Meta Ads, WhatsApp and Instagram share the
``meta`` provider and stay native.

The browser makes three trips, all on the Meta callback path, because that is
the only path xagent-saas routes per region and the only path the actor proof
cookie is sent to:

1. **Start** (server side, called by xagent-saas): the usual signed ``state``,
   single-use flow row and actor proof cookie, but ``Location`` points back to
   our own callback path with only ``state``. Nothing is sent to Composio yet.
2. **Begin** (``?state=`` only): the state, proof cookie and flow row are
   checked *before* Composio is called; a refused request creates no link.
   Then a Composio link is created and the browser is sent to it. Composio's
   verification trip will not carry ``state``, so it is kept in an encrypted
   flow cookie together with the connected account id the link created.
3. **Verify** (``?session_uri=`` only, sent by Composio's OAuth callback
   identity verification): the flow cookie gives back the state, the same
   actor checks run, the flow is consumed, and ``complete_auth`` is called
   with the Composio user id derived from *our* session. The account it
   activates must be the one this flow created, ``ACTIVE``, of our auth config
   and of the Facebook toolkit. ``GET /me`` through the proxy gives the
   Facebook user id, and the grant is stored for #3531 to execute.

The three steps are told apart by their parameters only: a ``session_uri``
never begins a flow and a ``state`` never completes one.

Disconnect: every local delete of a Composio grant goes through
``delete_scoped_user_oauth_accounts``, which schedules the Composio account for
deletion once the caller's transaction commits (``schedule_account_deletions``).

Nothing here logs or returns the API key, a connected account id, the
Composio user id or a ``session_uri``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from urllib.parse import urlencode, urlsplit

from cryptography.fernet import InvalidToken
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlalchemy import event

from ..core.utils import public_requests as requests

if TYPE_CHECKING:
    from sqlalchemy.orm import Query, Session

logger = logging.getLogger(__name__)

# Same value as #3531's ``models.user_oauth.COMPOSIO_TOKEN_TYPE``; whichever of
# the two lands second switches to the shared constant.
COMPOSIO_TOKEN_TYPE = "composio"

FACEBOOK_AUTH_BACKEND_ENV = "XAGENT_FACEBOOK_AUTH_BACKEND"
COMPOSIO_API_KEY_ENV = "COMPOSIO_API_KEY"
COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV = "COMPOSIO_FACEBOOK_AUTH_CONFIG_ID"
COMPOSIO_USER_ID_SECRET_ENV = "COMPOSIO_USER_ID_SECRET"
# xagent-saas's switch for per-connection region routing of OAuth callbacks.
# The verification trip carries no ``state``, so it cannot be routed that way
# (tracked in xorbitsai/xagent-saas#2461); the start refuses while it is on.
CONNECTOR_OAUTH_ROUTE_ENV = "XAGENT_TOBY_CONNECTOR_OAUTH_ROUTE_ENABLED"

COMPOSIO_API_BASE_URL = "https://backend.composio.dev/api/v3.1"
GRAPH_ME_URL = "https://graph.facebook.com/v25.0/me"
FACEBOOK_APP_ID = "facebook"
FACEBOOK_TOOLKIT_SLUG = "facebook"
META_PROVIDER = "meta"

# Marks a state minted for this flow. The native callback refuses it, so a
# Composio state can never be spent on a Meta code exchange, and begin refuses
# a state without it.
AUTH_BACKEND_CLAIM = "auth_backend"
AUTH_BACKEND = "composio"

FLOW_COOKIE_NAME = "xagent_composio_flow"
_FLOW_COOKIE_VERSION = 1
_FLOW_TTL_SECONDS = 600
_REQUEST_TIMEOUT_SECONDS = 10
# Composio ids are short nanoid-style strings. Anything else is refused before
# it is put in a URL path or stored.
_ACCOUNT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_FACEBOOK_USER_ID_RE = re.compile(r"[0-9]{1,32}")
_MAX_SESSION_URI_LENGTH = 4096
_NONCE_RE = re.compile(r"[0-9a-f]{64}")

Step = Literal["begin", "verify"]


class ComposioConfig(NamedTuple):
    api_key: str
    auth_config_id: str
    user_id_secret: str


class ComposioError(Exception):
    """A Composio call failed. ``rejected`` means Composio refused it."""

    def __init__(self, reason: str, *, rejected: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.rejected = rejected


def _env(name: str) -> str | None:
    """Read an environment variable, treating blank as unset."""
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip() or None


def facebook_uses_composio(provider: str, app_id: str | None) -> bool:
    """Whether a new connect of this app goes through Composio."""
    if provider.lower() != META_PROVIDER or not app_id:
        return False
    from .mcp_apps import _normalize_oauth_grant_key

    if _normalize_oauth_grant_key(app_id) != FACEBOOK_APP_ID:
        return False
    return (_env(FACEBOOK_AUTH_BACKEND_ENV) or "native").lower() == AUTH_BACKEND


def connector_oauth_route_enabled() -> bool:
    """Read xagent-saas's switch the same way xagent-saas does."""
    value = _env(CONNECTOR_OAUTH_ROUTE_ENV)
    return value is not None and value.lower() == "true"


def load_config() -> ComposioConfig | None:
    api_key = _env(COMPOSIO_API_KEY_ENV)
    auth_config_id = _env(COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV)
    user_id_secret = _env(COMPOSIO_USER_ID_SECRET_ENV)
    if not api_key or not auth_config_id or not user_id_secret:
        return None
    return ComposioConfig(api_key, auth_config_id, user_id_secret)


def _missing_config_names() -> list[str]:
    return [
        name
        for name in (
            COMPOSIO_API_KEY_ENV,
            COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV,
            COMPOSIO_USER_ID_SECRET_ENV,
        )
        if not _env(name)
    ]


def composio_user_id(config: ComposioConfig, *, user_id: int, owner_key: str) -> str:
    """One opaque Composio user per actor, never per workspace."""
    digest = hmac.new(
        config.user_id_secret.encode(),
        f"{user_id}:{owner_key}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"xagent-{digest}"


# --- Pages -----------------------------------------------------------------


def _page(title: str, *paragraphs: str, status_code: int) -> HTMLResponse:
    from .api.auth import _oauth_popup_feedback_response

    response = _oauth_popup_feedback_response(title, *paragraphs)
    response.status_code = status_code
    return response


def _unavailable_page() -> HTMLResponse:
    return _page(
        "Facebook can't be connected right now",
        "The Facebook connection service could not complete the request.",
        "Close this window and select Connect again in a few minutes.",
        status_code=503,
    )


def start_preflight() -> HTMLResponse | None:
    """Refuse a Composio start this deployment cannot finish."""
    if connector_oauth_route_enabled():
        # Fail closed instead of letting the verification trip be routed by
        # region preference (xorbitsai/xagent-saas#2461).
        logger.warning(
            "Composio Facebook connect refused: %s is on, and the verification "
            "step cannot be routed per connection",
            CONNECTOR_OAUTH_ROUTE_ENV,
        )
        return HTMLResponse(
            content=(
                "<h1>Facebook connection is unavailable</h1>"
                "<p>Facebook connections through Composio are not supported "
                "while per-connection OAuth routing is enabled.</p>"
            ),
            status_code=503,
        )
    missing = _missing_config_names()
    if missing:
        logger.error(
            "Composio Facebook connect refused: missing %s", ", ".join(missing)
        )
        from .api.auth import _oauth_provider_config_error

        return _oauth_provider_config_error(FACEBOOK_APP_ID, missing)
    return None


def begin_location(redirect_uri: str, state: str) -> str:
    """Where the start sends the browser: our callback path with ``state``."""
    separator = "&" if "?" in redirect_uri else "?"
    return f"{redirect_uri}{separator}{urlencode({'state': state})}"


def callback_step(provider: str, query: Mapping[str, Any]) -> Step | None:
    """Which Composio trip a Meta callback request is, if any.

    A native Meta callback always carries ``code`` or ``error``. Everything
    else that reaches this path used to be refused as "missing code or
    state", so routing it here changes no native outcome.
    """
    if provider.lower() != META_PROVIDER:
        return None

    def has(name: str) -> bool:
        return query.get(name) is not None

    if has("session_uri"):
        return "verify"
    if has("code") or has("error"):
        return None
    if has("state"):
        return "begin"
    return None


# --- Composio API ------------------------------------------------------------


def _call(
    config_api_key: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    try:
        response = requests.request(
            method=method,
            url=f"{COMPOSIO_API_BASE_URL}{path}",
            headers={
                "x-api-key": config_api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        # Type only: a connection error's text can include the request URL,
        # and the URL can carry a connected account id.
        logger.warning(
            "Composio request failed (method=%s exception_type=%s)",
            method,
            type(exc).__name__,
        )
        raise ComposioError("composio_unreachable") from None
    try:
        payload = response.json()
    except Exception:
        payload = None
    return int(response.status_code), payload


def _valid_account_id(value: object) -> str | None:
    if isinstance(value, str) and _ACCOUNT_ID_RE.fullmatch(value):
        return value
    return None


def create_link(
    config: ComposioConfig, *, user_id: str, callback_url: str
) -> tuple[str, str]:
    """Create a hosted-auth link; return ``(redirect_url, connected_account_id)``."""
    status, payload = _call(
        config.api_key,
        "POST",
        "/connected_accounts/link",
        body={
            "auth_config_id": config.auth_config_id,
            "user_id": user_id,
            "callback_url": callback_url,
        },
    )
    if not 200 <= status < 300 or not isinstance(payload, dict):
        logger.warning("Composio link creation failed (status=%s)", status)
        raise ComposioError("link_failed")
    redirect_url = payload.get("redirect_url")
    account_id = _valid_account_id(payload.get("connected_account_id"))
    if (
        not isinstance(redirect_url, str)
        or urlsplit(redirect_url).scheme != "https"
        or not urlsplit(redirect_url).netloc
        or account_id is None
    ):
        logger.warning("Composio link creation returned an unusable link")
        raise ComposioError("link_failed")
    return redirect_url, account_id


def complete_auth(config: ComposioConfig, *, session_uri: str, user_id: str) -> str:
    """Activate the paused connection; return its connected account id."""
    status, payload = _call(
        config.api_key,
        "POST",
        "/connected_accounts/complete_auth",
        body={"session_uri": session_uri, "user_id": user_id},
    )
    if status in (400, 404):
        # 400: the user id does not own this session. 404: expired, used,
        # or not this project's.
        logger.warning(
            "Composio refused to complete the connection (status=%s)", status
        )
        raise ComposioError("complete_auth_refused", rejected=True)
    if not 200 <= status < 300 or not isinstance(payload, dict):
        logger.warning("Composio could not complete the connection (status=%s)", status)
        raise ComposioError("complete_auth_failed")
    account_id = _valid_account_id(payload.get("connected_account_id"))
    if account_id is None:
        logger.warning("Composio completed a connection without a usable id")
        raise ComposioError("account_invalid", rejected=True)
    return account_id


def verify_account(config: ComposioConfig, *, account_id: str, user_id: str) -> None:
    """Require the account to be ACTIVE, ours, of our auth config and Facebook."""
    status, payload = _call(config.api_key, "GET", f"/connected_accounts/{account_id}")
    if not 200 <= status < 300 or not isinstance(payload, dict):
        logger.warning("Composio account lookup failed (status=%s)", status)
        raise ComposioError("account_lookup_failed")
    toolkit = payload.get("toolkit")
    auth_config = payload.get("auth_config")
    checks = {
        "id": payload.get("id") == account_id,
        "status": payload.get("status") == "ACTIVE",
        # Composio documents user_id as deprecated on this response. When it
        # is present it must match; the binding itself comes from
        # complete_auth and from the account id this flow created.
        "user_id": payload.get("user_id") is None or payload.get("user_id") == user_id,
        "auth_config": isinstance(auth_config, dict)
        and auth_config.get("id") == config.auth_config_id,
        "toolkit": isinstance(toolkit, dict)
        and toolkit.get("slug") == FACEBOOK_TOOLKIT_SLUG,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        logger.warning(
            "Composio account failed verification (checks=%s)", ",".join(failed)
        )
        raise ComposioError("account_invalid", rejected=True)


def fetch_facebook_user_id(
    config: ComposioConfig, *, account_id: str, user_id: str
) -> str:
    """Read the Facebook user id once through Composio's proxy."""
    status, payload = _call(
        config.api_key,
        "POST",
        "/tools/execute/proxy",
        body={
            "endpoint": GRAPH_ME_URL,
            "method": "GET",
            "connected_account_id": account_id,
            "user_id": user_id,
            "parameters": [{"name": "fields", "value": "id", "type": "query"}],
        },
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    upstream_status = payload.get("status") if isinstance(payload, dict) else None
    facebook_id = data.get("id") if isinstance(data, dict) else None
    if (
        not 200 <= status < 300
        or type(upstream_status) is not int
        or not 200 <= upstream_status < 300
        or not isinstance(facebook_id, str)
        or not _FACEBOOK_USER_ID_RE.fullmatch(facebook_id)
    ):
        logger.warning(
            "Composio proxy did not return a Facebook identity "
            "(status=%s upstream_status=%s)",
            status,
            upstream_status if type(upstream_status) is int else "-",
        )
        raise ComposioError("identity_missing")
    return facebook_id


def delete_account(api_key: str, account_id: str) -> bool:
    """Best-effort delete of one connected account. Never raises."""
    if _valid_account_id(account_id) is None:
        return False
    try:
        status, _ = _call(api_key, "DELETE", f"/connected_accounts/{account_id}")
    except ComposioError:
        return False
    except Exception as exc:  # pragma: no cover - _call already narrows
        logger.warning(
            "Composio account deletion failed (exception_type=%s)", type(exc).__name__
        )
        return False
    if 200 <= status < 300 or status == 404:
        # 404: Composio already considers it gone.
        return True
    logger.warning("Composio account deletion failed (status=%s)", status)
    return False


# --- Disconnect: delete Composio accounts after the local commit -------------

_PENDING_DELETES_KEY = "xagent_composio_pending_account_deletes"


def composio_account_ids(query: Query[Any]) -> list[str]:
    """Connected account ids of the Composio grants a query selects.

    Columns only, so no ``UserOAuth`` instance enters the identity map ahead
    of a bulk delete.
    """
    from .models.user_oauth import UserOAuth

    rows = (
        query.filter(UserOAuth.token_type == COMPOSIO_TOKEN_TYPE)
        .with_entities(UserOAuth.access_token)
        .all()
    )
    return [str(row[0]) for row in rows if row[0]]


def schedule_account_deletions(
    db: Session, *, user_id: int, account_ids: Iterable[str]
) -> None:
    """Delete these Composio accounts once the current transaction commits.

    A rollback or a transaction that ends without committing drops them.
    An account still referenced by another of this user's Composio grants
    (checked inside the transaction, after the local delete) is kept.
    """
    candidates = list(dict.fromkeys(account_ids))
    if not candidates:
        return
    from .models.user_oauth import UserOAuth

    remaining = {
        str(row[0])
        for row in db.query(UserOAuth.access_token)
        .filter(
            UserOAuth.user_id == user_id,
            UserOAuth.token_type == COMPOSIO_TOKEN_TYPE,
        )
        .all()
        if row[0]
    }
    to_delete = [account_id for account_id in candidates if account_id not in remaining]
    if not to_delete:
        return
    pending = db.info.get(_PENDING_DELETES_KEY)
    if pending is None:
        pending = db.info[_PENDING_DELETES_KEY] = []
        event.listen(db, "after_commit", _delete_pending_after_commit)
        event.listen(db, "after_transaction_end", _drop_pending_after_transaction)
    pending.extend(account_id for account_id in to_delete if account_id not in pending)


def _delete_pending_after_commit(session: Session) -> None:
    pending = session.info.get(_PENDING_DELETES_KEY)
    if not pending:
        return
    account_ids = list(pending)
    pending.clear()
    # Runs inside commit(): it must never raise, and must not emit SQL.
    try:
        api_key = _env(COMPOSIO_API_KEY_ENV)
        if not api_key:
            logger.warning(
                "Composio accounts were disconnected locally but not deleted: "
                "%s is not set (count=%d)",
                COMPOSIO_API_KEY_ENV,
                len(account_ids),
            )
            return
        failed = sum(
            1 for account_id in account_ids if not delete_account(api_key, account_id)
        )
        if failed:
            logger.warning(
                "Composio account deletion after disconnect failed (failed=%d of %d)",
                failed,
                len(account_ids),
            )
    except Exception as exc:  # pragma: no cover - defensive, see above
        logger.warning(
            "Composio account deletion after disconnect failed (exception_type=%s)",
            type(exc).__name__,
        )


def _drop_pending_after_transaction(session: Session, transaction: Any) -> None:
    # after_commit has already taken what a commit allows; whatever is left
    # when the outermost transaction ends was rolled back or abandoned.
    if transaction.parent is None:
        pending = session.info.get(_PENDING_DELETES_KEY)
        if pending:
            pending.clear()


# --- Browser trips -------------------------------------------------------------


@dataclass(frozen=True)
class _Flow:
    state: str
    nonce: str
    user_id: int
    owner_key: str
    redirect: str | None
    expires_at: float


class _Rejected(Exception):
    def __init__(self, reason: str, response: HTMLResponse) -> None:
        super().__init__(reason)
        self.reason = reason
        self.response = response


def _reject(reason: str) -> _Rejected:
    from .api.auth import (
        _actor_oauth_flow_rejected_response,
        _actor_oauth_flow_used_response,
    )

    if reason in {"flow_used_or_expired", "state_expired"}:
        return _Rejected(reason, _actor_oauth_flow_used_response())
    return _Rejected(reason, _actor_oauth_flow_rejected_response(reason))


def _log_rejection(reason: str) -> None:
    from .api.auth import _log_oauth_callback_rejection

    _log_oauth_callback_rejection(
        META_PROVIDER, reason, app_id=FACEBOOK_APP_ID, actor_flow=True
    )


def _verified_flow(state: str, request: Any, db: Session) -> _Flow:
    """Run every check that does not need Composio. Raises ``_Rejected``."""
    from .api.auth import (
        _MAX_USER_ID,
        _actor_oauth_cookie_name,
        _decrypt_actor_owner_claim,
        _expired_actor_oauth_state,
        _require_actor_oauth_personal_link,
        verify_token,
    )
    from .mcp_apps import BuiltinOAuthServerDefinitionError
    from .models.user import User
    from .services.user_oauth import normalize_user_oauth_resource_owner_key

    payload = verify_token(state)
    if payload is None:
        expired = _expired_actor_oauth_state(state, META_PROVIDER)
        if expired is not None and expired.get(AUTH_BACKEND_CLAIM) == AUTH_BACKEND:
            raise _reject("state_expired")
        raise _reject("state_invalid")
    nonce = payload.get("actor_flow_nonce")
    user_id = payload.get("user_id")
    expires_at = payload.get("exp")
    if (
        payload.get("type") != "oauth_state"
        or payload.get("provider") != META_PROVIDER
        or payload.get(AUTH_BACKEND_CLAIM) != AUTH_BACKEND
        or payload.get("app_id") != FACEBOOK_APP_ID
        or not isinstance(nonce, str)
        or not _NONCE_RE.fullmatch(nonce)
        or type(user_id) is not int
        or not 0 < user_id <= _MAX_USER_ID
        or not isinstance(expires_at, (int, float))
    ):
        raise _reject("claims_invalid")
    owner_claim = payload.get("resource_owner_key")
    try:
        if not isinstance(owner_claim, str):
            raise ValueError("missing actor owner")
        owner_key = normalize_user_oauth_resource_owner_key(
            _decrypt_actor_owner_claim(owner_claim)
        )
        if owner_key is None:
            raise ValueError("missing actor owner")
    except ValueError:
        raise _reject("owner_invalid") from None

    cookie_value = request.cookies.get(_actor_oauth_cookie_name(nonce))
    if not isinstance(cookie_value, str) or not cookie_value:
        raise _reject("cookie_missing")
    if not secrets.compare_digest(
        hashlib.sha256(cookie_value.encode()).hexdigest(), nonce
    ):
        raise _reject("cookie_mismatch")
    if db.query(User.id).filter(User.id == user_id).one_or_none() is None:
        raise _reject("user_missing")
    try:
        _require_actor_oauth_personal_link(
            db, user_id=user_id, provider=META_PROVIDER, app_id=FACEBOOK_APP_ID
        )
    except BuiltinOAuthServerDefinitionError:
        raise _reject("catalog_invalid") from None
    except ValueError:
        raise _reject("link_invalid") from None
    redirect = payload.get("redirect")
    return _Flow(
        state=state,
        nonce=nonce,
        user_id=user_id,
        owner_key=owner_key,
        redirect=redirect if isinstance(redirect, str) else None,
        expires_at=float(expires_at),
    )


def _flow_row_query(db: Session, nonce: str) -> Any:
    from .models.actor_oauth_flow import ActorOAuthFlowState

    return db.query(ActorOAuthFlowState).filter(
        ActorOAuthFlowState.nonce == nonce,
        ActorOAuthFlowState.expires_at > datetime.now(timezone.utc),
    )


def _cookie_scope(db_provider: Any) -> tuple[bool, str]:
    from .api.auth import _actor_oauth_cookie_scope

    return _actor_oauth_cookie_scope(META_PROVIDER, db_provider)


def _encode_flow_cookie(state: str, account_id: str) -> str:
    from ..core.utils.encryption import get_cipher

    envelope = json.dumps(
        {"account": account_id, "state": state, "version": _FLOW_COOKIE_VERSION},
        separators=(",", ":"),
        sort_keys=True,
    )
    return get_cipher().encrypt(envelope.encode()).decode()


def _decode_flow_cookie(value: str) -> tuple[str, str] | None:
    from ..core.utils.encryption import get_cipher

    try:
        envelope = json.loads(
            get_cipher().decrypt(value.encode(), ttl=_FLOW_TTL_SECONDS)
        )
    except (InvalidToken, ValueError, TypeError):
        return None
    if (
        not isinstance(envelope, dict)
        or set(envelope) != {"account", "state", "version"}
        or envelope.get("version") != _FLOW_COOKIE_VERSION
        or not isinstance(envelope.get("state"), str)
        or _valid_account_id(envelope.get("account")) is None
    ):
        return None
    return envelope["state"], envelope["account"]


def _clear_flow_cookie(response: Response, db_provider: Any) -> Response:
    secure, path = _cookie_scope(db_provider)
    response.delete_cookie(
        FLOW_COOKIE_NAME, path=path, secure=secure, httponly=True, samesite="lax"
    )
    return response


def handle_callback(
    step: Step, request: Any, db: Session, db_provider: Any
) -> Response:
    if step == "begin":
        return _handle_begin(request, db, db_provider)
    return _handle_verify(request, db, db_provider)


def _handle_begin(request: Any, db: Session, db_provider: Any) -> Response:
    state = request.query_params.get("state")
    try:
        flow = _verified_flow(state, request, db)
        if _flow_row_query(db, flow.nonce).count() != 1:
            raise _reject("flow_used_or_expired")
    except _Rejected as rejected:
        db.rollback()
        _log_rejection(rejected.reason)
        return rejected.response
    db.rollback()

    config = load_config()
    if config is None:
        _log_rejection("composio_not_configured")
        return _unavailable_page()
    from .api.auth import _resolve_oauth_redirect_uri

    try:
        redirect_url, account_id = create_link(
            config,
            user_id=composio_user_id(
                config, user_id=flow.user_id, owner_key=flow.owner_key
            ),
            callback_url=_resolve_oauth_redirect_uri(META_PROVIDER, db_provider),
        )
    except ComposioError as exc:
        _log_rejection(exc.reason)
        return _unavailable_page()

    response = RedirectResponse(redirect_url, status_code=302)
    # Our URL carries the state; Composio's page must not receive it.
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    secure, path = _cookie_scope(db_provider)
    response.set_cookie(
        FLOW_COOKIE_NAME,
        _encode_flow_cookie(flow.state, account_id),
        max_age=max(1, min(_FLOW_TTL_SECONDS, int(flow.expires_at - time.time()))),
        secure=secure,
        httponly=True,
        samesite="lax",
        path=path,
    )
    return response


def _handle_verify(request: Any, db: Session, db_provider: Any) -> Response:
    query = request.query_params
    session_uri = query.get("session_uri")
    if (
        query.get("state") is not None
        or query.get("code") is not None
        or not isinstance(session_uri, str)
        or not session_uri
        or len(session_uri) > _MAX_SESSION_URI_LENGTH
    ):
        _log_rejection("claims_invalid")
        return _reject("claims_invalid").response

    cookie = request.cookies.get(FLOW_COOKIE_NAME)
    decoded = _decode_flow_cookie(cookie) if isinstance(cookie, str) else None
    if decoded is None:
        _log_rejection("cookie_missing")
        return _reject("cookie_missing").response
    state, expected_account_id = decoded

    try:
        flow = _verified_flow(state, request, db)
        deleted = _flow_row_query(db, flow.nonce).delete(synchronize_session=False)
        if deleted != 1:
            raise _reject("flow_used_or_expired")
    except _Rejected as rejected:
        db.rollback()
        _log_rejection(rejected.reason)
        return _clear_flow_cookie(rejected.response, db_provider)
    # The flow is spent before Composio is asked to activate anything.
    db.commit()

    config = load_config()
    if config is None:
        _log_rejection("composio_not_configured")
        return _clear_flow_cookie(_unavailable_page(), db_provider)
    composio_user = composio_user_id(
        config, user_id=flow.user_id, owner_key=flow.owner_key
    )
    try:
        account_id = complete_auth(
            config, session_uri=session_uri, user_id=composio_user
        )
    except ComposioError as exc:
        _log_rejection(exc.reason)
        response = (
            _reject("composio_rejected").response
            if exc.rejected
            else _unavailable_page()
        )
        return _clear_flow_cookie(response, db_provider)

    if account_id != expected_account_id:
        # Not the link this flow created, so not ours to bind or to delete.
        logger.warning("Composio completed a connection this flow did not create")
        _log_rejection("account_invalid")
        return _clear_flow_cookie(_reject("composio_rejected").response, db_provider)

    # From here on the account is the one this flow created for this actor:
    # if it cannot be bound, it is deleted rather than left unbound.
    try:
        verify_account(config, account_id=account_id, user_id=composio_user)
        facebook_id = fetch_facebook_user_id(
            config, account_id=account_id, user_id=composio_user
        )
        stored = _store_grant(
            db,
            flow=flow,
            account_id=account_id,
            composio_user=composio_user,
            facebook_id=facebook_id,
        )
    except ComposioError as exc:
        _log_rejection(exc.reason)
        delete_account(config.api_key, account_id)
        response = (
            _reject("composio_rejected").response
            if exc.rejected
            else _unavailable_page()
        )
        return _clear_flow_cookie(response, db_provider)
    except _Rejected as rejected:
        _log_rejection(rejected.reason)
        delete_account(config.api_key, account_id)
        return _clear_flow_cookie(rejected.response, db_provider)
    except Exception as exc:
        db.rollback()
        logger.warning(
            "Composio Facebook grant could not be stored (exception_type=%s)",
            type(exc).__name__,
        )
        delete_account(config.api_key, account_id)
        return _clear_flow_cookie(
            HTMLResponse(
                content=(
                    "<h1>Authentication Failed</h1><p>An unexpected error "
                    "occurred while connecting this account. Please try "
                    "again.</p>"
                ),
                status_code=500,
            ),
            db_provider,
        )
    return _clear_flow_cookie(stored, db_provider)


def _store_grant(
    db: Session,
    *,
    flow: _Flow,
    account_id: str,
    composio_user: str,
    facebook_id: str,
) -> Response:
    from .api.auth import _lock_actor_link, _oauth_success_response
    from .models.user import User
    from .models.user_oauth import UserOAuth
    from .services.user_oauth import delete_scoped_user_oauth_accounts

    try:
        # Composio I/O can overlap a personal disconnect.
        _lock_actor_link(
            db, user_id=flow.user_id, provider=META_PROVIDER, app_id=FACEBOOK_APP_ID
        )
    except ValueError:
        db.rollback()
        raise _reject("link_changed_after_exchange") from None
    db.query(User.id).filter(User.id == flow.user_id).with_for_update().one()
    # Replaces this actor's previous Facebook grant; a previous Composio
    # account is deleted once this commits.
    delete_scoped_user_oauth_accounts(
        db,
        user_id=flow.user_id,
        resource_owner_key=flow.owner_key,
        providers=[FACEBOOK_APP_ID],
    )
    db.add(
        UserOAuth(
            user_id=flow.user_id,
            provider=FACEBOOK_APP_ID,
            resource_owner_key=flow.owner_key,
            provider_user_id=facebook_id,
            access_token=account_id,
            token_type=COMPOSIO_TOKEN_TYPE,
            refresh_token=None,
            expires_at=None,
            instance_url=composio_user,
            scope=None,
            email=None,
        )
    )
    db.commit()
    return _oauth_success_response(
        redirect_url=flow.redirect, email=None, provider_label=FACEBOOK_APP_ID
    )
