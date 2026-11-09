"""Connect apps through Composio-hosted auth (#3532, #3533, #3645).

While our own app approvals are pending, a connector can use Composio's app
instead: Facebook and Instagram (``meta`` provider) today. Each app in
``COMPOSIO_APPS`` names its own provider and has its own switch (for example
``XAGENT_FACEBOOK_AUTH_BACKEND``; set to ``composio``) and selects this flow
for its new connects only; Meta Ads and WhatsApp share the ``meta`` provider
and stay native. Each app reads its identity its own way; Composio's Instagram
toolkit uses Instagram Login, so its identity comes from ``graph.instagram.com``.

The browser makes three trips, all to one shared callback URL whatever the
app's provider (``shared_callback_url``: ``XAGENT_COMPOSIO_CALLBACK_URL``, or
the Meta callback URL while that is unset): Composio's OAuth user verification
is a single URL per project, and the flow and actor proof cookies are scoped
to the path they come back to. The dedicated ``/api/auth/composio/callback``
route and the Meta callback both accept the trips. A state is still minted
for, and only accepted under, its app's own provider:

1. **Start** (server side, called by xagent-saas): the usual signed ``state``,
   single-use flow row and actor proof cookie, but ``Location`` points back to
   our own callback path with only ``state``. Nothing is sent to Composio yet.
2. **Begin** (``?state=`` only): the state, proof cookie and flow row are
   checked *before* Composio is called; a refused request creates no link.
   Then a Composio link is created and the browser is sent to it. Composio's
   verification trip will not carry ``state``, so it is kept in an encrypted
   cookie of this flow's own, together with the connected account id the link
   created. Other flows of the same actor keep their cookies; another actor's
   are dropped, so a browser holds one actor's flows only.
3. **Verify** (``?session_uri=`` only, sent by Composio's OAuth callback
   identity verification): the session names no flow, so every open flow's
   checks run first, then ``complete_auth`` is called with that actor's
   Composio user id, derived from *our* session. The account it activates
   says which flow this was; only that flow is spent. It must be ``ACTIVE``,
   of its app's auth config and toolkit. ``GET /me`` through the proxy gives
   the Facebook user id or the Instagram ``user_id``, and the grant is stored
   for #3531 to execute. A refused completion spends nothing.

A begin needs a state carrying the Composio marker, a verification carries a
``session_uri``, and a denied or failed authorization comes back to
``callback_url`` with a ``connected_account_id`` and no ``state`` (handled only
while an app's switch is on; it gets the native "Permission was not granted"
page). Native Meta callbacks keep their own handling.

Cleanup: the connect flow deletes an account only when the request that spent
its flow fails to bind it. Every other unbound account (a lost complete_auth
answer, an expired flow cookie, a replaced link, a failed delete) is removed
by ``reconcile_orphan_accounts``. The app runs it on an interval only when
``XAGENT_COMPOSIO_RECONCILE_INTERVAL_SECONDS`` is set: it treats every xagent
account in the apps' auth configs that this database does not reference as
unbound, so it must only be enabled where no other database uses the same
auth configs. It only lists the auth configs that are still set: to retire an
app, switch its backend back to ``native`` first, and remove its auth config
only after its Composio grants are gone and a pass has started after
accounts from sign-ins begun before the switch are past the TTL plus grace
(see ``example.env``).

Disconnect: deleting a grant through ``delete_scoped_user_oauth_accounts``
schedules its Composio account for deletion once the caller's outermost
transaction commits (``schedule_account_deletions``); a rolled-back SAVEPOINT
cancels it. A grant removed another way, such as by the database cascade when
an admin deletes a user, leaves its account to the reconciler.

Nothing here logs or returns the API key, a connected account id, the
Composio user id or a ``session_uri``.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast
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
COMPOSIO_CALLBACK_URL_ENV = "XAGENT_COMPOSIO_CALLBACK_URL"

COMPOSIO_API_BASE_URL = "https://backend.composio.dev/api/v3.1"
GRAPH_ME_URL = "https://graph.facebook.com/v25.0/me"
FACEBOOK_APP_ID = "facebook"
FACEBOOK_TOOLKIT_SLUG = "facebook"
INSTAGRAM_AUTH_BACKEND_ENV = "XAGENT_INSTAGRAM_AUTH_BACKEND"
COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID_ENV = "COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID"
INSTAGRAM_GRAPH_ME_URL = "https://graph.instagram.com/v25.0/me"
INSTAGRAM_APP_ID = "instagram"
INSTAGRAM_TOOLKIT_SLUG = "instagram"
META_PROVIDER = "meta"
# Every Composio app's trips (begin, verification, denial) land on one shared
# callback URL, whatever the app's own provider: Composio's OAuth user
# verification is a single URL per project, and the flow and actor proof
# cookies are scoped to the path they come back to. XAGENT_COMPOSIO_CALLBACK_URL
# sets it (normally to COMPOSIO_CALLBACK_PATH); unset, it is the Meta callback
# URL, where these trips have always been accepted.
COMPOSIO_CALLBACK_PATH = "/api/auth/composio/callback"
DEFAULT_CALLBACK_PROVIDER = META_PROVIDER
# Logged as the provider of a trip whose app is not known yet.
UNKNOWN_APP_PROVIDER = "composio"


class ComposioApp(NamedTuple):
    """What differs between the apps that can connect through Composio."""

    app_id: str
    # The OAuth provider the app's connector is started under; its state is
    # minted for this provider and is refused under any other.
    provider: str
    label: str
    backend_env: str
    auth_config_env: str
    toolkit_slug: str
    # GET this through the proxy with ``identity_params`` as its query;
    # ``identity_field`` of the reply, which must fully match
    # ``identity_id_pattern``, is stored as the grant's provider_user_id.
    identity_url: str
    identity_field: str
    identity_params: tuple[tuple[str, str], ...]
    identity_id_pattern: re.Pattern[str]


# Graph ids, Facebook user ids and Instagram professional account ids alike.
_GRAPH_ID_RE = re.compile(r"[0-9]{1,32}")


COMPOSIO_APPS: dict[str, ComposioApp] = {
    FACEBOOK_APP_ID: ComposioApp(
        app_id=FACEBOOK_APP_ID,
        provider=META_PROVIDER,
        label="Facebook",
        backend_env=FACEBOOK_AUTH_BACKEND_ENV,
        auth_config_env=COMPOSIO_FACEBOOK_AUTH_CONFIG_ID_ENV,
        toolkit_slug=FACEBOOK_TOOLKIT_SLUG,
        identity_url=GRAPH_ME_URL,
        identity_field="id",
        identity_params=(("fields", "id"),),
        identity_id_pattern=_GRAPH_ID_RE,
    ),
    INSTAGRAM_APP_ID: ComposioApp(
        app_id=INSTAGRAM_APP_ID,
        provider=META_PROVIDER,
        label="Instagram",
        backend_env=INSTAGRAM_AUTH_BACKEND_ENV,
        auth_config_env=COMPOSIO_INSTAGRAM_AUTH_CONFIG_ID_ENV,
        toolkit_slug=INSTAGRAM_TOOLKIT_SLUG,
        identity_url=INSTAGRAM_GRAPH_ME_URL,
        # The professional account id the Instagram tools address.
        identity_field="user_id",
        identity_params=(("fields", "user_id"),),
        identity_id_pattern=_GRAPH_ID_RE,
    ),
}

# Marks a state minted for this flow. The native callback refuses it, so a
# Composio state can never be spent on a Meta code exchange, and begin refuses
# a state without it.
AUTH_BACKEND_CLAIM = "auth_backend"
AUTH_BACKEND = "composio"

# One cookie per open flow, named after its nonce, so a second connect in the
# same browser never overwrites the first one's verification context.
FLOW_COOKIE_PREFIX = "xagent_composio_flow_"
# Open flows one browser may hold for one actor. Each cookie is about 1.3 kB;
# a begin past this is refused rather than evicting a flow whose session could
# still come back and be activated with no way to bind it.
_MAX_OPEN_FLOWS = 3
_REQUEST_TIMEOUT_SECONDS = 10
# Composio ids are short nanoid-style strings. Anything else is refused before
# it is put in a URL path or stored.
_ACCOUNT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")
_MAX_SESSION_URI_LENGTH = 4096

Step = Literal["begin", "verify", "denied"]


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


def composio_app(provider: str, app_id: str | None) -> ComposioApp | None:
    """The app whose new connects go through Composio, if this one does.

    Each app follows only its own switch, so turning Facebook over never
    moves Instagram, and Meta Ads and WhatsApp have no switch at all. An app
    is only ever started under its own provider.
    """
    if not app_id:
        return None
    from .mcp_apps import _normalize_oauth_grant_key

    app = COMPOSIO_APPS.get(_normalize_oauth_grant_key(app_id) or "")
    if app is None or app.provider != provider.lower() or not _uses_composio(app):
        return None
    return app


def _uses_composio(app: ComposioApp) -> bool:
    backend = (_env(app.backend_env) or "native").lower()
    if (
        backend not in ("native", AUTH_BACKEND)
        and (
            app.backend_env,
            backend,
        )
        not in _WARNED_BACKENDS
    ):
        _WARNED_BACKENDS.add((app.backend_env, backend))
        logger.warning(
            "%s=%r is neither 'native' nor %r; %s connects stay native",
            app.backend_env,
            backend,
            AUTH_BACKEND,
            app.label,
        )
    return backend == AUTH_BACKEND


_WARNED_BACKENDS: set[tuple[str, str]] = set()


def connector_oauth_route_enabled() -> bool:
    """Read xagent-saas's switch the same way xagent-saas does."""
    value = _env(CONNECTOR_OAUTH_ROUTE_ENV)
    return value is not None and value.lower() == "true"


def load_config(app_id: str) -> ComposioConfig | None:
    api_key = _env(COMPOSIO_API_KEY_ENV)
    auth_config_id = _env(COMPOSIO_APPS[app_id].auth_config_env)
    user_id_secret = _env(COMPOSIO_USER_ID_SECRET_ENV)
    if not api_key or not auth_config_id or not user_id_secret:
        return None
    return ComposioConfig(api_key, auth_config_id, user_id_secret)


def _configured_apps() -> dict[str, ComposioConfig]:
    """Every app whose Composio config is complete. The API key and the
    user-id secret are shared, so any of them can complete or delete an
    account; the auth config is the app's own."""
    return {
        app_id: config
        for app_id in COMPOSIO_APPS
        if (config := load_config(app_id)) is not None
    }


def _missing_config_names(app_id: str) -> list[str]:
    return [
        name
        for name in (
            COMPOSIO_API_KEY_ENV,
            COMPOSIO_APPS[app_id].auth_config_env,
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


def _unavailable_page(*app_ids: str) -> HTMLResponse:
    """``app_ids`` are the apps the browser's flows were for, when known."""
    label = " or ".join(COMPOSIO_APPS[app_id].label for app_id in app_ids)
    return _page(
        f"{label or 'This app'} can't be connected right now",
        f"The {label + ' ' if label else ''}connection service could not "
        "complete the request.",
        "Close this window and select Connect again in a few minutes.",
        status_code=503,
    )


def start_preflight(app_id: str) -> HTMLResponse | None:
    """Refuse a Composio start this deployment cannot finish."""
    label = COMPOSIO_APPS[app_id].label
    if connector_oauth_route_enabled():
        # Fail closed instead of letting the verification trip be routed by
        # region preference (xorbitsai/xagent-saas#2461).
        logger.warning(
            "Composio %s connect refused: %s is on, and the verification "
            "step cannot be routed per connection",
            label,
            CONNECTOR_OAUTH_ROUTE_ENV,
        )
        return HTMLResponse(
            content=(
                f"<h1>{label} connection is unavailable</h1>"
                f"<p>{label} connections through Composio are not supported "
                "while per-connection OAuth routing is enabled.</p>"
            ),
            status_code=503,
        )
    try:
        from ..config import get_composio_callback_url

        get_composio_callback_url()
    except ValueError:
        logger.error(
            "Composio %s connect refused: %s is not a valid callback URL",
            label,
            COMPOSIO_CALLBACK_URL_ENV,
        )
        return _unavailable_page(app_id)
    missing = _missing_config_names(app_id)
    if missing:
        logger.error(
            "Composio %s connect refused: missing %s", label, ", ".join(missing)
        )
        from .api.auth import _oauth_provider_config_error

        return _oauth_provider_config_error(app_id, missing)
    return None


def shared_callback_url(db: Session, meta_row: Any = None) -> str:
    """The callback URL every Composio trip returns to.

    ``XAGENT_COMPOSIO_CALLBACK_URL`` when set (``ValueError`` when invalid);
    otherwise the Meta callback URL, read from ``meta_row`` when the caller
    already holds the Meta provider row and from the database otherwise.
    """
    from ..config import get_composio_callback_url
    from .api.auth import _resolve_oauth_redirect_uri

    configured = get_composio_callback_url()
    if configured is not None:
        return configured
    if meta_row is None:
        from .models.oauth_provider import OAuthProvider

        meta_row = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == DEFAULT_CALLBACK_PROVIDER)
            .first()
        )
    return _resolve_oauth_redirect_uri(DEFAULT_CALLBACK_PROVIDER, meta_row)


def begin_location(redirect_uri: str, state: str) -> str:
    """Where the start sends the browser: our callback path with ``state``."""
    separator = "&" if "?" in redirect_uri else "?"
    return f"{redirect_uri}{separator}{urlencode({'state': state})}"


def callback_step(query: Mapping[str, Any]) -> Step | None:
    """Which Composio trip a callback request is, if any.

    Asked on the dedicated Composio callback and on the Meta one, where native
    callbacks keep their own handling: a begin needs a state that carries the
    Composio marker, and a denial is only taken while an app connects through
    Composio. A ``session_uri`` is never part of a native callback.
    """

    def has(name: str) -> bool:
        return query.get(name) is not None

    if has("session_uri"):
        return "verify"
    # Composio sends a denied or failed authorization back to callback_url
    # with its connected_account_id and no state (seen live, #3532).
    if (
        has("connected_account_id")
        and not has("state")
        and not has("code")
        and any(_uses_composio(app) for app in COMPOSIO_APPS.values())
    ):
        return "denied"
    if has("code") or has("error"):
        return None
    if has("state") and _carries_composio_marker(str(query.get("state"))):
        return "begin"
    return None


def _carries_composio_marker(state: str) -> bool:
    """Whether a genuinely signed state, expired or not, is a Composio one."""
    from .api.auth import JWT_ALGORITHM, JWT_SECRET_KEY
    from .jwt_codec import JWTError, decode

    try:
        payload = decode(
            state,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"verify_exp": False},
        )
    except JWTError:
        return False
    return isinstance(payload, dict) and payload.get(AUTH_BACKEND_CLAIM) == AUTH_BACKEND


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
        # Composio reports success, so an account may be active: treat it like
        # a lost answer, not a refusal.
        logger.warning("Composio completed a connection without a usable id")
        raise ComposioError("complete_auth_failed")
    return account_id


def _get_account(config: ComposioConfig, account_id: str) -> dict[str, Any] | None:
    """One connected account as Composio reports it, or None when unreadable."""
    status, payload = _call(config.api_key, "GET", f"/connected_accounts/{account_id}")
    if not 200 <= status < 300 or not isinstance(payload, dict):
        logger.warning("Composio account lookup failed (status=%s)", status)
        return None
    return payload


def verify_account(
    config: ComposioConfig,
    *,
    account_id: str,
    user_id: str,
    toolkit_slug: str,
) -> None:
    """Require the account to be ACTIVE, ours, of our auth config and toolkit."""
    payload = _get_account(config, account_id)
    if payload is None:
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
        "toolkit": isinstance(toolkit, dict) and toolkit.get("slug") == toolkit_slug,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        logger.warning(
            "Composio account failed verification (checks=%s)", ",".join(failed)
        )
        raise ComposioError("account_invalid", rejected=True)


def fetch_provider_user_id(
    config: ComposioConfig,
    *,
    account_id: str,
    user_id: str,
    app_id: str,
) -> str:
    """Read the account's own id once through Composio's proxy."""
    app = COMPOSIO_APPS[app_id]
    status, payload = _call(
        config.api_key,
        "POST",
        "/tools/execute/proxy",
        body={
            "endpoint": app.identity_url,
            "method": "GET",
            "connected_account_id": account_id,
            "user_id": user_id,
            "parameters": [
                {"name": name, "value": value, "type": "query"}
                for name, value in app.identity_params
            ],
        },
    )
    data = payload.get("data") if isinstance(payload, dict) else None
    upstream_status = payload.get("status") if isinstance(payload, dict) else None
    provider_user_id = data.get(app.identity_field) if isinstance(data, dict) else None
    if (
        not 200 <= status < 300
        or type(upstream_status) is not int
        or not 200 <= upstream_status < 300
        or not isinstance(provider_user_id, str)
        or not app.identity_id_pattern.fullmatch(provider_user_id)
    ):
        logger.warning(
            "Composio proxy did not return a %s identity "
            "(status=%s upstream_status=%s)",
            app.label,
            status,
            upstream_status if type(upstream_status) is int else "-",
        )
        raise ComposioError("identity_missing")
    return provider_user_id


def delete_account(api_key: str, account_id: str) -> bool:
    """Best-effort delete of one connected account. Never raises."""
    if _valid_account_id(account_id) is None:
        return False
    try:
        status, _ = _call(api_key, "DELETE", f"/connected_accounts/{account_id}")
    except ComposioError:
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
    pending = _pending_by_transaction(db)
    level = _transaction_level(db)
    bucket = pending.setdefault(level, [])
    bucket.extend(account_id for account_id in to_delete if account_id not in bucket)


def _transaction_level(session: Session) -> Any:
    """The transaction a deletion belongs to: the innermost SAVEPOINT, if any."""
    return session.get_nested_transaction() or session.get_transaction()


def _scope_of(transaction: Any) -> Any:
    """The nearest enclosing SAVEPOINT or outermost transaction."""
    parent = transaction.parent
    while parent is not None and not parent.nested and parent.parent is not None:
        parent = parent.parent
    return parent


def _pending_by_transaction(session: Session) -> dict[Any, list[str]]:
    """Account ids to delete, by the transaction that removed their grant.

    A SAVEPOINT's ids move to its enclosing transaction when it commits and are
    dropped when it rolls back. Only a commit of the outermost transaction
    deletes anything: SQLAlchemy also fires ``after_commit`` for a SAVEPOINT.
    """
    pending = session.info.get(_PENDING_DELETES_KEY)
    if pending is None:
        pending = session.info[_PENDING_DELETES_KEY] = {}
        session.info[_COMMITTED_SAVEPOINTS_KEY] = set()
        event.listen(session, "after_commit", _after_commit)
        event.listen(session, "after_transaction_end", _after_transaction_end)
    return cast(dict[Any, list[str]], pending)


_COMMITTED_SAVEPOINTS_KEY = "xagent_composio_committed_savepoints"


def _after_commit(session: Session) -> None:
    if session.in_nested_transaction():
        # A SAVEPOINT committed; its ids wait for the enclosing transaction.
        session.info[_COMMITTED_SAVEPOINTS_KEY].add(session.get_nested_transaction())
        return
    pending = session.info.get(_PENDING_DELETES_KEY) or {}
    account_ids = list(
        dict.fromkeys(account_id for ids in pending.values() for account_id in ids)
    )
    pending.clear()
    if account_ids:
        _delete_after_commit(account_ids)


def _after_transaction_end(session: Session, transaction: Any) -> None:
    pending = session.info.get(_PENDING_DELETES_KEY)
    if pending is None:
        return
    committed = session.info[_COMMITTED_SAVEPOINTS_KEY]
    ids = pending.pop(transaction, [])
    if transaction.nested and transaction in committed:
        committed.discard(transaction)
        scope = _scope_of(transaction)
        if scope is not None:
            bucket = pending.setdefault(scope, [])
            bucket.extend(i for i in ids if i not in bucket)
        return
    committed.discard(transaction)
    if transaction.parent is None:
        # The outermost transaction ended: a commit has already taken what it
        # allows, so anything left was rolled back or abandoned.
        pending.clear()


def _delete_after_commit(account_ids: list[str]) -> None:
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


# --- Reconciler: the one place unbound accounts are removed -----------------

# Composio user ids xagent derives (composio_user_id). Any other account in an
# auth config, such as one connected from Composio's dashboard, is not ours.
_XAGENT_USER_ID_PREFIX = "xagent-"
# Past the flow TTL: a connect still in progress spends its flow within the
# TTL of its start, and the account is created after that start. Binding after
# the spend takes a few bounded Composio and Graph calls; a bind still not
# committed after the grace period could lose its account to a pass.
_RECONCILE_GRACE_SECONDS = 30 * 60
_RECONCILE_PAGE_SIZE = 100


def _created_at(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _unbound_candidates(
    config: ComposioConfig, *, cutoff: float, stopping: Callable[[], bool]
) -> list[str]:
    """xagent's accounts of one auth config created before ``cutoff``.

    Follows every page and stops on a repeated cursor. A failed page ends the
    listing; what the earlier pages gave is still returned.
    """
    candidates: list[str] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while not stopping():
        query: dict[str, Any] = {
            "auth_config_ids": config.auth_config_id,
            "limit": _RECONCILE_PAGE_SIZE,
        }
        if cursor:
            query["cursor"] = cursor
        try:
            status, payload = _call(
                config.api_key, "GET", f"/connected_accounts?{urlencode(query)}"
            )
        except ComposioError:
            break
        if not 200 <= status < 300 or not isinstance(payload, dict):
            logger.warning("Composio account listing failed (status=%s)", status)
            break
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            account_id = _valid_account_id(item.get("id"))
            user_id = item.get("user_id")
            auth_config = item.get("auth_config")
            created = _created_at(item.get("created_at"))
            if (
                account_id is not None
                and isinstance(user_id, str)
                and user_id.startswith(_XAGENT_USER_ID_PREFIX)
                and isinstance(auth_config, dict)
                and auth_config.get("id") == config.auth_config_id
                and created is not None
                and created.timestamp() < cutoff
            ):
                candidates.append(account_id)
        next_cursor = payload.get("next_cursor")
        cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
        if cursor is None:
            break
        if cursor in seen_cursors:
            logger.warning("Composio account listing repeated a cursor; stopping")
            break
        seen_cursors.add(cursor)
    return candidates


def reconcile_orphan_accounts(
    db: Session,
    *,
    now: datetime | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """Delete xagent's Composio accounts that no stored grant uses.

    The connect flow deletes an account only when the request that spent its
    flow fails to bind it. Everything else that can leave an unbound account
    (a lost complete_auth answer, an expired flow cookie, a replaced link, a
    failed delete after disconnect, a user deleted with their grants) is
    cleaned up here, for every app whose Composio config is complete. Only
    accounts of an app's auth config whose Composio user id xagent derived,
    and that are older than the flow TTL plus a grace period, are considered;
    an account without a user id is never deleted. A grant of any app counts
    as a reference. The list is used for cleanup only, never for binding. A
    pass deletes nothing when a stored grant cannot be decrypted (its account
    would look unused), and stops between requests once ``stop_event`` is
    set. Returns how many were deleted.
    """
    from .models.user_oauth import UserOAuth, is_undecryptable_oauth_token

    def stopping() -> bool:
        return stop_event is not None and stop_event.is_set()

    cutoff = (now or datetime.now(timezone.utc)).timestamp() - (
        _flow_ttl_seconds() + _RECONCILE_GRACE_SECONDS
    )
    candidates: dict[str, str] = {}  # account id -> API key
    for config in _configured_apps().values():
        for account_id in _unbound_candidates(config, cutoff=cutoff, stopping=stopping):
            candidates.setdefault(account_id, config.api_key)
    if not candidates or stopping():
        return 0
    tokens = [
        row[0]
        for row in db.query(UserOAuth.access_token)
        .filter(UserOAuth.token_type == COMPOSIO_TOKEN_TYPE)
        .all()
    ]
    db.rollback()
    if any(is_undecryptable_oauth_token(token) for token in tokens):
        logger.warning(
            "Skipping Composio reconcile: a stored grant cannot be decrypted"
        )
        return 0
    referenced = {str(token) for token in tokens if token}
    deleted = 0
    for account_id, api_key in candidates.items():
        if stopping():
            break
        if account_id not in referenced and delete_account(api_key, account_id):
            deleted += 1
    if deleted:
        logger.info("Deleted %d unbound Composio accounts", deleted)
    return deleted


def reconcile_orphan_accounts_isolated(
    session_factory: Callable[[], Session] | None = None,
    *,
    stop_event: threading.Event | None = None,
) -> int:
    """Run one reconcile pass in its own short-lived Session."""
    if session_factory is None:
        from .models.database import get_session_local

        session_factory = get_session_local()
    with session_factory() as db:
        return reconcile_orphan_accounts(db, stop_event=stop_event)


async def run_reconcile_loop(
    *, interval_seconds: int, stop_event: threading.Event
) -> None:
    """Reconcile on an interval; a failed pass is logged and retried later.

    Cancelling the task does not stop a pass already handed to a worker
    thread, so shutdown sets ``stop_event`` first and the pass returns at its
    next request boundary.
    """
    import asyncio

    while not stop_event.is_set():
        try:
            await asyncio.to_thread(
                reconcile_orphan_accounts_isolated, stop_event=stop_event
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Composio reconcile pass failed (exception_type=%s)",
                type(exc).__name__,
            )
        await asyncio.sleep(interval_seconds)


# --- Browser trips -------------------------------------------------------------


@dataclass(frozen=True)
class _Flow:
    state: str
    nonce: str
    user_id: int
    owner_key: str
    redirect: str | None
    expires_at: float
    app_id: str


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


def _provider_of(app_id: str | None) -> str:
    """The app's own provider; the callback's while the app is unknown."""
    app = COMPOSIO_APPS.get(app_id) if isinstance(app_id, str) else None
    return app.provider if app is not None else UNKNOWN_APP_PROVIDER


def _log_rejection(reason: str, app_id: str | None = None) -> None:
    """``app_id`` is unknown until the signed state has been read."""
    from .api.auth import _log_oauth_callback_rejection

    _log_oauth_callback_rejection(
        _provider_of(app_id), reason, app_id=app_id, actor_flow=True
    )


def _is_expired_composio_state(state: str) -> bool:
    """Whether a genuinely signed, expired state was minted for a Composio app."""
    from .api.auth import _expired_actor_oauth_state

    return any(
        expired is not None and expired.get(AUTH_BACKEND_CLAIM) == AUTH_BACKEND
        for expired in (
            _expired_actor_oauth_state(state, provider)
            for provider in {app.provider for app in COMPOSIO_APPS.values()}
        )
    )


def _verified_flow(state: str, request: Any, db: Session) -> _Flow:
    """Run every check that does not need Composio. Raises ``_Rejected``."""
    from .api.auth import (
        _MAX_USER_ID,
        ActorOAuthFlowRejected,
        check_actor_oauth_flow,
        verify_token,
    )

    payload = verify_token(state)
    if payload is None:
        if _is_expired_composio_state(state):
            raise _reject("state_expired")
        raise _reject("state_invalid")
    nonce = payload.get("actor_flow_nonce")
    user_id = payload.get("user_id")
    expires_at = payload.get("exp")
    app_id = payload.get("app_id")
    # The Composio-specific claims; the shared actor checks follow.
    if (
        payload.get("type") != "oauth_state"
        or payload.get(AUTH_BACKEND_CLAIM) != AUTH_BACKEND
        or not isinstance(app_id, str)
        or app_id not in COMPOSIO_APPS
        # A state is spent only on a flow of the provider it was minted for.
        or payload.get("provider") != COMPOSIO_APPS[app_id].provider
        or type(user_id) is not int
        or not 0 < user_id <= _MAX_USER_ID
        or not isinstance(expires_at, (int, float))
    ):
        raise _reject("claims_invalid")
    try:
        owner_key = check_actor_oauth_flow(
            db,
            request,
            provider=COMPOSIO_APPS[app_id].provider,
            user_id=user_id,
            app_id=app_id,
            nonce=nonce,
            owner_claim=payload.get("resource_owner_key"),
        )
    except ActorOAuthFlowRejected as exc:
        raise _reject(exc.reason) from None
    redirect = payload.get("redirect")
    return _Flow(
        state=state,
        nonce=str(nonce),
        user_id=user_id,
        owner_key=owner_key,
        redirect=redirect if isinstance(redirect, str) else None,
        expires_at=float(expires_at),
        app_id=app_id,
    )


def _flow_row_query(db: Session, nonce: str) -> Any:
    from .models.actor_oauth_flow import ActorOAuthFlowState

    return db.query(ActorOAuthFlowState).filter(
        ActorOAuthFlowState.nonce == nonce,
        ActorOAuthFlowState.expires_at > datetime.now(timezone.utc),
    )


def _cookie_scope(callback: str) -> tuple[bool, str]:
    from .api.auth import _oauth_cookie_scope_for

    return _oauth_cookie_scope_for(callback)


def _flow_cookie_name(nonce: str) -> str:
    return f"{FLOW_COOKIE_PREFIX}{nonce[:24]}"


def _flow_ttl_seconds() -> int:
    from .api.auth import _ACTOR_OAUTH_FLOW_TTL

    return int(_ACTOR_OAUTH_FLOW_TTL.total_seconds())


def _encode_flow_cookie(state: str, account_id: str) -> str:
    from ..core.utils.encryption import get_cipher

    # Fernet already authenticates the value and dates it for the TTL check.
    return get_cipher().encrypt(f"{account_id} {state}".encode()).decode()


def _decode_flow_cookie(value: str) -> tuple[str, str] | None:
    from ..core.utils.encryption import get_cipher

    try:
        plain = get_cipher().decrypt(value.encode(), ttl=_flow_ttl_seconds()).decode()
    except (InvalidToken, ValueError, TypeError):
        return None
    account_id, _, state = plain.partition(" ")
    if _valid_account_id(account_id) is None or not state:
        return None
    return state, account_id


@dataclass(frozen=True)
class _OpenFlow:
    """A flow this browser holds, checked except for its Composio outcome."""

    flow: _Flow
    account_id: str
    cookie_name: str

    @property
    def actor(self) -> tuple[int, str]:
        return self.flow.user_id, self.flow.owner_key


@dataclass
class _BrowserFlows:
    """What the flow cookies a browser sent amount to."""

    open: list[_OpenFlow]
    # Cookie names to clear: their flow is spent, expired or not genuine.
    dead: list[str]
    # Why each dead flow failed its checks.
    reasons: list[str]


def _open_flows(request: Any, db: Session) -> _BrowserFlows:
    """Check every flow cookie the browser sent, without spending any flow."""
    open_flows: list[_OpenFlow] = []
    dead: list[str] = []
    reasons: list[str] = []
    for name, value in sorted(request.cookies.items()):
        if not name.startswith(FLOW_COOKIE_PREFIX):
            continue
        decoded = _decode_flow_cookie(value) if isinstance(value, str) else None
        if decoded is None:
            dead.append(name)
            reasons.append("cookie_mismatch")
            continue
        state, account_id = decoded
        try:
            flow = _verified_flow(state, request, db)
            if name != _flow_cookie_name(flow.nonce):
                raise _reject("cookie_mismatch")
            if _flow_row_query(db, flow.nonce).count() != 1:
                raise _reject("flow_used_or_expired")
        except _Rejected as rejected:
            dead.append(name)
            reasons.append(rejected.reason)
            continue
        open_flows.append(_OpenFlow(flow, account_id, name))
    db.rollback()
    return _BrowserFlows(open_flows, dead, reasons)


def _clear_flow_cookies(
    response: Response, callback: str, names: Iterable[str]
) -> Response:
    secure, path = _cookie_scope(callback)
    for name in dict.fromkeys(names):
        response.delete_cookie(
            name, path=path, secure=secure, httponly=True, samesite="lax"
        )
    return response


def handle_callback(
    step: Step, request: Any, db: Session, meta_row: Any = None
) -> Response:
    """Handle one Composio trip; ``meta_row`` is the Meta provider row when the
    trip arrived on the Meta callback."""
    try:
        callback = shared_callback_url(db, meta_row)
    except ValueError:
        logger.error("%s is not a valid callback URL", COMPOSIO_CALLBACK_URL_ENV)
        _log_rejection("callback_url_invalid")
        return _unavailable_page()
    if step == "begin":
        return _handle_begin(request, db, callback)
    if step == "denied":
        return _handle_denied(request, db, callback)
    return _handle_verify(request, db, callback)


def handle_dedicated_callback(request: Any, db: Session) -> Response:
    """``/api/auth/composio/callback``: only Composio trips are accepted."""
    step = callback_step(request.query_params)
    if step is None:
        _log_rejection("not_a_composio_trip")
        return _reject("claims_invalid").response
    return handle_callback(step, request, db)


def _too_many_flows_page(app_id: str) -> HTMLResponse:
    # The limit counts all of the actor's Composio windows, whatever the app.
    label = COMPOSIO_APPS[app_id].label
    return _page(
        "Too many sign-in windows are open",
        f"{label} was not connected.",
        "Close the other sign-in windows, or wait ten minutes, and then "
        "select Connect again.",
        status_code=400,
    )


def _handle_begin(request: Any, db: Session, callback: str) -> Response:
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

    # Only one actor's flows stay open in a browser, so a verification can be
    # completed with the one Composio user id they share. Another actor's
    # flows are dropped: their sessions then fail complete_auth's user check
    # and are never activated.
    browser_flows = _open_flows(request, db)
    existing = browser_flows.open
    stale = browser_flows.dead + [
        open_flow.cookie_name
        for open_flow in existing
        if open_flow.actor != (flow.user_id, flow.owner_key)
        or open_flow.flow.nonce == flow.nonce
    ]
    kept = [open_flow for open_flow in existing if open_flow.cookie_name not in stale]
    if len(kept) >= _MAX_OPEN_FLOWS:
        _log_rejection("too_many_open_flows", flow.app_id)
        return _clear_flow_cookies(_too_many_flows_page(flow.app_id), callback, stale)

    config = load_config(flow.app_id)
    if config is None:
        _log_rejection("composio_not_configured", flow.app_id)
        return _unavailable_page(flow.app_id)
    try:
        redirect_url, account_id = create_link(
            config,
            user_id=composio_user_id(
                config, user_id=flow.user_id, owner_key=flow.owner_key
            ),
            callback_url=callback,
        )
    except ComposioError as exc:
        _log_rejection(exc.reason, flow.app_id)
        return _unavailable_page(flow.app_id)
    # A reloaded begin replaces this flow's link. The previous link is not
    # deleted here: the original window may be completing it right now. If it
    # ends up unbound, the reconciler removes it.

    response = RedirectResponse(redirect_url, status_code=302)
    # Our URL carries the state; Composio's page must not receive it.
    response.headers["Cache-Control"] = "no-store"
    response.headers["Referrer-Policy"] = "no-referrer"
    _clear_flow_cookies(response, callback, stale)
    secure, path = _cookie_scope(callback)
    response.set_cookie(
        _flow_cookie_name(flow.nonce),
        _encode_flow_cookie(flow.state, account_id),
        max_age=max(1, min(_flow_ttl_seconds(), int(flow.expires_at - time.time()))),
        secure=secure,
        httponly=True,
        samesite="lax",
        path=path,
    )
    return response


def _handle_denied(request: Any, db: Session, callback: str) -> Response:
    """Say what to do after Composio returns a denied or failed authorization.

    The returned id is trusted only to find the open flow whose encrypted
    cookie names the same account; that flow is ended. Its unfinished account
    is left to the reconciler (``reconcile_orphan_accounts``).
    """
    from .api.auth import (
        _actor_oauth_provider_error_response,
        _log_oauth_callback_rejection,
        _oauth_callback_provider_error,
    )

    query = request.query_params
    browser_flows = _open_flows(request, db)
    dead = browser_flows.dead
    returned = _valid_account_id(query.get("connected_account_id"))
    matches = [f for f in browser_flows.open if returned and f.account_id == returned]
    app_ids = sorted({match.flow.app_id for match in matches})
    log_app = app_ids[0] if len(app_ids) == 1 else None
    for match in matches:
        dead.append(match.cookie_name)
        _spend(db, match)
    if query.get("status") == "success":
        # Composio activated the connection without handing it to us: its
        # project has no OAuth user verification pointed at this callback.
        # Only a return for one of this browser's own flows says so; anyone
        # can send the bare query, so that is logged at warning level only.
        (logger.error if matches else logger.warning)(
            "Composio returned a connection without verification; set the "
            "project's OAuth user verification to this callback URL"
        )
        _log_rejection("verification_not_configured", log_app)
        return _clear_flow_cookies(_unavailable_page(*app_ids), callback, dead)
    provider = _provider_of(log_app)
    provider_error = _oauth_callback_provider_error(
        provider, query.get("error") or "access_denied", None
    )
    assert provider_error is not None  # a non-empty error is passed in
    _log_oauth_callback_rejection(
        provider,
        "provider_error",
        app_id=log_app,
        actor_flow=True,
        provider_error=provider_error,
    )
    return _clear_flow_cookies(
        _actor_oauth_provider_error_response(provider, provider_error),
        callback,
        dead,
    )


def _spend(db: Session, open_flow: _OpenFlow) -> bool:
    """Spend one flow; False when it was spent or expired meanwhile."""
    deleted = _flow_row_query(db, open_flow.flow.nonce).delete(
        synchronize_session=False
    )
    if deleted != 1:
        db.rollback()
        return False
    db.commit()
    return True


def _handle_verify(request: Any, db: Session, callback: str) -> Response:
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

    # The session_uri names no flow, so every flow this browser holds is
    # checked first; complete_auth's answer then says which one it was.
    browser_flows = _open_flows(request, db)
    open_flows, dead = browser_flows.open, browser_flows.dead
    if not open_flows:
        reasons = browser_flows.reasons
        reason = reasons[-1] if reasons else "cookie_missing"
        _log_rejection(reason)
        return _clear_flow_cookies(_reject(reason).response, callback, dead)
    actors = {open_flow.actor for open_flow in open_flows}
    if len(actors) != 1:
        # begin keeps one actor per browser; anything else is not completed.
        _log_rejection("claims_invalid")
        return _clear_flow_cookies(_reject("claims_invalid").response, callback, dead)

    # A browser can hold flows of several apps for the same actor at once
    # (Facebook and Instagram, say). They share the project's API key and the
    # Composio user id, so the session
    # is completed with any configured app before knowing which app it was;
    # the matched flow's own app then decides the auth config, toolkit and
    # identity checked. One app's missing config never blocks the other's.
    app_ids = sorted({open_flow.flow.app_id for open_flow in open_flows})
    log_app = app_ids[0] if len(app_ids) == 1 else None
    configs = _configured_apps()
    if not configs:
        _log_rejection("composio_not_configured", log_app)
        return _clear_flow_cookies(_unavailable_page(*app_ids), callback, dead)
    shared = next(iter(configs.values()))
    user_id, owner_key = actors.pop()
    composio_user = composio_user_id(shared, user_id=user_id, owner_key=owner_key)
    try:
        account_id = complete_auth(
            shared, session_uri=session_uri, user_id=composio_user
        )
    except ComposioError as exc:
        _log_rejection(exc.reason, log_app)
        if exc.rejected:
            # Nothing was activated, so every open flow stays usable.
            return _clear_flow_cookies(
                _reject("composio_rejected").response, callback, dead
            )
        # Composio may have activated an account before the answer was lost;
        # which one cannot be told here. Nothing is spent or deleted: an
        # account that ends up unbound is removed by the reconciler.
        return _clear_flow_cookies(_unavailable_page(*app_ids), callback, dead)

    matches = [
        open_flow for open_flow in open_flows if open_flow.account_id == account_id
    ]
    if not matches:
        # No open flow created it (for example, its cookie expired while the
        # member was signing in). It is not bound here; if it is ours and
        # stays unbound, the reconciler removes it.
        logger.warning("Composio completed a connection no open flow created")
        _log_rejection("account_invalid", log_app)
        return _clear_flow_cookies(
            _reject("composio_rejected").response, callback, dead
        )
    match = matches[0]
    app_id = match.flow.app_id
    dead.append(match.cookie_name)
    if not _spend(db, match):
        # Spent by another request (which binds it) or expired meanwhile (the
        # reconciler then removes the unbound account). Never delete here.
        _log_rejection("flow_used_or_expired", app_id)
        return _clear_flow_cookies(
            _reject("flow_used_or_expired").response, callback, dead
        )
    # This request spent the flow that created the account, so the account is
    # its own: if it cannot be bound, it is deleted rather than left unbound.
    config = configs.get(app_id)
    if config is None:
        # This app's config went away after its flow began.
        _log_rejection("composio_not_configured", app_id)
        delete_account(shared.api_key, account_id)
        return _clear_flow_cookies(_unavailable_page(app_id), callback, dead)
    try:
        verify_account(
            config,
            account_id=account_id,
            user_id=composio_user,
            toolkit_slug=COMPOSIO_APPS[app_id].toolkit_slug,
        )
        provider_user_id = fetch_provider_user_id(
            config, account_id=account_id, user_id=composio_user, app_id=app_id
        )
        stored = _store_grant(
            db,
            flow=match.flow,
            account_id=account_id,
            composio_user=composio_user,
            provider_user_id=provider_user_id,
        )
    except ComposioError as exc:
        _log_rejection(exc.reason, app_id)
        delete_account(config.api_key, account_id)
        response = (
            _reject("composio_rejected").response
            if exc.rejected
            else _unavailable_page(app_id)
        )
        return _clear_flow_cookies(response, callback, dead)
    except _Rejected as rejected:
        _log_rejection(rejected.reason, app_id)
        delete_account(config.api_key, account_id)
        return _clear_flow_cookies(rejected.response, callback, dead)
    except Exception as exc:
        db.rollback()
        logger.warning(
            "Composio %s grant could not be stored (exception_type=%s)",
            COMPOSIO_APPS[app_id].label,
            type(exc).__name__,
        )
        delete_account(config.api_key, account_id)
        return _clear_flow_cookies(
            HTMLResponse(
                content=(
                    "<h1>Authentication Failed</h1><p>An unexpected error "
                    "occurred while connecting this account. Please try "
                    "again.</p>"
                ),
                status_code=500,
            ),
            callback,
            dead,
        )
    return _clear_flow_cookies(stored, callback, dead)


def _store_grant(
    db: Session,
    *,
    flow: _Flow,
    account_id: str,
    composio_user: str,
    provider_user_id: str,
) -> Response:
    from .api.auth import _lock_actor_link, _oauth_success_response
    from .models.user import User
    from .models.user_oauth import UserOAuth
    from .services.user_oauth import delete_scoped_user_oauth_accounts

    try:
        # Composio I/O can overlap a personal disconnect.
        _lock_actor_link(
            db,
            user_id=flow.user_id,
            provider=COMPOSIO_APPS[flow.app_id].provider,
            app_id=flow.app_id,
        )
    except ValueError:
        db.rollback()
        raise _reject("link_changed_after_exchange") from None
    db.query(User.id).filter(User.id == flow.user_id).with_for_update().one()
    # Replaces this actor's previous grant for the app; a previous Composio
    # account is deleted once this commits.
    delete_scoped_user_oauth_accounts(
        db,
        user_id=flow.user_id,
        resource_owner_key=flow.owner_key,
        providers=[flow.app_id],
    )
    db.add(
        UserOAuth(
            user_id=flow.user_id,
            provider=flow.app_id,
            resource_owner_key=flow.owner_key,
            provider_user_id=provider_user_id,
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
        redirect_url=flow.redirect, email=None, provider_label=flow.app_id
    )
