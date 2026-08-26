"""Authentication API endpoints"""

import asyncio
import base64
import hashlib
import html
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, cast

import requests

# Relax token scope verification as Google might add extra scopes (like openid)
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from jose import JWTError, jwt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
)
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from ...config import get_app_base_url, get_password_reset_expire_minutes
from ...core.agent.voice_policy import VALID_VOICES as _CORE_VALID_VOICES
from ..auth_config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    PASSWORD_MIN_LENGTH,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from ..auth_dependencies import get_current_user
from ..models.database import get_db, get_session_local, release_db_connection_if_clean
from ..models.system_setting import SystemSetting
from ..models.user import User
from ..models.user_oauth import UserOAuth
from ..oauth_provider_quirks import (
    matches_provider_family,
    requires_json_accept_header,
    requires_pkce,
)
from ..services import gmail_provisioning
from ..services.auth_email import send_password_reset_email
from ..services.db_runtime import await_task_settlement, propagate_deferred_cancellation
from ..services.user_oauth import delete_scoped_user_oauth_accounts
from ..utils.graphql_errors import graphql_errors_message, truncate_error_text

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])

REGISTRATION_ENABLED_SETTING_KEY = "registration_enabled"
SETUP_COMPLETED_SETTING_KEY = "setup_completed"
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_USER_ID = 2_147_483_647


def _best_effort_ensure_gmail_watches_for_user(db: Session, *, user_id: int) -> None:
    """Provision Gmail watches without letting the outcome reach the caller.

    This runs after the OAuth token is committed, so the connect has already
    succeeded. Anything raised here would be rendered by the callback's outer
    handler as an ``Authentication Failed`` 500, telling the user a connector
    failed while it is in fact connected. A best-effort side effect must not
    be able to change what the callback returns.
    """
    # The module is imported at module level so that a genuine module
    # resolution failure surfaces at startup instead of being logged here as a
    # routine provisioning warning. The call stays inside the guard, and going
    # through the module keeps the attribute lookup late so tests can patch it.
    try:
        gmail_provisioning.best_effort_provision_gmail_watches_for_user(
            db,
            user_id=user_id,
            context="after OAuth callback",
        )
    except Exception:
        logger.warning(
            "Best-effort Gmail watch provisioning failed for user %s "
            "after OAuth callback",
            user_id,
            exc_info=True,
        )


def _run_post_commit_oauth_side_effects(
    db: Session, *, user_id: int, connector_key: str
) -> None:
    """Run the OAuth callback's post-commit work; never raise.

    ``connector_key`` is the value persisted as ``UserOAuth.provider``: the app
    id when the connect is app-scoped (``"gmail"``), otherwise the provider
    name (``"google"``).

    ``user_id`` is a validated ``oauth_state`` claim. The callback validates
    this claim before the provider exchange and database changes.

    Everything here runs once ``db.commit()`` has persisted the OAuth token, so
    the connect has already succeeded as far as the user is concerned. The
    callback's outer ``except Exception`` would render anything raised here as
    an ``Authentication Failed`` 500, reporting a failure for a connector that
    is in fact connected. That is the bug #1150 reproduced on staging.

    Guarding the region rather than each individual call is what makes that
    property structural: a side effect added here inherits it, instead of
    reintroducing #1150 whenever someone forgets to wrap their own call. The
    inner guards each side effect carries are kept as well, deliberately, so
    that neither layer is load-bearing on its own.

    One guard over the whole region does couple the side effects to each other:
    a raiser aborts the ones after it. That is why the inner guards matter and
    are worth keeping per side effect. Anything added here that must run even
    when an earlier entry fails needs its own guard, exactly as Gmail
    provisioning has.

    Response construction stays outside this region on purpose. It also runs
    after the commit, but a failure there leaves no response to return, so it
    must keep reaching the outer handler rather than being swallowed here.

    Failures are logged only. That is acceptable while every side effect in
    this region has its own recovery path: Gmail watches are re-provisioned by
    ``scan_due_gmail_watch_renewals``. A side effect that a swallowed failure
    would strand needs a user-visible signal instead.
    """
    try:
        if connector_key == "gmail":
            _best_effort_ensure_gmail_watches_for_user(db, user_id=user_id)
    except Exception:
        logger.warning(
            "Post-commit OAuth side effects failed for user %s on connector %s; "
            "the connect itself succeeded",
            user_id,
            connector_key,
            exc_info=True,
        )


def _oauth_env_name(provider: str, suffix: str) -> str:
    return f"{provider.upper()}_{suffix}"


def _is_salesforce_provider(provider: str) -> bool:
    """Match the Salesforce family, including admin-created sandbox rows.

    A prefix match, not exact equality: example.env's documented sandbox
    workaround is an admin hand-creating a second provider row (e.g.
    "salesforce-sandbox") pointing at test.salesforce.com, since the
    provider-row model has no per-user sandbox toggle. Every Salesforce-only
    code path below -- the instance_url presence guard and the
    provider_user_id identity backfill -- must use this same predicate; an
    exact match on either would silently grant that row the capability while
    skipping its safeguard.

    PKCE is gated separately, by requires_pkce() (oauth_provider_quirks.py):
    that predicate also covers Employment Hero, so it is no longer
    Salesforce-only and must not be conflated with this one. Both share the
    same underlying family-match algorithm (matches_provider_family below),
    just applied to a different provider set -- see that function's
    docstring for the "-"-anchored-prefix reasoning this predicate relies on.
    """
    return matches_provider_family(provider, "salesforce")


def _resolve_oauth_secret(
    provider: str, encrypted_value: Optional[str], env_suffix: str
) -> str:
    from ...core.utils.encryption import decrypt_value

    if encrypted_value:
        value = decrypt_value(encrypted_value)
        if value:
            return value
    return os.environ.get(_oauth_env_name(provider, env_suffix), "")


def _resolve_oauth_redirect_uri(provider: str, db_provider: Any) -> str:
    if getattr(db_provider, "redirect_uri", None):
        return str(db_provider.redirect_uri)
    return os.environ.get(
        _oauth_env_name(provider, "REDIRECT_URI"),
        f"http://localhost:8000/api/auth/{provider}/callback",
    )


def _oauth_provider_config_error(
    provider: str, missing_env_names: list[str]
) -> HTMLResponse:
    escaped_provider = html.escape(provider)
    escaped_missing = html.escape(", ".join(missing_env_names))
    return HTMLResponse(
        content=(
            "<h1>Error: OAuth provider not configured</h1>"
            f"<p>Missing {escaped_missing} for provider {escaped_provider}.</p>"
            "<p>Configure the provider in admin settings or set the corresponding "
            "environment variables and restart the backend.</p>"
        ),
        status_code=500,
    )


def _merge_oauth_scopes(
    default_scopes: list[str] | None, app_scopes: list[str] | None
) -> list[str]:
    """Preserve provider default order and sort app scopes for deterministic URLs."""
    scopes: list[str] = []
    seen: set[str] = set()

    for scope in default_scopes or []:
        if scope and scope not in seen:
            scopes.append(scope)
            seen.add(scope)

    for scope in sorted(app_scopes or []):
        if scope and scope not in seen:
            scopes.append(scope)
            seen.add(scope)

    return scopes


def _oauth_scope_separator(provider: str) -> str:
    # Linear's authorize endpoint documents scope as "a comma separated list
    # of scopes" -- unlike most providers here, which accept a space-joined
    # list.
    if provider.lower() in ("meta", "linear"):
        return ","
    return " "


def _meta_login_config_id() -> str:
    return os.environ.get("META_CONFIG_ID", "")


def _exchange_meta_long_lived_token(
    provider: str,
    token_url: str,
    token_data: dict[str, Any],
    client_id: str,
    client_secret: str,
) -> dict[str, Any]:
    if provider.lower() != "meta":
        return token_data

    access_token = token_data.get("access_token")
    if not access_token:
        return token_data

    try:
        response = requests.get(
            token_url,
            params={
                "grant_type": "fb_exchange_token",
                "client_id": client_id,
                "client_secret": client_secret,
                "fb_exchange_token": access_token,
            },
            timeout=10.0,
        )
        long_lived_token_data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning(
            "Meta long-lived token exchange failed; using short-lived token: %s",
            exc,
        )
        return token_data

    if (
        response.status_code != 200
        or not isinstance(long_lived_token_data, dict)
        or "error" in long_lived_token_data
    ):
        reason_parts = []
        if response.status_code != 200:
            reason_parts.append(f"status={response.status_code}")
        if not isinstance(long_lived_token_data, dict):
            reason_parts.append(f"body_type={type(long_lived_token_data).__name__}")
        else:
            error = long_lived_token_data.get("error")
            if error:
                reason_parts.append(f"error={error}")
        reason = ", ".join(reason_parts) or "unknown reason"
        logger.warning(
            "Meta long-lived token exchange returned unusable response; "
            "using short-lived token: %s",
            reason,
        )
        return token_data

    return {**token_data, **long_lived_token_data}


def _normalize_intercom_token_response(
    provider: str, token_data: dict[str, Any]
) -> dict[str, Any]:
    """Map Intercom's `{"token": ...}` token response onto `access_token`.

    Intercom's token endpoint (POST /auth/eagle/token) does not follow the
    standard OAuth2 token response shape: it returns only `{"token": "..."}`,
    with no `access_token`, `token_type`, or `expires_in` fields. Without this,
    generic_oauth_callback's `token_data.get("access_token")` would silently
    resolve to None and the connection would be persisted with no token.
    """
    if provider.lower() != "intercom":
        return token_data
    if "access_token" in token_data:
        return token_data
    token = token_data.get("token")
    if not token:
        return token_data
    return {**token_data, "access_token": token}


# One shared cap for every provider-supplied error detail echoed to the
# browser -- both renderers below default to it so the two error shapes
# (standard error/error_description, Intercom's error.list) can't silently
# diverge by someone tuning one magic default and not the other.
_OAUTH_ERROR_MESSAGE_LIMIT = 500

# Diagnostic fields safe to log verbatim from a token-endpoint response --
# everything else (access_token, refresh_token, client_secret, and any
# provider-specific field that might carry a token) is reduced to presence
# only. Server-side logging isn't a safe place for the raw payload either:
# hide_parameters isn't a substitute here since these are dict values being
# logged directly, not SQL bound parameters. Applied recursively (see
# _redact_oauth_log_value), so "message"/"type" also cover an allowlisted
# key's own nested object shape (Meta's "error" is itself
# {"message": ..., "type": ...}, the same shape _bounded_oauth_error_message
# already extracts from for the browser-facing message).
_OAUTH_LOG_SAFE_KEYS = frozenset(
    {"error", "error_description", "error_uri", "reason", "type", "message"}
)


def _redact_oauth_log_value(value: Any) -> Any:
    """Recursively apply _OAUTH_LOG_SAFE_KEYS to a value found under an
    already-allowlisted key.

    A first version of this redaction serialized an allowlisted key's
    whole value verbatim (e.g. via json.dumps) once it wasn't a plain str.
    That reopened the exact leak this function exists to close: a
    malformed/adversarial response can nest a live secret *inside* an
    allowlisted key's object (`{"error": {"access_token": "..."}}`), and a
    verbatim dump would still put it in the log even though the top-level
    `access_token` field is correctly redacted. Recursing with the same
    allowlist at every level closes that regardless of nesting depth.
    """
    if isinstance(value, dict):
        return {
            key: (
                _redact_oauth_log_value(nested)
                if key in _OAUTH_LOG_SAFE_KEYS
                else "<redacted>"
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_redact_oauth_log_value(item) for item in value]
    if isinstance(value, str):
        return value[:_OAUTH_ERROR_MESSAGE_LIMIT]
    return value


def _redact_oauth_log_payload(token_data: Dict[str, Any]) -> Dict[str, Any]:
    """Project a provider token/error response down to a safe-to-log shape.

    Only the allowlisted diagnostic fields keep their (recursively
    redacted, length-capped) value; every other key is replaced with a
    presence marker so a malformed or partial response can't put a live
    access/refresh token into the server log the way logging the raw dict
    would.

    An allowlisted key's value can still be non-str -- Meta's "error" is
    itself an object (`{"message": ..., "type": ...}`), the same shape
    `_bounded_oauth_error_message` already handles for the browser-facing
    message -- so this keeps that diagnostic content instead of blanking
    it, but via the same recursive allowlist rather than a verbatim dump.
    """
    return {
        key: (
            _redact_oauth_log_value(value)
            if key in _OAUTH_LOG_SAFE_KEYS
            else "<redacted>"
        )
        for key, value in token_data.items()
    }


def _extract_provider_error_message(
    token_data: dict[str, Any], *, limit: int = _OAUTH_ERROR_MESSAGE_LIMIT
) -> str | None:
    """Best-effort human-readable detail for a token exchange that yielded no
    access_token.

    Standard OAuth2 providers surface `error`/`error_description`, already
    handled by the `"error" in token_data` check earlier in the callback.
    This covers Intercom's differently-shaped `error.list` envelope instead
    (`{"type": "error.list", "errors": [{"message": "..."}]}`), which does
    not use an `error` key and so slips past that earlier check. Capped the
    same way as `_bounded_oauth_error_message`'s standard-shape sibling --
    this value is echoed to the browser too, and an unbounded provider
    message is exactly the risk that helper was added to close.
    """
    errors = token_data.get("errors")
    if isinstance(errors, list) and errors:
        first_error = errors[0]
        if isinstance(first_error, dict) and first_error.get("message"):
            return str(first_error["message"])[:limit]
    return None


def _bounded_oauth_error_message(
    token_data: Dict[str, Any], *, limit: int = _OAUTH_ERROR_MESSAGE_LIMIT
) -> str:
    """Bounded, allowlisted, HTML-escaped rendering of a token-endpoint
    error response for a `token_data["error"]` payload.

    Echoing `str(token_data)` in full (as this used to) was safe only by
    accident: it relied on no provider's error payload ever containing a
    token. Making GitHub's JSON error path reachable here (via the
    provider-quirk Accept header) removed that accidental guarantee, so
    only the standard OAuth2 `error`/`error_description` fields are
    rendered now, with every other field dropped and the result capped in
    length.

    `error` is a bare string for standard OAuth2 providers, but Meta's is
    itself an object (`{"message": ..., "type": "OAuthException", ...}`)
    -- str()-ing that directly would render a Python dict repr into the
    page instead of the actual message. `error_description` is likewise
    absent from Zoom's error shape, which instead carries the
    human-readable detail in a `reason` key.
    """
    raw_error = token_data.get("error")
    if isinstance(raw_error, dict):
        error = str(
            raw_error.get("message") or raw_error.get("type") or "unknown_error"
        )
    else:
        error = str(raw_error or "unknown_error")
    description = token_data.get("error_description") or token_data.get("reason")
    if not isinstance(description, str):
        # Only a plain-string description is rendered -- an object-valued
        # one would repr a Python dict into the page, the same defect the
        # dict-`error` branch above exists to prevent.
        description = None
    message = error if not description else f"{error}: {description}"
    return html.escape(message[:limit])


def _fetch_linear_viewer_identity(
    access_token: str,
) -> tuple[Optional[str], Optional[str]]:
    """Linear has no flat REST userinfo endpoint (GraphQL-only), so identity
    comes from a `viewer` query against the same GraphQL endpoint the
    connector's tools use, instead of the generic `userinfo_url` REST GET
    path below (left empty for Linear's provider row).

    This doubles as the post-exchange token verification every other
    provider gets for free from its REST userinfo call: a token Linear
    won't honour is caught here and reported, instead of being persisted
    as healthy and failing opaquely later from inside a tool call.

    Raises RuntimeError with a human-readable message on any failure, so
    the callback can report it the same way the Slack-style `ok: false`
    branch below does, rather than silently connecting.
    """
    response = requests.post(
        "https://api.linear.app/graphql",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={"query": "query { viewer { id email } }"},
        timeout=10.0,
    )

    def _payload_errors_message(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        errors = payload.get("errors")
        return graphql_errors_message(errors) if errors else None

    if response.status_code != 200:
        # Mirrors linear.py's _graphql(): prefer the structured GraphQL
        # "errors" shape, but fall back to the raw (truncated) body for any
        # other error shape (a differently-keyed JSON error, or an HTML
        # gateway/WAF page on a 502/504) rather than discarding it.
        detail = None
        try:
            detail = _payload_errors_message(response.json())
        except ValueError:
            pass
        if detail is None:
            detail = truncate_error_text(response.text.strip(), limit=500)
        raise RuntimeError(
            f"Linear API error (status {response.status_code})"
            + (f": {detail}" if detail else "")
        )
    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError(
            "Linear API returned a non-JSON response: "
            f"{truncate_error_text(response.text.strip(), limit=500)}"
        ) from None
    if not isinstance(payload, dict):
        raise RuntimeError("Linear API returned an unexpected response body")
    data = payload.get("data")
    viewer = data.get("viewer") if isinstance(data, dict) else None
    if not isinstance(viewer, dict):
        raise RuntimeError(
            _payload_errors_message(payload) or "Linear did not return a viewer"
        )
    return viewer.get("id"), viewer.get("email")


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    if "type" not in to_encode:
        to_encode["type"] = "access"
    encoded_jwt: str = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token with longer expiry"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt: str = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_refresh_token(token: str) -> Optional[dict[str, Any]]:
    """Verify JWT refresh token and return payload"""
    try:
        payload: dict[str, Any] = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
        )
        if payload.get("type") != "refresh":
            return None
        return payload
    except JWTError:
        return None


def verify_token(token: str) -> Optional[dict[str, Any]]:
    """Verify JWT token and return payload"""
    try:
        payload: dict[str, Any] = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None


class LoginRequest(BaseModel):
    """Login request model"""

    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model"""

    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    user_id: Optional[int] = None
    expires_in: Optional[int] = None
    refresh_expires_in: Optional[int] = None


class RegisterRequest(BaseModel):
    """User registration request model"""

    username: str
    email: Optional[str] = None
    password: str


class RegisterResponse(BaseModel):
    """User registration response model"""

    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None


class SetupStatusResponse(BaseModel):
    initialized: bool
    needs_setup: bool
    registration_enabled: bool


class RegisterSwitchRequest(BaseModel):
    enabled: bool


class RegisterSwitchResponse(BaseModel):
    success: bool
    registration_enabled: bool
    message: str


class ChangePasswordRequest(BaseModel):
    """Change password request model"""

    current_password: str
    new_password: str


class ChangePasswordResponse(BaseModel):
    """Change password response model"""

    success: bool
    message: str


class UserProfileResponse(BaseModel):
    success: bool
    message: str
    user: Dict[str, Any]


class UpdateEmailRequest(BaseModel):
    email: str


class UpdateEmailResponse(BaseModel):
    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None


# The 5 voice options the onboarding "Launch" step offers - each agent's
# system prompt gets a short instruction derived from whichever one the
# user picked (see apply_user_voice in api/agents.py). Re-exported from
# core.agent.voice_policy (the canonical source, not redeclared here) so
# this and _VOICE_INSTRUCTIONS there can never drift into two
# independently-maintained copies of the same 5-string set.
VALID_USER_VOICES = _CORE_VALID_VOICES

# Onboarding's "About you"/"Goals" steps are short free-text labels and a
# handful of goal picks, not open-ended documents - these mirror the
# max_length this codebase already puts on comparably-scoped free-text
# fields (e.g. agents.py's/custom_api.py's name/description Fields).
# Without a bound, the full preferences JSON is replayed through every
# login/`/me`/email-update/token-validation/PATCH response, so an
# unbounded field lets a user inflate all of those response bodies.
PREFERENCES_TEXT_FIELD_MAX_LENGTH = 200
PREFERENCES_GOALS_MAX_ITEMS = 20


class UpdatePreferencesRequest(BaseModel):
    """Partial update for the current user's onboarding/voice preferences.
    Only fields actually present in the request body are merged into the
    stored dict (see exclude_unset=True below) - onboarding writes these
    incrementally, one step at a time, not all at once."""

    # Pydantic's default (extra="ignore") would validate a typo'd key
    # (e.g. "voce") to an empty, all-unset model: model_dump(exclude_unset=True)
    # then returns {}, so the PATCH silently skips persistence and cache
    # invalidation while still reporting success - forbid so a typo/unknown
    # key is a 422, not a lost write the client believes succeeded.
    model_config = ConfigDict(extra="forbid")

    onboarded: Optional[StrictBool] = None
    department: Optional[str] = Field(
        default=None, max_length=PREFERENCES_TEXT_FIELD_MAX_LENGTH
    )
    industry: Optional[str] = Field(
        default=None, max_length=PREFERENCES_TEXT_FIELD_MAX_LENGTH
    )
    voice: Optional[str] = None
    goals: Optional[
        List[Annotated[str, Field(max_length=PREFERENCES_TEXT_FIELD_MAX_LENGTH)]]
    ] = Field(default=None, max_length=PREFERENCES_GOALS_MAX_ITEMS)

    @field_validator("voice")
    @classmethod
    def _validate_voice(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in VALID_USER_VOICES:
            raise ValueError(f"voice must be one of {sorted(VALID_USER_VOICES)}")
        return value

    # A blank string is meaningless stored data, not a "clear this field"
    # signal - a merge-style PATCH already has one for that (send `null`
    # for the key, same as tokens_must_not_be_blank rejects a blank
    # access/refresh token above rather than treating it as "no token").
    @field_validator("department", "industry")
    @classmethod
    def _reject_blank_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped

    @field_validator("goals")
    @classmethod
    def _reject_blank_goals(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        stripped_goals = [item.strip() for item in value]
        if any(not item for item in stripped_goals):
            raise ValueError("goal must not be blank")
        return stripped_goals


class UpdatePreferencesResponse(BaseModel):
    success: bool
    message: str
    user: Optional[Dict[str, Any]] = None


class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""

    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """Refresh token response model"""

    success: Literal[True]
    message: str
    access_token: Annotated[StrictStr, Field(min_length=1)]
    refresh_token: Annotated[StrictStr, Field(min_length=1)]
    expires_in: Annotated[StrictInt, Field(gt=0)]
    refresh_expires_in: Annotated[StrictInt, Field(gt=0)]

    @field_validator("access_token", "refresh_token")
    @classmethod
    def tokens_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Token must not be blank")
        return value


class ForgotPasswordRequest(BaseModel):
    email: str


class ForgotPasswordResponse(BaseModel):
    success: bool
    message: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class ResetPasswordResponse(BaseModel):
    success: bool
    message: str


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == password_hash


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_PATTERN.match(email))


def hash_password_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def generate_password_reset_token() -> str:
    return secrets.token_urlsafe(32)


def get_app_name() -> str:
    return os.getenv("XAGENT_APP_NAME") or os.getenv("NEXT_PUBLIC_APP_NAME") or "Xagent"


def build_password_reset_url(token: str) -> str:
    base_url = get_app_base_url()
    if not base_url:
        raise RuntimeError(
            "XAGENT_APP_BASE_URL must be configured for password reset emails"
        )
    return f"{base_url}/reset-password?token={token}"


def has_users(db: Session) -> bool:
    return db.query(User.id).first() is not None


def is_registration_enabled(db: Session) -> bool:
    setting = (
        db.query(SystemSetting)
        .filter(SystemSetting.key == REGISTRATION_ENABLED_SETTING_KEY)
        .first()
    )
    if setting is None:
        return True
    return str(setting.value).lower() == "true"


def set_registration_enabled(db: Session, enabled: bool) -> None:
    setting = (
        db.query(SystemSetting)
        .filter(SystemSetting.key == REGISTRATION_ENABLED_SETTING_KEY)
        .first()
    )
    value = "true" if enabled else "false"
    if setting is None:
        setting = SystemSetting(key=REGISTRATION_ENABLED_SETTING_KEY, value=value)
        db.add(setting)
    else:
        setattr(setting, "value", value)
    db.commit()


def is_setup_completed(db: Session) -> bool:
    setting = (
        db.query(SystemSetting)
        .filter(SystemSetting.key == SETUP_COMPLETED_SETTING_KEY)
        .first()
    )
    return setting is not None and str(setting.value).lower() == "true"


@auth_router.get("/setup-status", response_model=SetupStatusResponse)
async def setup_status(db: Session = Depends(get_db)) -> SetupStatusResponse:
    initialized = has_users(db)
    registration_enabled = is_registration_enabled(db)
    return SetupStatusResponse(
        initialized=initialized,
        needs_setup=not initialized,
        registration_enabled=registration_enabled,
    )


@auth_router.post("/setup-admin", response_model=RegisterResponse)
async def setup_admin(
    request: RegisterRequest, db: Session = Depends(get_db)
) -> RegisterResponse:
    if len(request.password) < PASSWORD_MIN_LENGTH:
        return RegisterResponse(
            success=False,
            message=f"Password must be at least {PASSWORD_MIN_LENGTH} characters",
        )

    try:
        if has_users(db) or is_setup_completed(db):
            return RegisterResponse(success=False, message="Setup already completed")

        existing_user = get_user_by_username(db, request.username)
        if existing_user:
            return RegisterResponse(success=False, message="Username already exists")

        username_namespace_error = validate_username_for_login_namespace(
            db, request.username
        )
        if username_namespace_error:
            return RegisterResponse(success=False, message=username_namespace_error)

        email = None
        if request.email:
            email = normalize_email(request.email)
            if not is_valid_email(email):
                return RegisterResponse(success=False, message="Invalid email address")
            email_namespace_error = validate_email_for_login_namespace(db, email)
            if email_namespace_error:
                return RegisterResponse(success=False, message=email_namespace_error)
            existing_email_user = get_user_by_email(db, email)
            if existing_email_user:
                return RegisterResponse(success=False, message="Email already exists")

        user = User(
            username=request.username,
            email=email,
            password_hash=hash_password(request.password),
            is_admin=True,
        )
        db.add(user)
        db.flush()

        setup_setting = SystemSetting(key=SETUP_COMPLETED_SETTING_KEY, value="true")
        db.add(setup_setting)

        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        return RegisterResponse(success=False, message="Setup already completed")

    return RegisterResponse(
        success=True,
        message="Administrator account created successfully",
        user={
            "id": user.id,
            "username": user.username,
            "is_admin": bool(cast(Any, user.is_admin)),
            "createdAt": (
                cast(Any, user.created_at).isoformat()
                if getattr(user, "created_at", None) is not None
                else None
            ),
        },
    )


@auth_router.get("/register-switch", response_model=RegisterSwitchResponse)
async def get_register_switch(
    user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> RegisterSwitchResponse:
    if not bool(cast(Any, user.is_admin)):
        raise HTTPException(status_code=403, detail="Admin privileges required")

    enabled = is_registration_enabled(db)
    return RegisterSwitchResponse(
        success=True,
        registration_enabled=enabled,
        message="Registration switch fetched successfully",
    )


@auth_router.patch("/register-switch", response_model=RegisterSwitchResponse)
async def update_register_switch(
    request: RegisterSwitchRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> RegisterSwitchResponse:
    if not bool(cast(Any, user.is_admin)):
        raise HTTPException(status_code=403, detail="Admin privileges required")

    set_registration_enabled(db, request.enabled)
    return RegisterSwitchResponse(
        success=True,
        registration_enabled=request.enabled,
        message="Registration switch updated successfully",
    )


def create_user(db: Session, username: str, email: str, password: str) -> User:
    """Create a new user without default model configurations.

    Users will use admin defaults via dynamic fallback logic until they set their own.
    No pre-creation of UserModel or UserDefaultModel records.
    """
    password_hash = hash_password(password)
    user = User(username=username, email=email, password_hash=password_hash)
    db.add(user)
    db.flush()  # Get the user ID without committing
    db.refresh(user)

    # Commit everything together
    db.commit()
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    normalized_email = normalize_email(email)
    return db.query(User).filter(User.email == normalized_email).first()


def validate_username_for_login_namespace(
    db: Session, username: str, *, current_user_id: int | None = None
) -> Optional[str]:
    normalized_username = username.strip()
    if is_valid_email(normalized_username):
        return "Username cannot be an email address"

    conflicting_email_user = get_user_by_email(db, normalized_username)
    if (
        conflicting_email_user is not None
        and conflicting_email_user.id != current_user_id
    ):
        return "Username conflicts with an existing email"

    return None


def validate_email_for_login_namespace(
    db: Session, email: str, *, current_user_id: int | None = None
) -> Optional[str]:
    normalized_email = normalize_email(email)
    conflicting_username_user = get_user_by_username(db, normalized_email)
    if (
        conflicting_username_user is not None
        and conflicting_username_user.id != current_user_id
    ):
        return "Email conflicts with an existing username"
    return None


def _normalized_preferences(user: User) -> Dict[str, Any]:
    """The user's stored preferences as a plain dict, tolerating a NULL or
    malformed value. ``preferences`` has no nested-type constraint (same
    reasoning as apply_output_voice's isinstance guard on the ``voice``
    value it holds), so a corrupted/hand-edited row could store a non-dict
    JSON value here - ``dict(value or {})`` would raise on any of those
    instead of degrading to an empty dict."""
    preferences = cast(Any, user.preferences)
    return dict(preferences) if isinstance(preferences, dict) else {}


def serialize_auth_user(user: User, include_login_time: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": bool(cast(Any, user.is_admin)),
        "preferences": _normalized_preferences(user),
    }
    if include_login_time:
        payload["loginTime"] = datetime.now(timezone.utc).timestamp()
    return payload


def get_user_by_login_identifier(db: Session, identifier: str) -> Optional[User]:
    """Resolve an identifier as email first, then fall back to username."""
    login_identifier = identifier.strip()
    if is_valid_email(login_identifier):
        user = get_user_by_email(db, login_identifier)
        if user is not None:
            return user
    return get_user_by_username(db, login_identifier)


@auth_router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """User login endpoint"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_user_sync() -> User:
            # Get user from database
            user = get_user_by_login_identifier(db, request.username)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password",
                )
            return user

        # Execute database query in thread pool to avoid blocking
        user = await asyncio.to_thread(_get_user_sync)

        # Verify password
        if not verify_password(request.password, str(user.password_hash)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )

        # Create JWT tokens
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id},
            expires_delta=access_token_expires,
        )

        # Create refresh token
        refresh_token = create_refresh_token(
            data={"sub": user.username, "user_id": user.id}
        )

        # Store refresh token in database - run in thread pool to avoid blocking
        def _update_user_sync() -> None:
            setattr(user, "refresh_token", refresh_token)
            setattr(
                user,
                "refresh_token_expires_at",
                datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            )
            db.commit()

        # Execute database update in thread pool to avoid blocking
        await asyncio.to_thread(_update_user_sync)

        # Login successful
        return {
            "success": True,
            "message": "Login successful",
            "user": serialize_auth_user(user, include_login_time=True),
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # seconds
            "refresh_expires_in": REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,  # seconds
            "user_id": user.id,
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception:
        # str(e) is not put in the response: _update_user_sync's db.commit()
        # above persists the just-issued refresh_token as a bound SQL
        # parameter, and a SQLAlchemy StatementError's default __str__
        # would otherwise echo that live session token back to the client
        # -- the same class of leak fixed in generic_oauth_callback's
        # callback handler. hide_parameters=True on the engine
        # (models/database.py) now hides it there too, but this handler
        # doesn't rely on that alone. logger.exception still captures it
        # server-side.
        logger.exception("Login failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during login.",
        )


@auth_router.post("/register", response_model=RegisterResponse)
async def register(
    request: RegisterRequest, db: Session = Depends(get_db)
) -> RegisterResponse:
    """User registration endpoint with default configuration inheritance"""
    try:
        # Validate password length
        if len(request.password) < PASSWORD_MIN_LENGTH:
            return RegisterResponse(
                success=False,
                message=f"Password must be at least {PASSWORD_MIN_LENGTH} characters",
            )

        # Check if user already exists
        existing_user = get_user_by_username(db, request.username)
        if existing_user:
            return RegisterResponse(success=False, message="Username already exists")

        username_namespace_error = validate_username_for_login_namespace(
            db, request.username
        )
        if username_namespace_error:
            return RegisterResponse(success=False, message=username_namespace_error)

        initialized = has_users(db)
        if not initialized:
            return RegisterResponse(
                success=False,
                message="System is not initialized. Please create the first admin account.",
            )

        if not is_registration_enabled(db):
            return RegisterResponse(success=False, message="Registration is disabled")

        if not request.email:
            return RegisterResponse(success=False, message="Email is required")

        email = normalize_email(request.email)
        if not is_valid_email(email):
            return RegisterResponse(success=False, message="Invalid email address")

        email_namespace_error = validate_email_for_login_namespace(db, email)
        if email_namespace_error:
            return RegisterResponse(success=False, message=email_namespace_error)

        existing_email_user = get_user_by_email(db, email)
        if existing_email_user:
            return RegisterResponse(success=False, message="Email already exists")

        # Create new user with inherited defaults
        user = create_user(db, request.username, email, request.password)

        return RegisterResponse(
            success=True,
            message="Registration successful",
            user={
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "createdAt": (
                    cast(Any, user.created_at).isoformat()
                    if getattr(user, "created_at", None) is not None
                    else None
                ),
            },
        )

    except Exception:
        # Same reasoning as login's handler: create_user's db.commit() binds
        # password_hash as a SQL parameter, which a SQLAlchemy
        # StatementError's default __str__ would otherwise put into str(e)
        # and, via this response, into the client-facing error.
        logger.exception("Registration failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during registration.",
        )


@auth_router.post("/forgot-password", response_model=ForgotPasswordResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    db: Session = Depends(get_db),
) -> ForgotPasswordResponse:
    email = normalize_email(request.email)
    if not is_valid_email(email):
        return ForgotPasswordResponse(
            success=False,
            message="Please enter a valid email address",
        )

    user = get_user_by_email(db, email)
    if user is None:
        return ForgotPasswordResponse(
            success=True,
            message="If the email exists, a password reset link has been sent",
        )

    reset_token = generate_password_reset_token()
    reset_token_hash = hash_password_reset_token(reset_token)
    reset_expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=get_password_reset_expire_minutes()
    )

    setattr(user, "password_reset_token_hash", reset_token_hash)
    setattr(user, "password_reset_expires_at", reset_expires_at)
    db.commit()

    try:
        reset_link = build_password_reset_url(reset_token)
        await asyncio.to_thread(
            send_password_reset_email,
            email,
            reset_link,
            get_app_name(),
        )
    except Exception as exc:
        setattr(user, "password_reset_token_hash", None)
        setattr(user, "password_reset_expires_at", None)
        db.commit()
        logger.error("Failed to send password reset email to %s: %s", email, exc)

    return ForgotPasswordResponse(
        success=True,
        message="If the email exists, a password reset link has been sent",
    )


@auth_router.post("/reset-password", response_model=ResetPasswordResponse)
async def reset_password(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db),
) -> ResetPasswordResponse:
    if len(request.new_password) < PASSWORD_MIN_LENGTH:
        return ResetPasswordResponse(
            success=False,
            message=f"New password must be at least {PASSWORD_MIN_LENGTH} characters",
        )

    token_hash = hash_password_reset_token(request.token)
    user = db.query(User).filter(User.password_reset_token_hash == token_hash).first()
    if user is None:
        return ResetPasswordResponse(success=False, message="Invalid reset token")

    reset_expires_at = getattr(user, "password_reset_expires_at", None)
    now = datetime.now(timezone.utc)
    if reset_expires_at is None:
        return ResetPasswordResponse(success=False, message="Invalid reset token")
    if (
        hasattr(reset_expires_at, "tzinfo")
        and getattr(reset_expires_at, "tzinfo", None) is not None
    ):
        if cast(Any, reset_expires_at) < now:
            return ResetPasswordResponse(
                success=False, message="Reset token has expired"
            )
    else:
        if cast(Any, reset_expires_at) < now.replace(tzinfo=None):
            return ResetPasswordResponse(
                success=False, message="Reset token has expired"
            )

    setattr(user, "password_hash", hash_password(request.new_password))
    setattr(user, "password_reset_token_hash", None)
    setattr(user, "password_reset_expires_at", None)
    setattr(user, "refresh_token", None)
    setattr(user, "refresh_token_expires_at", None)
    db.commit()

    return ResetPasswordResponse(
        success=True,
        message="Password has been reset successfully",
    )


@auth_router.post("/change-password", response_model=ChangePasswordResponse)
async def change_password(
    request: ChangePasswordRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChangePasswordResponse:
    """Change user password endpoint"""
    try:
        # Verify current password
        if not verify_password(request.current_password, str(user.password_hash)):
            return ChangePasswordResponse(
                success=False, message="Current password is incorrect"
            )

        # Validate new password
        if len(request.new_password) < PASSWORD_MIN_LENGTH:
            return ChangePasswordResponse(
                success=False,
                message=f"New password must be at least {PASSWORD_MIN_LENGTH} characters",
            )

        # Update password
        setattr(user, "password_hash", hash_password(request.new_password))
        db.commit()

        return ChangePasswordResponse(
            success=True, message="Password updated successfully"
        )

    except Exception:
        # Same reasoning as login's handler: the db.commit() above binds the
        # new password_hash as a SQL parameter, which a SQLAlchemy
        # StatementError's default __str__ would otherwise put into str(e)
        # and, via this response, into the client-facing error.
        logger.exception("Password update failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating the password.",
        )


@auth_router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    user: User = Depends(get_current_user),
) -> UserProfileResponse:
    return UserProfileResponse(
        success=True,
        message="User profile fetched successfully",
        user=serialize_auth_user(user),
    )


@auth_router.patch("/email", response_model=UpdateEmailResponse)
async def update_current_user_email(
    request: UpdateEmailRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> UpdateEmailResponse:
    email = normalize_email(request.email)
    if not is_valid_email(email):
        return UpdateEmailResponse(success=False, message="Invalid email address")

    existing_user = get_user_by_email(db, email)
    if existing_user and existing_user.id != user.id:
        return UpdateEmailResponse(success=False, message="Email already exists")

    email_namespace_error = validate_email_for_login_namespace(
        db, email, current_user_id=int(user.id)
    )
    if email_namespace_error:
        return UpdateEmailResponse(success=False, message=email_namespace_error)

    setattr(user, "email", email)
    db.commit()
    db.refresh(user)

    return UpdateEmailResponse(
        success=True,
        message="Email updated successfully",
        user=serialize_auth_user(user),
    )


def _lock_user_row_for_preferences_update(db: Session, user_id: int) -> bool:
    """Serialize concurrent preferences PATCHes for one user, in every
    database - mirrors acquire_runtime_key_transition_fence's dual-dialect
    pattern (services/api_keys.py). PostgreSQL/MySQL take a row-level
    ``FOR UPDATE`` lock; SQLite ignores that clause, so a no-op write grabs
    its write lock instead. Held until this transaction commits, so a
    second concurrent PATCH's read-modify-write of the same JSON column
    blocks here instead of reading stale data and silently dropping the
    first request's disjoint fields on its own commit.

    Returns ``False`` when the user no longer exists (deleted between the
    caller resolving the id and this call), the same contract the
    mirrored helper uses - letting the caller turn that into a clean 404
    instead of an unhandled ``ObjectDeletedError`` from the subsequent
    fetch."""
    if db.get_bind().dialect.name == "sqlite":
        db.execute(
            text("UPDATE users SET id = id WHERE id = :user_id"),
            {"user_id": user_id},
        )
        return db.query(User.id).filter(User.id == user_id).first() is not None
    return (
        db.query(User.id).filter(User.id == user_id).with_for_update().first()
        is not None
    )


def _merge_user_preferences_locked(
    user_id: int, updates: Dict[str, Any]
) -> Dict[str, Any] | None:
    """Entirely self-contained: opens and closes its own Session and never
    touches a request-scoped `user`/`db`. This is required, not just
    tidy: the lock wait this is built around is a real, potentially slow
    blocking DB call under contention, so the endpoint runs this whole
    operation through asyncio.to_thread and drains it via
    await_task_settlement (the same primitive run_db_io_cancellation_safe
    wraps, called directly here so the endpoint can still invalidate the
    voice cache off a settled-but-cancelled result before that
    cancellation propagates) - if the request is cancelled (client
    disconnect, timeout) while this worker is mid-transaction, FastAPI
    can tear down the request's own Session concurrently, and the
    contract is exactly "operation must create/use/close its own Session
    and return only detached data" so that race can't happen (see
    db_runtime.py's docstring on it; this mirrors admin_users.py's
    _delete_user_rows_sync).

    Returns the serialized response payload (not the ORM object), built
    BEFORE commit: Session.commit() defaults to expire_on_commit=True, so
    any attribute access after commit would force a fresh SELECT - itself
    a race, since the lock is released the instant commit() returns and a
    concurrent delete could land in that gap. The preferences merge is
    already reflected in the in-memory object the moment it's set, so
    nothing here needs a post-commit read at all.

    Returns ``None`` if the user no longer exists (deleted between the
    caller resolving ``user_id`` and this call)."""
    session_factory = get_session_local()
    worker_db = session_factory()
    try:
        if not _lock_user_row_for_preferences_update(worker_db, user_id):
            return None
        # Loaded fresh, under the lock, in this operation's own session -
        # not whatever `User` row the caller may have loaded earlier,
        # which could be stale (a concurrent PATCH may have committed its
        # own merge while this request was waiting on the lock).
        worker_user = worker_db.get(User, user_id)
        if worker_user is None:
            return None
        current_preferences = _normalized_preferences(worker_user)
        for key, value in updates.items():
            # An explicit `null` is this endpoint's only clear-a-field
            # signal (see UpdatePreferencesRequest's blank-string
            # rejection) - storing a literal `{key: None}` entry instead
            # of deleting the key would have every future read replay a
            # stale explicit-null forever. Note this doesn't apply to
            # `goals: []`: an empty list is itself a valid value (not a
            # clear signal) and is stored as-is, so `null` and `[]` are
            # two different representations of "no goals" that can
            # coexist - harmless today since nothing branches on the
            # distinction, but worth knowing if that ever changes.
            if value is None:
                current_preferences.pop(key, None)
            else:
                current_preferences[key] = value
        setattr(worker_user, "preferences", current_preferences)
        payload = serialize_auth_user(worker_user)
        worker_db.commit()
        return payload
    finally:
        worker_db.close()


@auth_router.patch("/me/preferences", response_model=UpdatePreferencesResponse)
async def update_current_user_preferences(
    request: UpdatePreferencesRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> UpdatePreferencesResponse:
    """Merge the given fields into the current user's stored preferences.
    Each onboarding step (About you, Goals, Launch) calls this with only
    its own fields - a merge, not a replace, so an earlier step's answer
    survives a later step's PATCH."""
    updates = request.model_dump(exclude_unset=True)
    if not updates:
        return UpdatePreferencesResponse(
            success=True,
            message="Preferences updated successfully",
            user=serialize_auth_user(user),
        )

    # Read before releasing `db` below: Session.rollback() unconditionally
    # expires every object loaded through that session, `user` included
    # (unlike expire_on_commit, this isn't conditional on a session
    # setting), so touching `user.id` after release would force an
    # implicit reload - reacquiring the very connection just released,
    # synchronously on the event loop, defeating the point of releasing
    # it at all, and raising ObjectDeletedError outright if the row was
    # deleted concurrently in the meantime.
    user_id = int(user.id)

    # `db` is declared only to release it here, not to do any work with it
    # directly: FastAPI's per-request dependency caching means this is the
    # same (read-only, since get_current_user only did a SELECT) session
    # get_current_user already used, and _merge_user_preferences_locked
    # is about to open a second, independent session and block on a real
    # row lock - without this, that session would sit idle-in-transaction,
    # holding a pool slot, for the whole lock wait (issue #889, same
    # pattern already used before chat.py's sandbox startup and
    # workforce_creator.py's ReAct builder call). A read-only session
    # should always release cleanly; treating a failure as a hard error
    # (mirroring workforce_creator.py's own release_db_connection_if_clean
    # call) surfaces that as a signal instead of silently holding the
    # connection through the lock wait anyway.
    if not release_db_connection_if_clean(db):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "preferences_update_unavailable",
                "message": "Could not release the database before updating preferences.",
            },
        )

    # run_db_io_cancellation_safe's own contract discards a settled result
    # in favor of re-raising the caller's deferred cancellation (see its
    # docstring), which would otherwise skip the cache invalidation below
    # entirely - the merge already committed by the time that cancellation
    # is observed, so the cache must still be invalidated before this
    # function lets that cancellation propagate. propagate_deferred_
    # cancellation (the same helper task_command_transport.py's heartbeat
    # settlement uses) makes that ordering safe even if invalidation
    # itself raises, or 404s below: cancellation always wins over a
    # later exception or return, never the reverse.
    worker = asyncio.get_running_loop().create_task(
        asyncio.to_thread(_merge_user_preferences_locked, user_id, updates)
    )
    serialized_user, cancellation = await await_task_settlement(worker)
    with propagate_deferred_cancellation(cancellation):
        if serialized_user is None:
            raise HTTPException(status_code=404, detail="User not found")

        if "voice" in updates:
            # A cached AgentService bakes voice into its system prompt at
            # construction time and won't re-check preferences on later
            # turns (see invalidate_cached_agents_for_owner's docstring),
            # so an already-warm task would otherwise keep speaking in the
            # old (or now-cleared) voice until incidental eviction/rebuild.
            #
            # The merge above already committed - this is best-effort
            # follow-up, not part of "did the write succeed." An
            # unguarded raise here would turn a successful PATCH into a
            # client-visible 500 for a write that already happened,
            # mirroring the exact bug _run_post_commit_oauth_side_effects
            # (issue #1150) exists to prevent; same fix, same reasoning.
            from .chat import get_agent_manager

            try:
                await get_agent_manager().invalidate_cached_agents_for_owner(user_id)
            except Exception:
                logger.warning(
                    "Voice cache invalidation failed for user %s after a "
                    "successful preferences write; a warm task may keep "
                    "speaking in the old voice until incidental eviction",
                    user_id,
                    exc_info=True,
                )

        return UpdatePreferencesResponse(
            success=True,
            message="Preferences updated successfully",
            user=serialized_user,
        )


@auth_router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    db: Session = Depends(get_db),
) -> RefreshTokenResponse:
    """Refresh JWT access token using refresh token"""
    try:
        # Verify refresh token
        payload = verify_refresh_token(request.refresh_token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        # Get user from database
        user_id = payload.get("user_id")
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        # Check if refresh token matches and is not expired
        user_refresh_token = getattr(user, "refresh_token", None)
        refresh_token_expires_at = getattr(user, "refresh_token_expires_at", None)
        if (
            user_refresh_token != request.refresh_token
            or refresh_token_expires_at is None
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

        # Check expiration - handle timezone-aware and naive datetimes
        now = datetime.now(timezone.utc)
        if (
            hasattr(refresh_token_expires_at, "tzinfo")
            and getattr(refresh_token_expires_at, "tzinfo", None) is not None
        ):
            # Timezone-aware datetime
            if cast(Any, refresh_token_expires_at) < now:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token has expired",
                )
        else:
            # Naive datetime - assume UTC
            if cast(Any, refresh_token_expires_at) < now.replace(tzinfo=None):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Refresh token has expired",
                )

        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.id},
            expires_delta=access_token_expires,
        )

        # Optionally: Create new refresh token (rotation)
        new_refresh_token = create_refresh_token(
            data={"sub": user.username, "user_id": user.id}
        )
        setattr(user, "refresh_token", new_refresh_token)
        setattr(
            user,
            "refresh_token_expires_at",
            datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        )
        db.commit()

        return RefreshTokenResponse(
            success=True,
            message="Token refreshed successfully",
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # seconds
            refresh_expires_in=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,  # seconds
        )

    except HTTPException:
        raise
    except SQLAlchemyError:
        logger.exception("Token refresh failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Token refresh is temporarily unavailable",
        )


@auth_router.get("/check")
async def check_auth() -> Dict[str, Any]:
    """Check authentication status endpoint"""
    return {"success": True, "message": "Authentication API is working"}


@auth_router.get("/verify")
async def verify_current_token(
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Verify current token validity"""
    return {
        "success": True,
        "message": "Token is valid",
        "user": serialize_auth_user(current_user),
    }


def generic_oauth_login(
    provider: str,
    token: Optional[str] = None,
    app_id: Optional[str] = None,
    redirect: Optional[str] = None,
    db: Optional[Session] = None,
    db_provider: Optional[Any] = None,
) -> Any:
    """Start generic OAuth flow"""
    if db is None:
        raise RuntimeError("db session is required")
    if not db_provider:
        return HTMLResponse(
            content="<h1>Error: Provider not configured</h1>", status_code=500
        )

    client_id = _resolve_oauth_secret(provider, db_provider.client_id, "CLIENT_ID")
    if not client_id:
        return _oauth_provider_config_error(
            provider, [_oauth_env_name(provider, "CLIENT_ID")]
        )
    auth_url = db_provider.auth_url

    redirect_uri = _resolve_oauth_redirect_uri(provider, db_provider)

    user_id = None
    if token:
        payload = verify_token(token)
        if payload and payload.get("type") == "access":
            username = payload.get("sub")
            user = db.query(User).filter(User.username == username).first()
            if user:
                user_id = user.id

    if not user_id:
        return HTMLResponse(
            content="<h1>Error: Not authenticated</h1><p>Please provide a valid token.</p>",
            status_code=401,
        )

    if not app_id:
        from ..mcp_apps import requires_app_scoped_oauth_grant

        # A bare (app_id-less) login persists to UserOAuth.provider==provider
        # -- for a provider in APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT whose
        # sole app's app_id is that same provider string (e.g. github), that
        # is the EXACT SAME key an app-scoped login uses. Without this
        # guard, re-running the bare route would silently delete-and-replace
        # a fully-scoped connection's grant with an identity-only one via
        # generic_oauth_callback's delete-then-recreate step below, while
        # the existing MCPServer/UserMCPServer row is left active and now
        # backed by an under-scoped token. Facebook/Instagram are unaffected
        # (their bare provider string "meta" is never itself a member of the
        # set, only their app_ids "facebook"/"instagram" are).
        #
        # Checked after the config/auth checks above (not before) so an
        # unauthenticated or misconfigured caller still gets the more
        # specific 401/config-error response instead of a 404 that implies
        # the connector itself doesn't support this route -- this still
        # runs before any state token is minted below, which is the actual
        # security property this guard needs to hold.
        if requires_app_scoped_oauth_grant(provider):
            return HTMLResponse(
                content=(
                    "<h1>Cannot Connect</h1>"
                    "<p>This connector must be started from its catalog "
                    "entry.</p>"
                ),
                status_code=404,
            )

    state_payload = {
        "type": "oauth_state",
        "user_id": user_id,
        "provider": provider,
        "app_id": app_id,
        "redirect": redirect,
    }
    # Newer Salesforce orgs enforce PKCE on this authorization-code grant at
    # the org level, with no per-app way to disable it (Setup > External
    # Client Apps > Security > "Require Proof Key for Code Exchange" is
    # locked once an org has it on); Employment Hero mandates it for every
    # app from 2026-09-14. requires_pkce() (oauth_provider_quirks.py) is the
    # shared predicate for this provider family, not just _is_salesforce_
    # provider, since it's no longer Salesforce-only. The verifier rides
    # inside this signed, short-lived state token rather than a new DB row,
    # to avoid a schema change for this -- but `state` itself goes out as a
    # URL query param on the redirect to the provider and back, so it lands
    # in browser history/Referer headers/proxy logs. HS256 signing alone
    # doesn't hide the payload (it's base64, not encrypted), so the verifier
    # is encrypted here before being embedded, and decrypted back out in the
    # callback below. The token exchange still requires the server-held
    # client_secret regardless, so this is defense-in-depth on top of that,
    # not the only thing standing between an interceptor and a token.
    code_verifier = secrets.token_urlsafe(64) if requires_pkce(provider) else None
    if code_verifier:
        from ...core.utils.encryption import encrypt_value

        try:
            state_payload["code_verifier"] = encrypt_value(code_verifier)
        except ValueError:
            # get_cipher() raises this when ENCRYPTION_KEY is unset outside
            # development -- every other provider's login route never calls
            # encrypt_value at all, so this misconfiguration is otherwise
            # invisible until the first PKCE-gated connect attempt (now
            # Salesforce or Employment Hero, per requires_pkce() -- the
            # provider name is interpolated below rather than hardcoded, so
            # this stays accurate as that set grows). Not routed through
            # _oauth_provider_config_error: that helper's "Missing X for
            # provider Y" phrasing is written for a provider-prefixed env var
            # (e.g. SALESFORCE_CLIENT_ID) and would misleadingly suggest a
            # SALESFORCE_ENCRYPTION_KEY-style variable exists, when
            # ENCRYPTION_KEY is a single global setting unrelated to any one
            # provider.
            return HTMLResponse(
                content=(
                    "<h1>Error: Server misconfigured</h1>"
                    "<p>The ENCRYPTION_KEY environment variable is not set. "
                    f"This is required to connect {html.escape(provider)}; set "
                    "it and restart the backend.</p>"
                ),
                status_code=500,
            )
    state = create_access_token(data=state_payload, expires_delta=timedelta(minutes=10))

    app_scopes: list[str] | None = None
    app_optional_scopes: list[str] = []
    from ..mcp_apps import get_app_by_id

    if app_id:
        app_info = get_app_by_id(db, app_id)
        if app_info:
            # Reject a hidden app here too, not only at the callback --
            # otherwise a hidden connector's consent screen is still shown at
            # the real provider (no security bypass, since the callback
            # still blocks the connect, but a confusing UX: the app doesn't
            # feel hidden if you can watch it start connecting). Unlike the
            # callback's gate, an unknown app_id is deliberately left alone
            # here too, for the same reason (e.g. gmail's bare-app-id flows).
            from .mcp import _reject_hidden_catalog_app

            try:
                _reject_hidden_catalog_app(app_info)
            except HTTPException:
                return HTMLResponse(
                    content=(
                        "<h1>Cannot Connect</h1>"
                        "<p>This app is not currently available.</p>"
                    ),
                    status_code=404,
                )
            if "oauth_scopes" in app_info:
                app_scopes = app_info["oauth_scopes"]
            app_optional_scopes = app_info.get("optional_oauth_scopes") or []

    scopes = _merge_oauth_scopes(db_provider.default_scopes or [], app_scopes)
    scope_str = _oauth_scope_separator(provider).join(scopes)
    # Sent via the authorize request's own optional_scope parameter (see
    # get_builtin_execution_fields_and_optional_scopes) rather than merged
    # into `scopes` above: a scope tier-gated on the connected account's
    # plan would otherwise block the whole authorization if the account
    # can't grant it. Scopes already in the required set are dropped here
    # too - not just a duplicate-avoidance nicety, but correctness: HubSpot
    # blocks installation outright if the same scope appears in both `scope`
    # and `optional_scope` on one authorize request. dict.fromkeys then
    # dedupes what's left before joining, in case the registry ever lists a
    # scope twice within optional_oauth_scopes itself.
    required_scopes = set(scopes)
    optional_scope_str = _oauth_scope_separator(provider).join(
        sorted(
            dict.fromkeys(
                scope
                for scope in app_optional_scopes
                if scope and scope not in required_scopes
            )
        )
    )

    from urllib.parse import urlencode

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    if provider.lower() == "google":
        params["access_type"] = "offline"
        params["include_granted_scopes"] = "true"
        params["prompt"] = "consent"
    if provider.lower() == "zoom":
        params["prompt"] = "login"
    if code_verifier:
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        params["code_challenge"] = (
            base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        )
        params["code_challenge_method"] = "S256"
    if provider.lower() == "jira":
        # Required by Atlassian's authorize endpoint for every 3LO app,
        # regardless of scopes requested -- identifies the resource server
        # the requested token is meant for (api.atlassian.com), not an
        # actual audience restriction on this connector's own behalf.
        params["audience"] = "api.atlassian.com"
        # Atlassian does not silently re-prompt on a scope change like most
        # providers here -- without forcing the consent screen, a user who
        # previously granted a narrower scope set can be silently handed a
        # token still limited to that earlier grant.
        params["prompt"] = "consent"
    if provider.lower() == "linear":
        # Linear's OAuth docs confirm only that prompt=consent always shows
        # the consent screen -- they don't document what happens by default
        # on a scope-escalation request without it. Forcing consent
        # sidesteps needing to know that default: a bare provider connect
        # (read only) followed by an app-scoped connect (read+write) is
        # guaranteed to end up with the broader grant either way.
        params["prompt"] = "consent"
    meta_config_id = _meta_login_config_id() if provider.lower() == "meta" else ""
    if meta_config_id:
        params["config_id"] = meta_config_id
    else:
        if scope_str:
            params["scope"] = scope_str
        if optional_scope_str:
            params["optional_scope"] = optional_scope_str

    separator = "&" if "?" in auth_url else "?"
    full_auth_url = f"{auth_url}{separator}{urlencode(params)}"
    return RedirectResponse(full_auth_url)


class AppNotOAuthError(ValueError):
    """Raised when an app routed through the OAuth flow is not builtin_oauth.

    A dedicated subclass so the batch-connect loop can skip only this case while
    still surfacing the genuine metadata-conflict ValueErrors _ensure_user_mcp_server
    raises for legitimate OAuth apps.
    """


def _ensure_user_mcp_server(
    db: Session, user_id: int, app_info: Dict[str, Any]
) -> None:
    """Ensure MCPServer and UserMCPServer records exist for an OAuth app."""
    from sqlalchemy.exc import IntegrityError

    from ..models.mcp import MCPServer, UserMCPServer

    # Symmetric with the key-based gate in _ensure_catalog_app_server: only apps
    # classified as builtin_oauth may land here. Otherwise a key-based app routed
    # through OAuth would get a token injected but never its required_env API key.
    if app_info.get("auth_type") != "builtin_oauth":
        raise AppNotOAuthError(
            f"App '{app_info.get('name')}' is not an OAuth app and cannot be "
            "connected via the OAuth flow."
        )

    def _oauth_auth_metadata() -> dict[str, str]:
        metadata = {"app_id": str(app_info["id"])}
        provider = app_info.get("provider")
        if provider:
            metadata["provider"] = str(provider)
        return metadata

    def _ensure_server_matches_oauth_app(server: MCPServer) -> None:
        if server.transport != "oauth":
            raise ValueError(
                f"OAuth app '{app_info['name']}' conflicts with an existing MCP server "
                f"using transport '{server.transport}'. Delete or rename that custom "
                "server before connecting the official OAuth app."
            )

        auth: dict[str, Any] = server.auth if isinstance(server.auth, dict) else {}
        expected_app_id = str(app_info["id"])
        existing_app_id = auth.get("app_id")
        if existing_app_id and str(existing_app_id) != expected_app_id:
            raise ValueError(
                f"OAuth app '{app_info['name']}' conflicts with MCP server metadata "
                f"for app '{existing_app_id}'."
            )

        expected_provider = app_info.get("provider")
        existing_provider = auth.get("provider")
        if (
            expected_provider
            and existing_provider
            and str(existing_provider) != str(expected_provider)
        ):
            raise ValueError(
                f"OAuth app '{app_info['name']}' conflicts with MCP server provider "
                f"'{existing_provider}'."
            )

        auth.update(_oauth_auth_metadata())
        cast(Any, server).auth = auth
        server.description = app_info.get("description") or server.description

    mcp_server = db.query(MCPServer).filter(MCPServer.name == app_info["name"]).first()
    if not mcp_server:
        mcp_server = MCPServer(
            name=app_info["name"],
            description=app_info["description"],
            managed="external",
            transport="oauth",
            auth=_oauth_auth_metadata(),
        )
        db.add(mcp_server)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            mcp_server = (
                db.query(MCPServer).filter(MCPServer.name == app_info["name"]).first()
            )
            if not mcp_server:
                raise

    _ensure_server_matches_oauth_app(mcp_server)

    user_mcp = (
        db.query(UserMCPServer)
        .filter(
            UserMCPServer.user_id == user_id,
            UserMCPServer.mcpserver_id == mcp_server.id,
        )
        .first()
    )

    if not user_mcp:
        user_mcp = UserMCPServer(
            user_id=user_id, mcpserver_id=mcp_server.id, is_owner=True, is_active=True
        )
        db.add(user_mcp)


def generic_oauth_callback(
    provider: str,
    request: Request,
    db: Optional[Session] = None,
    db_provider: Optional[Any] = None,
) -> Any:
    """Handle generic OAuth callback"""
    if db is None:
        raise RuntimeError("db session is required")
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    if error:
        return HTMLResponse(
            content=f"<h1>Error: {html.escape(str(error))}</h1>", status_code=400
        )

    if not code or not state:
        return HTMLResponse(
            content="<h1>Error: Missing code or state</h1>", status_code=400
        )

    payload = verify_token(state)
    if (
        not payload
        or payload.get("type") != "oauth_state"
        or payload.get("provider") != provider
    ):
        return HTMLResponse(
            content="<h1>Error: Invalid or expired state</h1>", status_code=400
        )

    user_id_claim = payload.get("user_id")
    if user_id_claim is not None and (
        type(user_id_claim) is not int or not 0 < user_id_claim <= _MAX_USER_ID
    ):
        # Reject malformed state before exchanging the provider code or
        # mutating OAuth rows. ``User.id`` uses a signed database integer, so
        # values accepted only by SQLite must also fail at this boundary.
        return HTMLResponse(
            content="<h1>Error: Invalid or expired state</h1>", status_code=400
        )
    user_id = user_id_claim
    app_id = payload.get("app_id")
    encrypted_code_verifier = payload.get("code_verifier")
    code_verifier = None
    if encrypted_code_verifier:
        from ...core.utils.encryption import decrypt_value_strict

        # Strict, not the lenient decrypt_value: a verifier this can't open
        # (e.g. ENCRYPTION_KEY rotated mid-flight, inside the state token's
        # 10-minute window) must not silently fall back to sending the raw
        # ciphertext to Salesforce as code_verifier -- that only surfaces as
        # an opaque invalid_grant from Salesforce instead of a clear cause.
        # Catching ValueError, not just its EncryptionDecodeError subclass:
        # decrypt_value_strict's own get_cipher() call raises a bare
        # ValueError when ENCRYPTION_KEY is unset outside development,
        # which is exactly the same "surface it clearly" case, not just a
        # token that fails to decrypt under a present key.
        try:
            code_verifier = decrypt_value_strict(encrypted_code_verifier)
        except ValueError:
            return HTMLResponse(
                content=(
                    "<h1>Error: Session expired</h1><p>Please try connecting again.</p>"
                ),
                status_code=400,
            )

    if not app_id:
        from ..mcp_apps import requires_app_scoped_oauth_grant

        # Symmetric to generic_oauth_login's own guard: that guard only
        # stops a NEW bare state from being minted, so it can't protect a
        # bare state that was already signed (and is still within its
        # 10-minute TTL) before this guard was deployed, or a future
        # internal caller that reaches this callback with an app_id-less
        # state directly. Checked here, before any token exchange or the
        # delete-then-recreate UserOAuth write below, so a bare grant can
        # never replace an existing app-scoped one for these providers.
        if requires_app_scoped_oauth_grant(provider):
            return HTMLResponse(
                content=(
                    "<h1>Cannot Connect</h1>"
                    "<p>This connector must be started from its catalog "
                    "entry.</p>"
                ),
                status_code=404,
            )

    if app_id:
        # Reject a hidden app before spending the authorization code against
        # the real provider, not just before persisting -- is_visible_in_
        # connector is also used as a release gate (e.g. an unverified
        # builtin_oauth connector shipping hidden), and this builtin_oauth
        # provider-redirect flow used to be the one connect path
        # _reject_hidden_catalog_app's own docstring flagged as NOT enforcing
        # that gate (#1203): an authenticated user who knew (or guessed) the
        # app_id could still connect a hidden connector. Checked here, ahead
        # of the token exchange below, rather than only later next to
        # _ensure_user_mcp_server, so a hidden app never even reaches the
        # provider -- matching the "indistinguishable from nonexistent" intent
        # the helper already documents for its other two call sites. The bare
        # (app_id-less) batch branch below cannot move this early: it doesn't
        # know which apps share the provider until it queries them, so its
        # own per-app check stays where it is.
        from ..mcp_apps import get_app_by_id
        from .mcp import _reject_hidden_catalog_app

        # An app_id that resolves to no catalog row at all is left alone,
        # deliberately not folded into this gate: several existing OAuth
        # flows (e.g. gmail in the test suite) legitimately reach this
        # callback with an app_id that has no PublicMCPApp row, and rely on
        # falling through to a bare/provider-scoped grant rather than being
        # rejected -- confirmed by running the full oauth test suite before
        # settling on this narrower check. Only a row that exists AND is
        # hidden is rejected here.
        target_app_info = get_app_by_id(db, app_id)
        if target_app_info:
            try:
                _reject_hidden_catalog_app(target_app_info)
            except HTTPException:
                return HTMLResponse(
                    content=(
                        "<h1>Cannot Connect</h1>"
                        "<p>This app is not currently available.</p>"
                    ),
                    status_code=404,
                )

    if not db_provider:
        return HTMLResponse(
            content="<h1>Error: Provider not configured</h1>", status_code=500
        )

    client_id = _resolve_oauth_secret(provider, db_provider.client_id, "CLIENT_ID")
    client_secret = _resolve_oauth_secret(
        provider, db_provider.client_secret, "CLIENT_SECRET"
    )
    missing_config = []
    if not client_id:
        missing_config.append(_oauth_env_name(provider, "CLIENT_ID"))
    if not client_secret:
        missing_config.append(_oauth_env_name(provider, "CLIENT_SECRET"))
    if missing_config:
        return _oauth_provider_config_error(provider, missing_config)
    token_url = db_provider.token_url
    userinfo_url = db_provider.userinfo_url

    redirect_uri = _resolve_oauth_redirect_uri(provider, db_provider)

    try:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }
        if code_verifier:
            data["code_verifier"] = code_verifier
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if requires_json_accept_header(provider):
            headers["Accept"] = "application/json"
        auth: tuple[str, str] | None = None
        if provider.lower() == "zoom":
            # Zoom's token endpoint requires HTTP Basic Auth for client
            # credentials (client_id:client_secret, base64).
            auth = (client_id, client_secret)
        else:
            data["client_id"] = client_id
            data["client_secret"] = client_secret

        # Atlassian's token endpoint requires a JSON body -- unlike every
        # other provider here, it does not accept form-urlencoded and
        # answers a form-encoded POST with a 400.
        post_kwargs: dict[str, Any] = {"data": data}
        if provider.lower() == "jira":
            # In-place, not a full `headers = {...}` reassignment -- that
            # would silently drop an Accept header set above by
            # requires_json_accept_header(), matching the refresh-path
            # branch in tools/config.py. Currently a latent distinction
            # only (no provider needs both quirks at once yet).
            headers["Content-Type"] = "application/json"
            post_kwargs = {"json": data}

        token_response = requests.post(
            token_url, headers=headers, timeout=10.0, auth=auth, **post_kwargs
        )
        try:
            token_data = token_response.json()
        except ValueError:
            # The Accept-header quirk above only pushes providers toward a
            # JSON body -- it doesn't guarantee one (a proxy stripping the
            # header, a misconfigured enterprise server). Without this, a
            # non-JSON body reaches the outer handler as a bare
            # JSONDecodeError instead of the same clear, actionable message
            # every other failure branch in this callback gives.
            logger.warning(
                "OAuth token exchange for provider=%s returned a non-JSON "
                "response (status %s)",
                provider,
                token_response.status_code,
            )
            return HTMLResponse(
                content=(
                    "<h1>Error exchanging token</h1>"
                    f"<p>{html.escape(provider)} returned a response that "
                    "could not be parsed.</p>"
                ),
                status_code=400,
            )

        if not isinstance(token_data, dict):
            # A JSON-parseable but non-object body (a bare list/string/
            # number) would otherwise reach `"error" in token_data` --
            # which doesn't raise for a list/string (it does a membership/
            # substring check, not a key check) -- and then
            # `token_data.get("access_token")` below, which does raise
            # (AttributeError) on anything but a dict. That would escape
            # to the outer handler as an opaque 500 instead of the same
            # clear, actionable 400 every other malformed-body case here
            # gives -- the same class of gap the userinfo guards below
            # close for that response.
            logger.warning(
                "OAuth token exchange for provider=%s returned a non-object body (%s)",
                provider,
                type(token_data).__name__,
            )
            return HTMLResponse(
                content=(
                    "<h1>Error exchanging token</h1>"
                    f"<p>{html.escape(provider)} returned an unexpected "
                    "response.</p>"
                ),
                status_code=400,
            )

        if "error" in token_data:
            # The response to the browser is deliberately trimmed to the
            # allowlisted error/error_description fields (see
            # _bounded_oauth_error_message's docstring) -- log the same
            # allowlisted projection server-side rather than the raw dict,
            # since a malformed/partial response can carry a live
            # access_token alongside an error field, and logging the raw
            # dict would put it in the server log instead of the browser.
            logger.warning(
                "OAuth token exchange failed for provider=%s: %s",
                provider,
                _redact_oauth_log_payload(token_data),
            )
            return HTMLResponse(
                content=(
                    "<h1>Error exchanging token</h1>"
                    f"<p>{_bounded_oauth_error_message(token_data)}</p>"
                ),
                status_code=400,
            )

        if not 200 <= token_response.status_code < 300:
            # A non-2xx response with no "error" key still isn't a
            # successful exchange -- without this, a body that happens to
            # be JSON-parseable and (coincidentally, or via a
            # misbehaving proxy/gateway) carries an access_token-shaped
            # field would be trusted as success purely because
            # `"error" in token_data` was false, regardless of the HTTP
            # status actually returned.
            logger.warning(
                "OAuth token exchange for provider=%s returned status %s "
                "with no explicit error field: %s",
                provider,
                token_response.status_code,
                _redact_oauth_log_payload(token_data),
            )
            return HTMLResponse(
                content=(
                    "<h1>Error exchanging token</h1>"
                    f"<p>{html.escape(provider)} returned an unexpected "
                    f"response (status {token_response.status_code}).</p>"
                ),
                status_code=400,
            )

        token_data = _exchange_meta_long_lived_token(
            provider, token_url, token_data, client_id, client_secret
        )
        token_data = _normalize_intercom_token_response(provider, token_data)
        access_token = token_data.get("access_token")

        if not access_token:
            # Without this, a token response that slips past the "error" in
            # token_data check above (e.g. Intercom's error.list envelope,
            # which doesn't use that key) would fall through to the
            # persistence block below with access_token=None, hit
            # UserOAuth.access_token's NOT NULL constraint, and surface as a
            # raw SQLAlchemy IntegrityError message through the generic
            # exception handler instead of a clear, actionable error.
            logger.warning(
                "OAuth token exchange for provider=%s returned no access_token: %s",
                provider,
                _redact_oauth_log_payload(token_data),
            )
            message = f"{html.escape(provider)} did not return an access token."
            detail = _extract_provider_error_message(token_data)
            if detail:
                message = f"{message} {html.escape(detail)}"
            return HTMLResponse(
                content=f"<h1>Error exchanging token</h1><p>{message}</p>",
                status_code=400,
            )

        salesforce_instance_url = token_data.get("instance_url")
        if _is_salesforce_provider(provider) and (
            not isinstance(salesforce_instance_url, str) or not salesforce_instance_url
        ):
            # Every real Salesforce token exchange includes a non-empty
            # string instance_url; anything else here (missing, empty, or a
            # non-string value) means the response is unusable for this
            # connector (launch_config.env_mapping requires it, so the
            # server would come back unavailable/reconnect-required on the
            # very next load) -- full host/scheme validation still only
            # happens at use-time in salesforce.py's _instance_url(), this
            # is just "is this even a plausible value to store" before
            # committing it. Checked before the delete-then-recreate below,
            # not after: that block unconditionally drops any existing
            # UserOAuth row for this user+provider first, so letting a bad
            # response through here would destroy a prior *working* grant
            # while still telling the user "Connected Successfully".
            return HTMLResponse(
                content=(
                    "<h1>Error exchanging token</h1>"
                    f"<p>{html.escape(provider)} did not return an instance_url.</p>"
                ),
                status_code=400,
            )

        provider_user_id = None
        email = None

        if provider.lower() == "linear":
            # Checked before the generic userinfo_url branch below (not
            # "elif" on it), not just as an ordering nicety: Linear's
            # provider row leaves userinfo_url empty today (GraphQL-only, no
            # flat REST endpoint fits that branch), but if userinfo_url were
            # ever populated on Linear's row (e.g. an admin edit), the
            # generic branch's REST GET would run instead, fail silently
            # against Linear's GraphQL-only API, and persist the connection
            # as "healthy" with no identity. Checking the provider name
            # first means this path always wins for Linear regardless of
            # what userinfo_url holds -- see _fetch_linear_viewer_identity's
            # docstring for why this is not just a label workaround.
            try:
                provider_user_id, email = _fetch_linear_viewer_identity(access_token)
            except RuntimeError as e:
                # A deliberate failure raised by _fetch_linear_viewer_identity
                # itself -- Linear's API responded, just not usably.
                return HTMLResponse(
                    content=(
                        "<h1>Error verifying the connected account</h1>"
                        f"<p>The provider reported: {html.escape(str(e))}</p>"
                    ),
                    status_code=400,
                )
            except Exception as e:
                # A network-level failure (timeout, connection error) --
                # distinct from the case above: Linear never actually
                # responded, so attributing this to "the provider reported"
                # would be misleading.
                return HTMLResponse(
                    content=(
                        "<h1>Error verifying the connected account</h1>"
                        f"<p>Could not reach Linear to verify the connection: "
                        f"{html.escape(str(e))}</p>"
                    ),
                    status_code=400,
                )
        elif _is_salesforce_provider(provider):
            # Salesforce's userinfo_url is deliberately left empty (see the
            # registry row's comment: an extra round-trip just for a label
            # this connector doesn't otherwise need), which also means
            # provider_user_id stays NULL here -- and UserOAuth's unique
            # constraint is (user_id, provider, provider_user_id), which
            # SQL treats as non-conflicting across multiple NULLs. Every
            # other provider gets real protection from that constraint
            # because their provider_user_id is a real value; Salesforce
            # would get none at all, letting concurrent callbacks for the
            # same user leave more than one row with no error. The token
            # response's own "id" field (Salesforce's identity URL, unique
            # per org+user) closes that gap for free -- no extra network
            # call, unlike a real userinfo lookup. This branch preempting
            # the generic `elif userinfo_url and access_token` branch below
            # is deliberate, not an oversight: it means the seeded row's
            # user_id_path/email_path columns are dead by construction for
            # Salesforce, which is fine -- Salesforce's userinfo endpoint is
            # still reachable (salesforce_get_current_user calls it
            # directly against the fixed USERINFO_URL host), it's just not
            # used for callback-time identity, on purpose.
            raw_provider_user_id = token_data.get("id")
            # Every real Salesforce token response's "id" is a non-empty
            # string URL; a non-string or empty value (a malformed/
            # proxy-mangled response) would otherwise get str()-ified into
            # the uniqueness key below instead of falling back to the same
            # NULL-tolerant path a missing "id" already takes.
            provider_user_id = (
                raw_provider_user_id
                if isinstance(raw_provider_user_id, str) and raw_provider_user_id
                else None
            )
            if raw_provider_user_id and provider_user_id is None:
                # Only a truthy-but-wrong-type "id" is anomalous enough to
                # warn about -- a falsy one (missing entirely) is the
                # already-expected, silent case every other provider using
                # this same fallback also hits.
                logger.warning(
                    'Salesforce token response\'s "id" field was not a '
                    "usable string (got %s); falling back to NULL "
                    "provider_user_id for this grant",
                    type(raw_provider_user_id).__name__,
                )
        elif userinfo_url and access_token:
            info_headers = {"Authorization": f"Bearer {access_token}"}
            # Replace {{access_token}} placeholder if present
            actual_url = userinfo_url.replace("{{access_token}}", access_token)
            info_response = requests.get(actual_url, headers=info_headers, timeout=10.0)
            if info_response.status_code != 200:
                # A non-200 userinfo response (401/403/429/5xx -- an
                # expired/insufficiently-scoped token, or the provider's
                # own outage) used to be silently skipped here, leaving
                # provider_user_id/email at None and falling through to
                # persist a "connected" account with no identity. On a
                # reconnect, that replaces a previously working grant with
                # a broken one while still reporting success to the user.
                logger.warning(
                    "OAuth userinfo fetch for provider=%s returned status %s",
                    provider,
                    info_response.status_code,
                )
                return HTMLResponse(
                    content=(
                        "<h1>Error verifying the connected account</h1>"
                        f"<p>{html.escape(provider)} could not be verified "
                        f"(status {info_response.status_code}). Please try "
                        "again.</p>"
                    ),
                    status_code=400,
                )
            try:
                info_data = info_response.json()
            except ValueError:
                # Same reasoning as the token-exchange guard above: a
                # non-JSON 200 body must not reach the unguarded .get()
                # calls below as an unhelpful raw exception, and must not
                # be silently treated as "no identity, proceed anyway."
                logger.warning(
                    "OAuth userinfo fetch for provider=%s returned a non-JSON response",
                    provider,
                )
                return HTMLResponse(
                    content=(
                        "<h1>Error verifying the connected account</h1>"
                        f"<p>{html.escape(provider)} returned a response "
                        "that could not be parsed. Please try again.</p>"
                    ),
                    status_code=400,
                )
            if not isinstance(info_data, dict):
                logger.warning(
                    "OAuth userinfo fetch for provider=%s returned a "
                    "non-object body (%s)",
                    provider,
                    type(info_data).__name__,
                )
                return HTMLResponse(
                    content=(
                        "<h1>Error verifying the connected account</h1>"
                        f"<p>{html.escape(provider)} returned an "
                        "unexpected response. Please try again.</p>"
                    ),
                    status_code=400,
                )
            if info_data.get("ok") is False:
                # Slack-style APIs answer HTTP 200 with {"ok": false,
                # "error": ...} on failure; a status check alone would
                # treat a bad/revoked token as success and persist a
                # "connected" account with no identity. Fail the
                # callback instead. Providers without Slack semantics
                # never carry an "ok" key, so they are unaffected.
                escaped_error = html.escape(
                    str(info_data.get("error") or "unknown error")
                )
                return HTMLResponse(
                    content=(
                        "<h1>Error verifying the connected account</h1>"
                        f"<p>The provider reported: {escaped_error}</p>"
                    ),
                    status_code=400,
                )
            provider_user_id = info_data.get(db_provider.user_id_path or "id")
            email = info_data.get(db_provider.email_path or "email")

        if user_id:
            delete_scoped_user_oauth_accounts(
                db,
                user_id=user_id,
                resource_owner_key=None,
                providers=[app_id or provider],
            )

            oauth_account = UserOAuth(
                user_id=user_id,
                provider=(app_id or provider),
                resource_owner_key=None,
                provider_user_id=str(provider_user_id) if provider_user_id else None,
            )
            db.add(oauth_account)

            setattr(oauth_account, "access_token", access_token)
            setattr(oauth_account, "token_type", token_data.get("token_type", "Bearer"))
            # Most providers return "scope" as a single space/comma-joined
            # string, but Linear OAuth applications created before December
            # 1, 2023 return it as a list of strings -- UserOAuth.scope is a
            # plain String column, so committing a list there would raise
            # at flush time instead of saving a valid connection. Always
            # join with a space regardless of provider: `_oauth_scope_separator`
            # governs only the outbound authorize-request format (comma for
            # Linear/Meta), and the two readers of this column already split
            # on a space, so reusing that separator here would make the
            # stored format provider-dependent and silently mis-parse every
            # Linear row wherever this column is read.
            token_scope = token_data.get("scope", "")
            if isinstance(token_scope, list):
                token_scope = " ".join(str(scope) for scope in token_scope)
            setattr(oauth_account, "scope", token_scope)
            setattr(oauth_account, "email", email)
            if "refresh_token" in token_data:
                setattr(oauth_account, "refresh_token", token_data.get("refresh_token"))
            # Salesforce returns the per-org API host here instead of using a
            # fixed domain -- no other provider sends this key, and
            # oauth_account is freshly created above (never an update to an
            # existing row), so token_data.get() returning None for every
            # other provider is already the correct, final value: no `if
            # "instance_url" in token_data` guard needed to avoid clobbering
            # anything.
            setattr(oauth_account, "instance_url", token_data.get("instance_url"))
            if "expires_in" in token_data:
                setattr(
                    oauth_account,
                    "expires_at",
                    datetime.now(timezone.utc)
                    + timedelta(seconds=int(token_data["expires_in"])),
                )

            from ..mcp_apps import get_all_mcp_apps
            from .mcp import _reject_hidden_catalog_app

            if app_id:
                # Reuse target_app_info from the earlier hidden-app-gate
                # check above rather than re-fetching by app_id: nothing
                # mutates public_mcp_apps between there and here, so a second
                # fetch would just be redundant, not more correct.
                app_info = target_app_info
                if app_info:
                    # A stale/crafted app_id in the OAuth state can point at a
                    # non-oauth app. Fail with a clear error instead of a generic
                    # 500 after the user already completed provider consent —
                    # symmetric with the batch branch's AppNotOAuthError catch.
                    try:
                        _ensure_user_mcp_server(db, user_id, app_info)
                    except AppNotOAuthError:
                        db.rollback()
                        return HTMLResponse(
                            content=(
                                "<h1>Cannot Connect</h1>"
                                "<p>This app is not an OAuth app.</p>"
                            ),
                            status_code=400,
                        )
            else:
                from ..mcp_apps import requires_app_scoped_oauth_grant

                apps = [
                    app
                    for app in get_all_mcp_apps(db)
                    if app.get("provider") == provider
                ]
                for app_info in apps:
                    # Same release-gate enforcement as the single-app branch
                    # above (same shared helper, so there is one source of
                    # truth for this check), but a hidden app must not abort
                    # the whole bare batch connect — skip it and keep
                    # connecting the other visible apps under the same
                    # provider, matching the mis-tagged-app skip below.
                    try:
                        _reject_hidden_catalog_app(app_info)
                    except HTTPException:
                        logger.info(
                            "Skipping hidden app %s during bare %s OAuth batch connect",
                            app_info.get("id"),
                            provider,
                        )
                        continue
                    # This bare app_id-less login only ever requests
                    # db_provider.default_scopes (see the app_scopes=None
                    # branch above), never an app's own oauth_scopes. Creating
                    # a UserMCPServer row here for an app that requires an
                    # app-scoped grant would leave an orphan the agent runtime
                    # picks up directly (bypassing the connected-state check)
                    # and can never resolve a token for; see
                    # APPS_REQUIRING_APP_SCOPED_OAUTH_GRANT.
                    if requires_app_scoped_oauth_grant(app_info.get("id")):
                        logger.info(
                            "Skipping app-scoped-only app %s during bare %s "
                            "OAuth batch connect",
                            app_info.get("id"),
                            provider,
                        )
                        continue
                    # A mis-tagged non-oauth app sharing this provider must not
                    # abort the whole batch login: skip it and keep connecting
                    # the legitimate oauth apps under the same provider. Only the
                    # not-oauth case is skipped; genuine metadata-conflict
                    # ValueErrors still propagate as before.
                    try:
                        _ensure_user_mcp_server(db, user_id, app_info)
                    except AppNotOAuthError:
                        logger.warning(
                            "Skipping non-oauth app %s during %s OAuth batch connect",
                            app_info.get("id"),
                            provider,
                        )

            db.commit()
            # Everything past the commit belongs in the helper, which cannot
            # change what this callback returns. Add new post-commit work
            # there, not here.
            _run_post_commit_oauth_side_effects(
                db, user_id=user_id, connector_key=(app_id or provider)
            )

        from urllib.parse import urlparse

        redirect_url = payload.get("redirect")
        target_origin = "window.location.origin"
        if redirect_url:
            try:
                parsed = urlparse(redirect_url)
                if parsed.scheme and parsed.netloc:
                    target_origin = json.dumps(f"{parsed.scheme}://{parsed.netloc}")
            except Exception:
                pass

        return HTMLResponse(
            content=f"""
        <html>
            <head>
                <title>Connected</title>
                <script>
                    window.opener.postMessage({{
                        type: 'oauth-success',
                        email: {json.dumps(email)},
                        provider: {json.dumps(app_id or provider)}
                    }}, {target_origin});
                    window.close();
                </script>
            </head>
            <body>
                <h1>Connected Successfully</h1>
                <p>You can close this window now.</p>
            </body>
        </html>
        """
        )
    except Exception:
        # str(e) is not rendered to the client: db.add(oauth_account)/db.commit()
        # above persist the just-obtained access/refresh token as bound SQL
        # parameters, and a SQLAlchemy StatementError's default __str__
        # includes those bound values -- a DB error here (constraint
        # violation, connection drop, oversized field) would otherwise echo
        # the plaintext token back to the browser in this 500 response.
        # hide_parameters=True on the engine (models/database.py) now hides
        # it there too, but this handler doesn't rely on that alone.
        # logger.exception still captures it server-side for debugging.
        logger.exception("Generic OAuth callback failed")
        return HTMLResponse(
            content="<h1>Authentication Failed</h1><p>An unexpected error occurred "
            "while connecting this account. Please try again.</p>",
            status_code=500,
        )


# --- Unified OAuth Routes ---


@auth_router.get("/{provider}/login")
def oauth_login(
    provider: str,
    token: Optional[str] = None,
    app_id: Optional[str] = None,
    redirect: Optional[str] = None,
    db: Session = Depends(get_db),
) -> Any:
    """Unified entry point for OAuth login"""
    from ..models.oauth_provider import OAuthProvider

    db_provider = (
        db.query(OAuthProvider).filter(OAuthProvider.provider_name == provider).first()
    )
    if not db_provider:
        return HTMLResponse(
            content=f"<h1>Unsupported provider: {provider}</h1>", status_code=400
        )

    # But now everything can be routed through generic
    return generic_oauth_login(provider, token, app_id, redirect, db, db_provider)


@auth_router.get("/{provider}/callback")
def oauth_callback(
    provider: str, request: Request, db: Session = Depends(get_db)
) -> Any:
    """Unified entry point for OAuth callback"""
    from ..models.oauth_provider import OAuthProvider

    db_provider = (
        db.query(OAuthProvider).filter(OAuthProvider.provider_name == provider).first()
    )
    if not db_provider:
        return HTMLResponse(
            content=f"<h1>Unsupported provider: {provider}</h1>", status_code=400
        )

    return generic_oauth_callback(provider, request, db, db_provider)


from .oidc_google import router as google_oidc_router  # noqa: E402

auth_router.include_router(google_oidc_router)
