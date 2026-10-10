"""Cloud Storage API Endpoints"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ...core.utils.encryption import decrypt_value
from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.oauth_provider import OAuthProvider
from ..models.user import User
from ..models.user_oauth import UserOAuth
from ..services.google_picker import (
    classify_google_drive_picker_scopes,
    get_google_picker_config,
    google_picker_app_id_matches_client,
)
from ..services.user_oauth import (
    get_scoped_user_oauth_account,
    normalize_user_oauth_resource_owner_key,
    scoped_user_oauth_query,
)

logger = logging.getLogger(__name__)

cloud_router = APIRouter(prefix="/api/cloud", tags=["Cloud Storage"])

# Google OAuth Constants
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"
GOOGLE_TOKEN_REFRESH_SKEW = timedelta(minutes=5)
# A token refresh for a resource owner's connection sets a 10-second connect
# and per-read timeout on each request; google-auth's retries can make the
# whole refresh longer. The connector runtime sets the same timeout when it
# refreshes these credentials (``tools/config.py``); google-auth's own
# default is two minutes.
_RESOURCE_OWNER_REFRESH_TIMEOUT = 10.0
# ``UserOAuth.provider`` of a Google Drive connection (its connector app id).
_GOOGLE_DRIVE_PROVIDER = "google-drive"
# The OAuth error code for a refresh token that was revoked or has expired.
# Google also answers ``invalid_grant`` (with an ``error_subtype`` such as
# ``invalid_rapt``) when the user must reauthenticate.
_GOOGLE_INVALID_GRANT = "invalid_grant"

# Response details shared by several failure reasons. They are part of the
# public API contract of the routes below and must not change.
_DRIVE_NOT_CONNECTED_DETAIL = "Google Drive account not connected"
_DRIVE_ACCOUNT_NOT_FOUND_DETAIL = "Selected Google Drive account not found"
_DRIVE_RECONNECT_DETAIL = "Google Drive session expired. Please reconnect."
_OAUTH_CONFIG_MISSING_DETAIL = "Google OAuth configuration missing"
_PICKER_NOT_CONFIGURED_DETAIL = (
    "Google Drive Picker is not configured. Set the dedicated, "
    "referrer-restricted GOOGLE_PICKER_API_KEY and either "
    "GOOGLE_PICKER_APP_ID or a numeric Google OAuth client_id. "
    "The access token and Picker key are sent to the browser."
)
_PICKER_SCOPE_DETAIL = (
    "This Google Drive connection uses an outdated permission. "
    "Reconnect it before opening Google Drive Picker."
)

GoogleDriveCredentialReason = Literal[
    # Picker key or app id missing.
    "picker_not_configured",
    # No stored Google Drive credential in the requested owner namespace.
    "account_not_connected",
    # ``account_id`` does not name a credential in the requested namespace.
    "account_not_found",
    # The credential cannot be used again without a new authorization: no
    # access token, no refresh token when a refresh is due, or the token
    # endpoint's last answer to the refresh was ``invalid_grant`` (revoked or
    # expired, or reauthentication needed). Nothing is cleared.
    "reauth_required",
    # Any other refresh failure: transport error or timeout, retryable or
    # 5xx answer, a body that is not an OAuth error, another OAuth error
    # code (for example ``invalid_client``), an unexpected exception, a
    # stored credential that could not be locked for the refresh, or a
    # failed commit of the refreshed token.
    "refresh_unavailable",
    # No Google OAuth client id/secret is configured.
    "oauth_unconfigured",
    # The grant includes full Drive access, so there is nothing to pick.
    "scope_full_drive",
    # Unexpected mix of Drive scopes (or extra scopes on a minimal grant).
    "scope_mismatch",
    # The grant carries no Drive scope at all.
    "scope_drive_missing",
]


class GoogleDriveCredentialError(HTTPException):
    """HTTP error raised while resolving Google Drive credentials.

    ``status_code`` and ``detail`` are exactly what the ``/api/cloud`` routes
    return. ``reason`` is a stable, machine-readable classification for
    in-process callers. ``oauth_account_id`` names the stored credential row
    the failure concerns: it is set on every failure about a row that was
    found, except a scope error for the user's own connection (see
    ``issue_google_drive_picker_config``), and ``None`` when no row was found,
    including a row that was gone by the time it was locked for a refresh.
    Neither is sent to clients.
    """

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        reason: GoogleDriveCredentialReason,
        oauth_account_id: int | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.reason: GoogleDriveCredentialReason = reason
        self.oauth_account_id = oauth_account_id


def _google_credentials_expiry(value: datetime | None) -> datetime | None:
    """Return the naive UTC datetime required by google-auth.

    ``UserOAuth.expires_at`` is timezone-aware on PostgreSQL but google-auth
    compares ``Credentials.expiry`` with a naive UTC value.  Normalize here so
    expired Drive tokens refresh before they are handed to the API or Picker.
    """
    if value is None or value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _google_database_expiry(value: datetime | None) -> datetime | None:
    """Return an aware UTC datetime for ``UserOAuth.expires_at``."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _google_token_needs_refresh(
    creds: Credentials, min_ttl: timedelta = GOOGLE_TOKEN_REFRESH_SKEW
) -> bool:
    """Refresh a token before Picker/Drive calls get close to its expiry."""
    if creds.expired:
        return True
    expiry_value = getattr(creds, "expiry", None)
    if expiry_value is None:
        return False
    expiry = _google_credentials_expiry(expiry_value)
    if expiry is None:
        return False
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return expiry - now <= min_ttl


def get_google_oauth_config(db: Session) -> tuple[Optional[str], Optional[str]]:
    """Load Google OAuth client credentials from admin provider config."""
    provider = (
        db.query(OAuthProvider).filter(OAuthProvider.provider_name == "google").first()
    )
    if not provider:
        return None, None

    client_id = cast(str, provider.client_id)
    client_secret = cast(str, provider.client_secret)
    return decrypt_value(client_id), decrypt_value(client_secret)


def _resolve_google_oauth_client_per_field(db: Session) -> tuple[str, str]:
    """Resolve the Google OAuth client the way the connector runtime does.

    Each field falls back to its ``GOOGLE_*`` environment variable on its own
    when the provider row leaves it blank, through the helper the runtime's
    token refresh uses, and a missing provider row means the client is not
    configured.
    """
    from .auth import _resolve_oauth_client_per_field

    provider = (
        db.query(OAuthProvider).filter(OAuthProvider.provider_name == "google").first()
    )
    if provider is None:
        return "", ""
    return _resolve_oauth_client_per_field("google", provider)


def _oauth_account_row_id(oauth_account: Any) -> int | None:
    value = getattr(oauth_account, "id", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _google_drive_query(
    db: Session,
    *,
    user_id: int,
    account_id: Optional[int],
    resource_owner_key: str | None,
) -> Any:
    query = scoped_user_oauth_query(
        db,
        user_id=user_id,
        resource_owner_key=resource_owner_key,
    ).filter(UserOAuth.provider == _GOOGLE_DRIVE_PROVIDER)
    if account_id is not None:
        query = query.filter(UserOAuth.id == account_id)
    return query


def _require_usable_google_drive_row(
    oauth_account: Any, *, account_id: Optional[int]
) -> int | None:
    """Return the id of a stored credential that has an access token, or raise."""
    if not oauth_account:
        if account_id is not None:
            raise GoogleDriveCredentialError(
                status_code=404,
                detail=_DRIVE_ACCOUNT_NOT_FOUND_DETAIL,
                reason="account_not_found",
            )
        raise GoogleDriveCredentialError(
            status_code=401,
            detail=_DRIVE_NOT_CONNECTED_DETAIL,
            reason="account_not_connected",
        )
    oauth_account_id = _oauth_account_row_id(oauth_account)
    if not oauth_account.access_token:
        raise GoogleDriveCredentialError(
            status_code=401,
            detail=_DRIVE_RECONNECT_DETAIL,
            reason="reauth_required",
            oauth_account_id=oauth_account_id,
        )
    return oauth_account_id


def _google_credentials_from_row(
    oauth_account: Any,
    client_id: Optional[str],
    client_secret: Optional[str],
    *,
    oauth_account_id: int | None,
) -> Any:
    if not client_id or not client_secret:
        raise GoogleDriveCredentialError(
            status_code=500,
            detail=_OAUTH_CONFIG_MISSING_DETAIL,
            reason="oauth_unconfigured",
            oauth_account_id=oauth_account_id,
        )
    return Credentials(
        token=oauth_account.access_token,
        refresh_token=oauth_account.refresh_token,
        token_uri=GOOGLE_TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=oauth_account.scope.split(" ") if oauth_account.scope else None,
        expiry=_google_credentials_expiry(
            cast("datetime | None", oauth_account.expires_at)
        ),
    )


def get_google_credentials(
    user_id: int,
    db: Session,
    account_id: Optional[int] = None,
    *,
    resource_owner_key: str | None = None,
    min_ttl: timedelta = GOOGLE_TOKEN_REFRESH_SKEW,
) -> Any:
    """Get Google Credentials for user, refreshing if necessary.

    By default this reads the user's own Google Drive connection, as the
    ``/api/cloud`` routes always have. With ``resource_owner_key`` it reads
    the credential stored for that key under ``user_id`` (a delegated
    connection) instead. That credential is handled the way the connector
    runtime refreshes it:

    * the newest row wins;
    * each field of the OAuth client falls back to its environment variable
      on its own;
    * a refresh sets a 10-second connect and per-read timeout on each request
      (retries can make the whole refresh longer);
    * a due refresh first takes the runtime's lock on the stored row.

    A failed refresh is classified here, not as the runtime does: an
    ``invalid_grant`` answer is ``reauth_required`` (see
    ``GoogleDriveCredentialReason``), which is safe because nothing is
    cleared here.

    This function belongs to the cross-repository contract documented on
    ``issue_google_drive_picker_config``. That covers its signature (``user_id``,
    ``db`` and ``account_id`` positional or keyword, the rest keyword-only), its
    reasons, the ``status_code`` and ``detail`` of every error (unchanged from
    the website routes), and the caller's precondition. Nothing is authorized
    here, and the result carries a raw access token.

    ``min_ttl`` is a refresh threshold: a token that expires within it is
    refreshed first (google-auth also treats a token as expired a few minutes
    before its expiry). Nothing checks how long the refreshed token lives,
    and a token stored without an expiry is never refreshed.

    The session is committed after a refresh stores the new token, and it is
    rolled back when storing the token fails. For a resource owner's
    connection a due refresh also locks the stored row, and the transaction
    always ends before this returns or raises: it is committed after a stored
    refresh and rolled back in every other case. In-process callers should
    therefore pass a dedicated session.

    Failures raise ``GoogleDriveCredentialError``; stored credentials are never
    cleared here. ``resource_owner_key`` is stripped. A blank or oversized key,
    or a ``min_ttl`` that is not a non-negative ``timedelta``, raises
    ``ValueError`` (a programming error) before anything is read.
    """
    owner_key = normalize_user_oauth_resource_owner_key(resource_owner_key)
    _require_min_ttl(min_ttl)
    if owner_key is not None:
        creds, _oauth_account_id = _resource_owner_google_credentials(
            user_id,
            db,
            account_id,
            resource_owner_key=owner_key,
            min_ttl=min_ttl,
        )
        return creds
    return _own_google_credentials(user_id, db, account_id, min_ttl=min_ttl)


def _require_min_ttl(min_ttl: timedelta) -> None:
    if not isinstance(min_ttl, timedelta) or min_ttl < timedelta(0):
        raise ValueError("min_ttl must be a non-negative timedelta")


def _own_google_credentials(
    user_id: int, db: Session, account_id: Optional[int], *, min_ttl: timedelta
) -> Any:
    """Read the user's own connection, as the ``/api/cloud`` routes always have."""
    oauth_account = _google_drive_query(
        db, user_id=user_id, account_id=account_id, resource_owner_key=None
    ).first()
    oauth_account_id = _require_usable_google_drive_row(
        oauth_account, account_id=account_id
    )

    client_id, client_secret = get_google_oauth_config(db)
    if not client_id or not client_secret:
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    creds = _google_credentials_from_row(
        oauth_account, client_id, client_secret, oauth_account_id=oauth_account_id
    )

    # Refresh with a safety margin: the Picker request and user interaction
    # can consume several minutes, so returning a token that is technically
    # valid but close to expiry creates an avoidable mid-flow failure.
    if _google_token_needs_refresh(creds, min_ttl):
        _refresh_google_credentials(
            creds, oauth_account, db, oauth_account_id=oauth_account_id
        )
    return creds


def _resource_owner_google_credentials(
    user_id: int,
    db: Session,
    account_id: Optional[int],
    *,
    resource_owner_key: str,
    min_ttl: timedelta,
    oauth_client: tuple[str, str] | None = None,
) -> tuple[Any, int | None]:
    """Read the credential stored for ``resource_owner_key`` (a delegated one).

    Returns the credentials and the id of the row they were read from.
    ``oauth_client`` is the client from ``_resolve_google_oauth_client_per_field``
    when the caller has already resolved it.
    """
    # Delegated connections are replaced by delete-and-insert, so the newest
    # row is the live one. ``populate_existing`` refreshes a row the caller's
    # session may already hold from an earlier read.
    oauth_account = (
        _google_drive_query(
            db,
            user_id=user_id,
            account_id=account_id,
            resource_owner_key=resource_owner_key,
        )
        .order_by(UserOAuth.id.desc())
        .populate_existing()
        .first()
    )
    oauth_account_id = _require_usable_google_drive_row(
        oauth_account, account_id=account_id
    )
    if oauth_client is None:
        oauth_client = _resolve_google_oauth_client_per_field(db)
    creds = _google_credentials_from_row(
        oauth_account, *oauth_client, oauth_account_id=oauth_account_id
    )
    if not _google_token_needs_refresh(creds, min_ttl):
        return creds, oauth_account_id

    # A refresh is due. Serialize it with every other refresher of this
    # credential, the connector runtime's included: Google may rotate the
    # refresh token, and an overlapping refresh would store a token the other
    # one has already replaced. The runtime also orders its own refreshers
    # with an asyncio.Lock (``_actor_oauth_refresh_lock`` in
    # ``tools/config.py``); that lock belongs to one event loop and cannot be
    # taken from this synchronous code, so this relies on the row lock alone.
    # The runtime takes the same row lock, and it holds across threads,
    # event loops and processes.
    try:
        oauth_account = _lock_resource_owner_google_drive_row(
            db,
            user_id=user_id,
            account_id=account_id,
            resource_owner_key=resource_owner_key,
        )
    except SQLAlchemyError as exc:
        logger.error(
            "Failed to lock the stored Google Drive credential: %s",
            _exception_label(exc),
        )
        _rollback_quietly(db)
        raise GoogleDriveCredentialError(
            status_code=401,
            detail=_DRIVE_RECONNECT_DETAIL,
            reason="refresh_unavailable",
            oauth_account_id=oauth_account_id,
        ) from exc
    try:
        # Read under the lock. While this one waited, another refresher may
        # have stored a new token, a reconnect may have replaced the row, or a
        # disconnect may have deleted it.
        oauth_account_id = _require_usable_google_drive_row(
            oauth_account, account_id=account_id
        )
        creds = _google_credentials_from_row(
            oauth_account, *oauth_client, oauth_account_id=oauth_account_id
        )
        if _google_token_needs_refresh(creds, min_ttl):
            # Commits ``db``, which releases the lock.
            _refresh_google_credentials(
                creds,
                oauth_account,
                db,
                oauth_account_id=oauth_account_id,
                timeout=_RESOURCE_OWNER_REFRESH_TIMEOUT,
            )
            return creds, oauth_account_id
    except Exception:
        if db.in_transaction():
            _rollback_quietly(db)
        raise
    # Another refresher already stored a token that is fresh enough. Nothing
    # was changed here; end the transaction to release the lock.
    _rollback_quietly(db)
    return creds, oauth_account_id


def _lock_resource_owner_google_drive_row(
    db: Session,
    *,
    user_id: int,
    account_id: Optional[int],
    resource_owner_key: str,
) -> Any:
    """Lock a resource owner's Google Drive credential and read it again.

    This is the lock the connector runtime takes before it refreshes the same
    credential (``_resolve_actor_oauth_access_token_in_worker`` in
    ``tools/config.py``): ``SELECT ... FOR UPDATE`` on the newest row, or on
    SQLite a no-op ``UPDATE`` of the owner's rows, which takes the database
    write lock. It is held until ``db``'s transaction ends.

    Returns ``None`` when no row is left, for example after a disconnect.
    """
    query = _google_drive_query(
        db,
        user_id=user_id,
        account_id=account_id,
        resource_owner_key=resource_owner_key,
    ).order_by(UserOAuth.id.desc())
    if db.get_bind().dialect.name == "sqlite":
        db.execute(
            text(
                "UPDATE user_oauth SET id = id "
                "WHERE user_id = :user_id "
                "AND resource_owner_key = :resource_owner_key "
                "AND provider = :provider"
            ),
            {
                "user_id": user_id,
                "resource_owner_key": resource_owner_key,
                "provider": _GOOGLE_DRIVE_PROVIDER,
            },
        )
        # No writer can commit after the lock is taken, so this read sees
        # every row a reconnect committed before it.
        return query.populate_existing().first()
    locked = query.with_for_update().populate_existing()
    oauth_account = locked.first()
    if oauth_account is None:
        # A reconnect deletes the row and inserts a new one. When it commits
        # while this statement waits for the row lock, PostgreSQL (READ
        # COMMITTED) skips the deleted row, and the new row is not in the
        # snapshot this statement started with. A second statement takes a
        # new snapshot, so it finds the new row and locks it.
        oauth_account = locked.first()
    return oauth_account


def _rollback_quietly(db: Session) -> None:
    try:
        db.rollback()
    except Exception as exc:
        logger.error(
            "Failed to roll back a Google Drive credential transaction: %s",
            _exception_label(exc),
        )


class _GoogleTokenRequest(Request):
    """google-auth HTTP transport for one token refresh.

    It records the status and body of the token endpoint's last response, so
    a failed refresh is classified by what Google answered rather than by how
    google-auth words its exception. Both are ``None`` when the last request
    got no response. With ``timeout``, each request gets that many seconds to
    connect and for each read (retries can make the whole refresh longer);
    without it google-auth's default applies.
    """

    def __init__(self, *, timeout: float | None = None) -> None:
        super().__init__()
        self._timeout = timeout
        self.last_status: int | None = None
        self.last_body: bytes | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # google-auth passes the timeout, like every other argument, by
        # keyword; without one of our own, its default stays in place.
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        self.last_status = None
        self.last_body = None
        response = super().__call__(*args, **kwargs)
        self.last_status = response.status
        self.last_body = response.data
        return response


def _oauth_error_code(body: bytes | None) -> str | None:
    """Return the OAuth ``error`` code of a token endpoint response body.

    Only a body that is a JSON object has one; a gateway's HTML page or any
    other body has none.
    """
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    return error if isinstance(error, str) else None


def _google_refresh_failure_reason(
    exc: Exception, *, status: int | None, error_code: str | None
) -> GoogleDriveCredentialReason:
    """Classify a failed google-auth refresh.

    ``status`` and ``error_code`` describe the token endpoint's last answer
    (both ``None`` when the last request got none). Only an ``invalid_grant``
    answer means the user has to authorize again. Everything else is
    ``refresh_unavailable``: transport errors (including timeouts), a
    retryable or 5xx answer, a body that is not an OAuth error (for example a
    gateway's HTML page), other OAuth error codes such as ``invalid_client``
    or ``unauthorized_client`` that a new authorization by the user cannot
    fix, and unexpected exceptions.
    """
    if not isinstance(exc, RefreshError):
        return "refresh_unavailable"
    if exc.retryable or status is None or status >= 500:
        return "refresh_unavailable"
    if error_code == _GOOGLE_INVALID_GRANT:
        # The connector runtime does not treat invalid_grant as a dead
        # refresh token (``_PROVIDER_DEAD_REFRESH_TOKEN_ERROR_CODES`` in
        # ``tools/config.py``): it also covers a token issued to another
        # client, for example after an admin rotates the OAuth client, and
        # the runtime clears credentials it believes dead. Nothing is ever
        # cleared on this path; the reason only selects the message a caller
        # shows (reconnect), and the stored credential stays for the
        # runtime's own refresh to judge.
        return "reauth_required"
    return "refresh_unavailable"


def _exception_label(exc: BaseException) -> str:
    cause = exc.__cause__
    if cause is None:
        return type(exc).__name__
    return f"{type(exc).__name__}({type(cause).__name__})"


def _refresh_google_credentials(
    creds: Any,
    oauth_account: Any,
    db: Session,
    *,
    oauth_account_id: int | None,
    timeout: float | None = None,
) -> None:
    """Refresh ``creds`` and persist the new token on ``oauth_account``.

    ``timeout`` is the connect and per-read timeout, in seconds, set on each
    request of the refresh (retries can make the whole refresh longer);
    ``None`` keeps google-auth's default. A failure raises
    ``GoogleDriveCredentialError`` and leaves the stored credential as it
    was; without a refresh token the user has to authorize again.
    """
    if not creds.refresh_token:
        raise GoogleDriveCredentialError(
            status_code=401,
            detail=_DRIVE_RECONNECT_DETAIL,
            reason="reauth_required",
            oauth_account_id=oauth_account_id,
        )
    request = _GoogleTokenRequest(timeout=timeout)
    try:
        creds.refresh(request)
    except Exception as exc:
        error_code = _oauth_error_code(request.last_body)
        reason = _google_refresh_failure_reason(
            exc, status=request.last_status, error_code=error_code
        )
        # Only the classification inputs are logged: the exception text and
        # the response body can carry a whole gateway page.
        logger.error(
            "Failed to refresh Google token (%s): %s, token endpoint status %s, "
            "error %s, retryable %s",
            reason,
            _exception_label(exc),
            request.last_status,
            error_code,
            getattr(exc, "retryable", None),
        )
        raise GoogleDriveCredentialError(
            status_code=401,
            detail=_DRIVE_RECONNECT_DETAIL,
            reason=reason,
            oauth_account_id=oauth_account_id,
        ) from exc

    try:
        setattr(oauth_account, "access_token", creds.token)
        if creds.expiry:
            setattr(
                oauth_account,
                "expires_at",
                _google_database_expiry(creds.expiry),
            )
        refreshed_refresh_token = getattr(creds, "refresh_token", None)
        if (
            refreshed_refresh_token
            and refreshed_refresh_token != oauth_account.refresh_token
        ):
            # Google may rotate the refresh token; keep the one it issued.
            setattr(oauth_account, "refresh_token", refreshed_refresh_token)
        db.commit()
    except Exception as exc:
        # For example the row was replaced by a concurrent reconnect. Clear
        # the failed transaction so the caller's session stays usable; a
        # later attempt reads the current row again. Only the exception type
        # is logged: a database error's text can carry the statement's
        # parameters, which include the new token.
        logger.error(
            "Failed to store refreshed Google token: %s", _exception_label(exc)
        )
        _rollback_quietly(db)
        raise GoogleDriveCredentialError(
            status_code=401,
            detail=_DRIVE_RECONNECT_DETAIL,
            reason="refresh_unavailable",
            oauth_account_id=oauth_account_id,
        ) from exc


@cloud_router.get("/accounts")
async def list_connected_accounts(
    provider: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List connected cloud accounts"""
    query = scoped_user_oauth_query(
        db,
        user_id=cast(int, user.id),
        resource_owner_key=None,
    )

    if provider:
        query = query.filter(UserOAuth.provider == provider)

    query = query.filter(UserOAuth.access_token != "")

    accounts = query.all()

    return [
        {
            "id": acc.id,
            "provider": acc.provider,
            "email": acc.email,
            "created_at": acc.created_at,
        }
        for acc in accounts
    ]


_PICKER_SCOPE_REASONS: dict[str, GoogleDriveCredentialReason] = {
    "full_drive": "scope_full_drive",
    "drive_missing": "scope_drive_missing",
    "other": "scope_mismatch",
}


def issue_google_drive_picker_config(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None = None,
    account_id: int | None = None,
    minimal_scopes: bool = False,
    min_ttl: timedelta = GOOGLE_TOKEN_REFRESH_SKEW,
) -> Dict[str, str]:
    """Return the short-lived credentials needed by Google's file picker.

    The result is exactly ``{"access_token", "developer_key", "app_id"}``. The
    refresh token and client secret never leave the server.

    The website route calls this with ``user_id`` and ``account_id`` only. An
    application that embeds xagent and serves delegated connections calls it
    in process with ``resource_owner_key``, so what follows is a contract
    across repositories:

    * Signature: ``db`` positional or keyword, every other argument
      keyword-only, with the defaults shown.
    * Errors: ``GoogleDriveCredentialError``. Its ``reason`` is one of
      ``GoogleDriveCredentialReason``, and its ``status_code`` and ``detail``
      are the ones the website route has always returned. Invalid arguments
      raise ``ValueError`` (a programming error) before anything else.
    * Check order: the Picker configuration first, before any credential is
      read; then the credential; then the scopes.
    * Precondition: the caller must already have authenticated the browser
      principal as the owner of the connection, meaning ``user_id`` and, when
      given, ``resource_owner_key``. This function does no authorization of
      its own and returns a raw access token.
    * Session: when a refresh is due, ``db`` is committed or rolled back as
      ``get_google_credentials`` describes. Pass a dedicated session.

    ``resource_owner_key`` reads the credential stored for that resource owner
    key (a delegated connection) instead of the user's own connection. The
    key is stripped, and a blank or oversized key raises ``ValueError``.
    ``minimal_scopes`` additionally requires that the grant carries nothing
    besides ``drive.file`` and basic identity scopes. ``min_ttl`` is the
    refresh threshold that ``get_google_credentials`` describes, not a
    guaranteed remaining lifetime of the returned token.
    """
    owner_key = normalize_user_oauth_resource_owner_key(resource_owner_key)
    _require_min_ttl(min_ttl)
    # Fail closed before touching account credentials. An unconfigured
    # deployment should consistently report 503 rather than leaking account
    # state through a 401/409 response.
    oauth_client_id: Optional[str]
    owner_client: tuple[str, str] | None = None
    if owner_key is None:
        oauth_client_id, _ = get_google_oauth_config(db)
    else:
        # Resolved once: the credential below is built with the same client.
        owner_client = _resolve_google_oauth_client_per_field(db)
        oauth_client_id = owner_client[0] or None
    picker_config = get_google_picker_config(db, oauth_client_id=oauth_client_id)
    if picker_config is None:
        raise GoogleDriveCredentialError(
            status_code=503,
            detail=_PICKER_NOT_CONFIGURED_DETAIL,
            reason="picker_not_configured",
        )
    oauth_account_id: int | None
    if owner_key is None:
        # The website route: the same call it has always made. That call
        # returns only the credentials, so a scope error below cannot name
        # the row; the route never reads it.
        creds = get_google_credentials(user_id, db, account_id, min_ttl=min_ttl)
        oauth_account_id = None
    else:
        creds, oauth_account_id = _resource_owner_google_credentials(
            user_id,
            db,
            account_id,
            resource_owner_key=owner_key,
            min_ttl=min_ttl,
            oauth_client=owner_client,
        )
    # Warns (once per pair) when the app id points at another project than
    # the OAuth client the token was issued with. ``creds.client_id`` is that
    # client after any environment fallback, which can differ from the
    # provider row's client id read above.
    google_picker_app_id_matches_client(
        picker_config.app_id, getattr(creds, "client_id", None)
    )
    scope_state = classify_google_drive_picker_scopes(
        creds.scopes, minimal=minimal_scopes
    )
    if scope_state != "drive_file":
        raise GoogleDriveCredentialError(
            status_code=409,
            detail=_PICKER_SCOPE_DETAIL,
            reason=_PICKER_SCOPE_REASONS[scope_state],
            oauth_account_id=oauth_account_id,
        )

    return {
        "access_token": str(creds.token),
        "developer_key": picker_config.developer_key,
        "app_id": picker_config.app_id,
    }


@cloud_router.get("/google-drive/picker-config")
async def get_google_drive_picker_config(
    response: Response,
    account_id: Optional[int] = Query(None, gt=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, str]:
    """Return the short-lived credentials needed by Google's file picker.

    ``drive.file`` deliberately exposes only files selected in Picker (or
    created by the app).  The browser therefore needs a Picker access token
    before the Drive browser can operate on an existing file or folder.  The
    refresh token and client secret never leave the server; the returned access
    token is scoped to the authenticated user and expires normally.
    """
    response.headers["Cache-Control"] = "no-store"
    return issue_google_drive_picker_config(
        db, user_id=cast(int, user.id), account_id=account_id
    )


@cloud_router.get("/google-drive/picker-availability")
async def get_google_drive_picker_availability(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, bool]:
    """Expose Picker configuration readiness without touching OAuth tokens."""
    del user
    return {"configured": get_google_picker_config(db) is not None}


@cloud_router.delete("/accounts/{account_id}")
async def delete_connected_account(
    account_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a connected cloud account"""
    account = get_scoped_user_oauth_account(
        db,
        user_id=cast(int, user.id),
        account_id=account_id,
        resource_owner_key=None,
    )

    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    # Snapshot before the delete: it's a bulk-free single-row delete here,
    # but the same "read the token before it's gone" constraint applies as
    # in mcp.py's disconnect endpoints -- this must run before db.delete,
    # while the row (and its access_token) still exists.
    from .auth import resolve_builtin_oauth_revocation

    revocation = resolve_builtin_oauth_revocation(
        db,
        provider=str(account.provider),
        access_token=str(account.access_token) if account.access_token else "",
        provider_user_id=(
            str(account.provider_user_id)
            if account.provider_user_id is not None
            else None
        ),
    )

    db.delete(account)
    db.commit()

    if revocation is not None:
        from .auth import revoke_builtin_oauth_grants

        await revoke_builtin_oauth_grants(
            db, [revocation], context=f"deleting connected account {account_id}"
        )

    return {"success": True, "message": "Account deleted successfully"}


@cloud_router.get("/google-drive/drives")
async def list_google_drives(
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List Google Drives (My Drive + Shared Drives)"""
    try:
        creds = get_google_credentials(cast(int, user.id), db, account_id)

        # Build Drive API service
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        drives_list = [{"id": "root", "name": "My Drive", "kind": "drive#drive"}]

        # List Shared Drives
        try:
            results = service.drives().list(pageSize=100).execute()
            shared_drives = results.get("drives", [])
            drives_list.extend(shared_drives)
        except Exception:
            # logger.warning(f"Failed to list shared drives: {e}")
            pass

        return drives_list

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing Google Drives: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@cloud_router.get("/google-drive/files")
async def list_google_drive_files(
    folder_id: str = "root",
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List files in Google Drive folder"""
    try:
        creds = get_google_credentials(cast(int, user.id), db, account_id)

        # Build Drive API service
        # cache_discovery=False to avoid file system issues and timeouts
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        # Query for files in folder, not trashed
        # fields reference: https://developers.google.com/drive/api/v3/reference/files/list
        query = f"'{folder_id}' in parents and trashed = false"

        # Include drives support for Shared Drives
        supports_all_drives = True
        include_items_from_all_drives = True

        results = (
            service.files()
            .list(
                q=query,
                pageSize=100,
                fields=(
                    "nextPageToken, files(id, name, mimeType, size, modifiedTime, resourceKey)"
                ),
                orderBy="folder,name",
                supportsAllDrives=supports_all_drives,
                includeItemsFromAllDrives=include_items_from_all_drives,
            )
            .execute()
        )

        files = results.get("files", [])

        # Map to frontend format
        cloud_files = []
        for file in files:
            mime_type = file.get("mimeType")
            is_folder = mime_type == "application/vnd.google-apps.folder"

            # Format size
            size_str = None
            if "size" in file:
                size_bytes = int(file["size"])
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                elif size_bytes < 1024 * 1024 * 1024:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

            # Format date (ISO 8601 to YYYY-MM-DD)
            updated_at = file.get("modifiedTime", "")
            if updated_at:
                try:
                    # Simple slice for YYYY-MM-DD, or use datetime parsing if needed
                    updated_at = updated_at.split("T")[0]
                except Exception:
                    pass

            cloud_files.append(
                {
                    "id": file.get("id"),
                    "name": file.get("name"),
                    "type": "folder" if is_folder else "file",
                    "size": size_str,
                    "updatedAt": updated_at,
                    "mimeType": mime_type,  # Optional, helpful for debugging
                    "resourceKey": file.get("resourceKey"),
                }
            )

        return cloud_files

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing Google Drive files: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
