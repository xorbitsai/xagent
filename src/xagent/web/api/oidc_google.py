from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

from authlib.integrations.base_client import OAuthError  # type: ignore[import-untyped]
from authlib.integrations.starlette_client import OAuth  # type: ignore[import-untyped]
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from itsdangerous import (  # type: ignore[import-not-found]
    BadSignature,
    SignatureExpired,
    URLSafeTimedSerializer,
)
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...config import (
    get_frontend_url,
    get_google_oidc_client_id,
    get_google_oidc_client_secret,
    get_google_oidc_redirect_uri,
    get_oidc_exchange_ttl_seconds,
    get_oidc_login_ttl_seconds,
    get_session_secret,
)
from ..auth_config import ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
from ..models.database import get_db
from ..models.oidc_consumed_token import OidcConsumedToken
from ..models.user import User
from ..models.user_identity import UserIdentity
from .auth import (
    create_access_token,
    create_refresh_token,
    has_users,
    is_registration_enabled,
)

logger = logging.getLogger(__name__)

GOOGLE_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"
GOOGLE_PROVIDER = "google"
OIDC_EXCHANGE_SALT = "oidc-google-exchange"
OIDC_UNUSABLE_PASSWORD_HASH = "!oidc-google"

router = APIRouter(prefix="/oidc/google", tags=["Authentication"])


@dataclass(frozen=True)
class OidcExchangeTransaction:
    user_id: int
    token_id: str


class OidcExchangeRequest(BaseModel):
    code: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _exchange_serializer() -> Any:
    return URLSafeTimedSerializer(get_session_secret(), salt=OIDC_EXCHANGE_SALT)


def _is_google_oidc_configured() -> bool:
    return bool(
        get_google_oidc_client_id()
        and get_google_oidc_client_secret()
        and get_google_oidc_redirect_uri()
    )


def _frontend_url(path: str, params: dict[str, str] | None = None) -> str:
    url = f"{get_frontend_url()}{path}"
    if params:
        return f"{url}?{urlencode(params)}"
    return url


def _login_error_redirect(error_code: str) -> RedirectResponse:
    return RedirectResponse(_frontend_url("/login", {"oidc_error": error_code}))


def _google_oauth_client() -> Any:
    client_id = get_google_oidc_client_id()
    client_secret = get_google_oidc_client_secret()
    if not client_id or not client_secret:
        raise RuntimeError("Google OIDC is not configured")

    oauth = OAuth()
    client = oauth.register(
        name=GOOGLE_PROVIDER,
        client_id=client_id,
        client_secret=client_secret,
        server_metadata_url=GOOGLE_METADATA_URL,
        client_kwargs={
            "scope": "openid email profile",
            "code_challenge_method": "S256",
        },
    )
    client.framework.expires_in = get_oidc_login_ttl_seconds()
    return client


async def start_google_oidc_authorization(request: Request) -> RedirectResponse:
    redirect_uri = get_google_oidc_redirect_uri()
    if not redirect_uri:
        raise RuntimeError("Google OIDC is not configured")
    response = await _google_oauth_client().authorize_redirect(
        request,
        redirect_uri,
        prompt="select_account",
    )
    return RedirectResponse(
        response.headers["location"],
        status_code=response.status_code,
        headers=dict(response.headers),
    )


async def complete_google_oidc_authorization(
    request: Request, *, redirect_uri: str
) -> dict[str, Any]:
    token = await _google_oauth_client().authorize_access_token(request)
    userinfo = token.get("userinfo")
    if userinfo is None:
        raise ValueError("invalid_id_token")
    return dict(userinfo)


def _create_exchange_code(user_id: int) -> str:
    return str(
        _exchange_serializer().dumps(
            {"user_id": user_id, "jti": secrets.token_urlsafe(32)}
        )
    )


def _decode_exchange_code(code: str) -> OidcExchangeTransaction | None:
    try:
        data = _exchange_serializer().loads(
            code,
            max_age=get_oidc_exchange_ttl_seconds(),
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(data, dict):
        return None
    user_id = data.get("user_id")
    if not isinstance(user_id, int):
        return None
    token_id = data.get("jti")
    if not isinstance(token_id, str) or not token_id:
        return None
    return OidcExchangeTransaction(user_id=user_id, token_id=token_id)


def _consume_exchange_code(db: Session, code: str) -> OidcExchangeTransaction | None:
    transaction = _decode_exchange_code(code)
    if transaction is None:
        return None

    now = _utcnow()
    db.query(OidcConsumedToken).filter(OidcConsumedToken.expires_at <= now).delete(
        synchronize_session=False
    )
    db.add(
        OidcConsumedToken(
            token_id=transaction.token_id,
            expires_at=now + timedelta(seconds=get_oidc_exchange_ttl_seconds()),
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return None
    return transaction


def _unique_google_username(db: Session, email: str, provider_subject: str) -> str:
    existing = db.query(User.id).filter(User.username == email).first()
    if existing is None:
        return email

    local, separator, domain = email.partition("@")
    suffix = hashlib.sha256(provider_subject.encode()).hexdigest()[:8]
    if separator:
        candidate = f"{local}+google-{suffix}@{domain}"
    else:
        candidate = f"{email}-google-{suffix}"

    counter = 1
    unique_candidate = candidate
    while db.query(User.id).filter(User.username == unique_candidate).first():
        counter += 1
        unique_candidate = f"{candidate}-{counter}"
    return unique_candidate


def _get_or_create_google_user(db: Session, claims: dict[str, Any]) -> User:
    provider_subject = str(claims.get("sub") or "")
    email = str(claims.get("email") or "")
    email_verified = bool(claims.get("email_verified"))
    if not provider_subject:
        raise ValueError("missing_subject")
    if not email_verified:
        raise ValueError("email_unverified")
    if not email:
        raise ValueError("missing_email")

    identity = (
        db.query(UserIdentity)
        .filter(
            UserIdentity.provider == GOOGLE_PROVIDER,
            UserIdentity.provider_subject == provider_subject,
        )
        .first()
    )
    if identity is not None:
        user = db.query(User).filter(User.id == identity.user_id).first()
        if user is None:
            raise ValueError("linked_user_not_found")
        if not user.email:
            setattr(user, "email", email)
        setattr(identity, "email", email)
        setattr(identity, "email_verified", email_verified)
        setattr(identity, "display_name", claims.get("name"))
        setattr(identity, "picture_url", claims.get("picture"))
        db.commit()
        return user

    if not has_users(db):
        raise ValueError("setup_required")
    if not is_registration_enabled(db):
        raise ValueError("registration_disabled")

    user = User(
        username=_unique_google_username(db, email, provider_subject),
        email=email,
        password_hash=OIDC_UNUSABLE_PASSWORD_HASH,
        is_admin=False,
    )
    db.add(user)
    db.flush()
    db.add(
        UserIdentity(
            user_id=user.id,
            provider=GOOGLE_PROVIDER,
            provider_subject=provider_subject,
            email=email,
            email_verified=email_verified,
            display_name=claims.get("name"),
            picture_url=claims.get("picture"),
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        identity = (
            db.query(UserIdentity)
            .filter(
                UserIdentity.provider == GOOGLE_PROVIDER,
                UserIdentity.provider_subject == provider_subject,
            )
            .first()
        )
        if identity is None:
            raise
        user = db.query(User).filter(User.id == identity.user_id).one()
    db.refresh(user)
    return user


def _issue_auth_payload(db: Session, user: User) -> dict[str, Any]:
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(
        data={"sub": user.username, "user_id": user.id}
    )
    setattr(user, "refresh_token", refresh_token)
    setattr(
        user,
        "refresh_token_expires_at",
        _utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )
    db.commit()

    return {
        "success": True,
        "message": "Login successful",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_admin": user.is_admin,
            "has_password": not str(user.password_hash).startswith("!"),
            "loginTime": _utcnow().timestamp(),
        },
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "refresh_expires_in": REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        "user_id": user.id,
    }


@router.get("/status")
def google_oidc_status() -> dict[str, Any]:
    return {"configured": _is_google_oidc_configured(), "provider": GOOGLE_PROVIDER}


@router.get("/login")
async def google_oidc_login(request: Request) -> RedirectResponse:
    if not _is_google_oidc_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OIDC is not configured",
        )
    try:
        return await start_google_oidc_authorization(request)
    except Exception:
        logger.exception("Google OIDC login start failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OIDC is not available",
        ) from None


@router.get("/callback")
async def google_oidc_callback(
    request: Request,
    db: Session = Depends(get_db),
) -> RedirectResponse:
    error = request.query_params.get("error")
    if error:
        return _login_error_redirect("provider_error")

    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or not state:
        return _login_error_redirect("invalid_callback")

    redirect_uri = get_google_oidc_redirect_uri()
    if not redirect_uri:
        return _login_error_redirect("not_configured")

    try:
        claims = await complete_google_oidc_authorization(
            request,
            redirect_uri=redirect_uri,
        )
        user = _get_or_create_google_user(db, claims)
        exchange_code = _create_exchange_code(int(user.id))
    except OAuthError as exc:
        logger.info("Google OIDC authorization failed: %s", exc.error)
        return _login_error_redirect("authorization_failed")
    except ValueError as exc:
        error_code = str(exc) or "verification_failed"
        logger.info("Google OIDC login rejected: %s", error_code)
        return _login_error_redirect(error_code)
    except Exception:
        logger.exception("Google OIDC callback failed")
        return _login_error_redirect("callback_failed")

    return RedirectResponse(
        _frontend_url(
            "/auth/oidc/callback",
            {"provider": GOOGLE_PROVIDER, "code": exchange_code},
        )
    )


@router.post("/exchange")
def google_oidc_exchange(
    payload: OidcExchangeRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    transaction = _consume_exchange_code(db, payload.code)
    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OIDC exchange code",
        )

    user = db.query(User).filter(User.id == transaction.user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OIDC exchange code",
        )
    return _issue_auth_payload(db, user)
