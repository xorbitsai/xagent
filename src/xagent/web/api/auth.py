"""Authentication API endpoints"""

import asyncio
import hashlib
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, cast

import requests

# Relax token scope verification as Google might add extra scopes (like openid)
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from ...config import get_app_base_url, get_password_reset_expire_minutes
from ..auth_config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    PASSWORD_MIN_LENGTH,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.system_setting import SystemSetting
from ..models.user import User
from ..models.user_oauth import UserOAuth
from ..services.auth_email import send_password_reset_email

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])

REGISTRATION_ENABLED_SETTING_KEY = "registration_enabled"
SETUP_COMPLETED_SETTING_KEY = "setup_completed"
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _best_effort_ensure_gmail_watches_for_user(db: Session, *, user_id: int) -> None:
    from ..services.gmail_provisioning import (
        best_effort_provision_gmail_watches_for_user,
    )

    best_effort_provision_gmail_watches_for_user(
        db,
        user_id=user_id,
        context="after OAuth callback",
    )


def _oauth_env_name(provider: str, suffix: str) -> str:
    return f"{provider.upper()}_{suffix}"


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
    import html

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
    if provider.lower() == "meta":
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


class RefreshTokenRequest(BaseModel):
    """Refresh token request model"""

    refresh_token: str


class RefreshTokenResponse(BaseModel):
    """Refresh token response model"""

    success: bool
    message: str
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    refresh_expires_in: Optional[int] = None


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


def serialize_auth_user(user: User, include_login_time: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "is_admin": bool(cast(Any, user.is_admin)),
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during login: {str(e)}",
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during registration: {str(e)}",
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

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during password update: {str(e)}",
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

    state_payload = {
        "type": "oauth_state",
        "user_id": user_id,
        "provider": provider,
        "app_id": app_id,
        "redirect": redirect,
    }
    state = create_access_token(data=state_payload, expires_delta=timedelta(minutes=10))

    app_scopes: list[str] | None = None
    from ..mcp_apps import get_app_by_id

    if app_id:
        app_info = get_app_by_id(db, app_id)
        if app_info and "oauth_scopes" in app_info:
            app_scopes = app_info["oauth_scopes"]

    scopes = _merge_oauth_scopes(db_provider.default_scopes or [], app_scopes)
    scope_str = _oauth_scope_separator(provider).join(scopes)

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
    meta_config_id = _meta_login_config_id() if provider.lower() == "meta" else ""
    if meta_config_id:
        params["config_id"] = meta_config_id
    elif scope_str:
        params["scope"] = scope_str

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
    db: Session, user_id: str, app_info: Dict[str, Any]
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
        import html

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

    user_id = payload.get("user_id")
    app_id = payload.get("app_id")

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
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        token_response = requests.post(
            token_url, data=data, headers=headers, timeout=10.0
        )
        token_data = token_response.json()

        if "error" in token_data:
            import html

            return HTMLResponse(
                content=f"<h1>Error exchanging token</h1><p>{html.escape(str(token_data))}</p>",
                status_code=400,
            )

        token_data = _exchange_meta_long_lived_token(
            provider, token_url, token_data, client_id, client_secret
        )
        access_token = token_data.get("access_token")

        provider_user_id = None
        email = None

        if userinfo_url and access_token:
            info_headers = {"Authorization": f"Bearer {access_token}"}
            # Replace {{access_token}} placeholder if present
            actual_url = userinfo_url.replace("{{access_token}}", access_token)
            info_response = requests.get(actual_url, headers=info_headers, timeout=10.0)
            if info_response.status_code == 200:
                info_data = info_response.json()
                provider_user_id = info_data.get(db_provider.user_id_path or "id")
                email = info_data.get(db_provider.email_path or "email")

        if user_id:
            db.query(UserOAuth).filter(
                UserOAuth.user_id == user_id, UserOAuth.provider == (app_id or provider)
            ).delete()

            oauth_account = UserOAuth(
                user_id=user_id,
                provider=(app_id or provider),
                provider_user_id=str(provider_user_id) if provider_user_id else None,
            )
            db.add(oauth_account)

            setattr(oauth_account, "access_token", access_token)
            setattr(oauth_account, "token_type", token_data.get("token_type", "Bearer"))
            setattr(oauth_account, "scope", token_data.get("scope", ""))
            setattr(oauth_account, "email", email)
            if "refresh_token" in token_data:
                setattr(oauth_account, "refresh_token", token_data.get("refresh_token"))
            if "expires_in" in token_data:
                setattr(
                    oauth_account,
                    "expires_at",
                    datetime.now(timezone.utc)
                    + timedelta(seconds=int(token_data["expires_in"])),
                )

            from ..mcp_apps import get_all_mcp_apps, get_app_by_id

            if app_id:
                app_info = get_app_by_id(db, app_id)
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
                apps = [
                    app
                    for app in get_all_mcp_apps(db)
                    if app.get("provider") == provider
                ]
                for app_info in apps:
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
            if (app_id or provider) == "gmail":
                _best_effort_ensure_gmail_watches_for_user(db, user_id=int(user_id))

        import json
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
    except Exception as e:
        import html

        logger.exception("Generic OAuth callback failed")
        return HTMLResponse(
            content=f"<h1>Authentication Failed</h1><p>{html.escape(str(e))}</p>",
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
