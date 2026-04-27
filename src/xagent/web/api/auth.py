"""Authentication API endpoints"""

import asyncio
import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, cast

import requests

# Relax token scope verification as Google might add extra scopes (like openid)
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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
from ..models.user import User, UserDefaultModel, UserModel
from ..models.user_oauth import UserOAuth

auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])

REGISTRATION_ENABLED_SETTING_KEY = "registration_enabled"
SETUP_COMPLETED_SETTING_KEY = "setup_completed"


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


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == password_hash


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

        user = User(
            username=request.username,
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


def create_user(
    db: Session, username: str, password: str, inherit_defaults: bool = False
) -> User:
    """Create a new user without default model configurations

    Users will use admin's defaults via fallback logic until they set their own.
    """
    password_hash = hash_password(password)
    user = User(username=username, password_hash=password_hash)
    db.add(user)
    db.flush()  # Get the user ID without committing
    db.refresh(user)

    # Always grant access to shared models first
    _grant_shared_model_access(db, user)

    # Inherit default model configurations from admin if requested
    # Default is now False - users should use fallback logic
    if inherit_defaults:
        _inherit_admin_defaults(db, user)

    # Commit everything together
    db.commit()
    return user


def _grant_shared_model_access(db: Session, new_user: User) -> None:
    """Grant access to admin's shared models (but not default configurations)"""
    try:
        # Get admin user
        admin_user = db.query(User).filter(User.is_admin).first()
        if not admin_user:
            return

        # Grant access to all shared models
        shared_models = (
            db.query(UserModel)
            .filter(UserModel.user_id == admin_user.id, UserModel.is_shared)
            .all()
        )

        for shared_model in shared_models:
            # Check if user already has access to this model (any configuration)
            existing_access = (
                db.query(UserModel)
                .filter(
                    UserModel.user_id == new_user.id,
                    UserModel.model_id == shared_model.model_id,
                )
                .first()
            )

            # Only create new access if user doesn't already have this model
            if not existing_access:
                # Grant read-only access to shared model
                user_access = UserModel(
                    user_id=new_user.id,
                    model_id=shared_model.model_id,
                    is_owner=False,
                    can_edit=False,
                    can_delete=False,
                    is_shared=True,
                )
                db.add(user_access)

    except Exception as e:
        # Log error but don't fail user creation
        import logging

        logger = logging.getLogger(__name__)
        logger.error(
            f"Error granting shared model access for user {new_user.username}: {e}"
        )
        # Don't rollback here, let the main transaction handle it


def _inherit_admin_defaults(db: Session, new_user: User) -> None:
    """Inherit admin's default model configurations (legacy function for backward compatibility)"""
    try:
        # Get admin user
        admin_user = db.query(User).filter(User.is_admin).first()
        if not admin_user:
            return

        # _grant_shared_model_access is called first in create_user to ensure
        # user has access to models before creating default configurations

        # Then, inherit admin's default model configurations
        admin_defaults = (
            db.query(UserDefaultModel)
            .filter(UserDefaultModel.user_id == admin_user.id)
            .all()
        )

        for admin_default in admin_defaults:
            # Check if new user has access to the model
            user_model = (
                db.query(UserModel)
                .filter(
                    UserModel.user_id == new_user.id,
                    UserModel.model_id == admin_default.model_id,
                )
                .first()
            )

            # Only create default config if user has access to the model
            if user_model:
                new_default = UserDefaultModel(
                    user_id=new_user.id,
                    model_id=admin_default.model_id,
                    config_type=admin_default.config_type,
                )
                db.add(new_default)

    except Exception as e:
        # Log error but don't fail user creation
        import logging

        logger = logging.getLogger(__name__)
        logger.error(
            f"Error inheriting admin defaults for user {new_user.username}: {e}"
        )
        # Don't rollback here, let the main transaction handle it


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()


@auth_router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """User login endpoint"""
    try:
        # Run synchronous database queries in thread pool to avoid blocking event loop
        def _get_user_sync() -> User:
            # Get user from database
            user = get_user_by_username(db, request.username)
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
            "user": {
                "id": user.id,
                "username": user.username,
                "is_admin": user.is_admin,
                "loginTime": datetime.now(timezone.utc).timestamp(),
            },
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

        initialized = has_users(db)
        if not initialized:
            return RegisterResponse(
                success=False,
                message="System is not initialized. Please create the first admin account.",
            )

        if not is_registration_enabled(db):
            return RegisterResponse(success=False, message="Registration is disabled")

        # Create new user with inherited defaults
        user = create_user(
            db, request.username, request.password, inherit_defaults=True
        )

        return RegisterResponse(
            success=True,
            message="Registration successful",
            user={
                "id": user.id,
                "username": user.username,
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
            return RefreshTokenResponse(
                success=False,
                message="Invalid refresh token",
            )

        # Get user from database
        user_id = payload.get("user_id")
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return RefreshTokenResponse(
                success=False,
                message="User does not exist",
            )

        # Check if refresh token matches and is not expired
        user_refresh_token = getattr(user, "refresh_token", None)
        refresh_token_expires_at = getattr(user, "refresh_token_expires_at", None)
        if (
            user_refresh_token != request.refresh_token
            or refresh_token_expires_at is None
        ):
            return RefreshTokenResponse(
                success=False,
                message="Invalid refresh token",
            )

        # Check expiration - handle timezone-aware and naive datetimes
        now = datetime.now(timezone.utc)
        if (
            hasattr(refresh_token_expires_at, "tzinfo")
            and getattr(refresh_token_expires_at, "tzinfo", None) is not None
        ):
            # Timezone-aware datetime
            if cast(Any, refresh_token_expires_at) < now:
                return RefreshTokenResponse(
                    success=False,
                    message="Refresh token has expired",
                )
        else:
            # Naive datetime - assume UTC
            if cast(Any, refresh_token_expires_at) < now.replace(tzinfo=None):
                return RefreshTokenResponse(
                    success=False,
                    message="Refresh token has expired",
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

    except Exception as e:
        return RefreshTokenResponse(
            success=False,
            message=f"Token refresh failed: {str(e)}",
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
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "is_admin": current_user.is_admin,
        },
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

    from ...core.utils.encryption import decrypt_value

    client_id = decrypt_value(db_provider.client_id)
    auth_url = db_provider.auth_url

    redirect_uri = None
    if getattr(db_provider, "redirect_uri", None):
        redirect_uri = db_provider.redirect_uri
    if not redirect_uri:
        redirect_uri = os.environ.get(
            f"{provider.upper()}_REDIRECT_URI",
            f"http://localhost:8000/api/auth/{provider}/callback",
        )

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

    scopes = db_provider.default_scopes or []
    from ..mcp_apps import get_app_by_id

    if app_id:
        app_info = get_app_by_id(db, app_id)
        if app_info and "oauth_scopes" in app_info:
            scopes = list(set(scopes + app_info["oauth_scopes"]))

    scope_str = " ".join(scopes)

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
    if scope_str:
        params["scope"] = scope_str

    full_auth_url = f"{auth_url}?{urlencode(params)}"
    return RedirectResponse(full_auth_url)


def _ensure_user_mcp_server(
    db: Session, user_id: str, app_info: Dict[str, Any]
) -> None:
    """Ensure MCPServer and UserMCPServer records exist for an OAuth app."""
    from sqlalchemy.exc import IntegrityError

    from ..models.mcp import MCPServer, UserMCPServer

    mcp_server = db.query(MCPServer).filter(MCPServer.name == app_info["name"]).first()
    if not mcp_server:
        mcp_server = MCPServer(
            name=app_info["name"],
            description=app_info["description"],
            managed="external",
            transport="oauth",
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

    from ...core.utils.encryption import decrypt_value

    client_id = decrypt_value(db_provider.client_id)
    client_secret = decrypt_value(db_provider.client_secret)
    token_url = db_provider.token_url
    userinfo_url = db_provider.userinfo_url

    redirect_uri = None
    if getattr(db_provider, "redirect_uri", None):
        redirect_uri = db_provider.redirect_uri
    if not redirect_uri:
        redirect_uri = os.environ.get(
            f"{provider.upper()}_REDIRECT_URI",
            f"http://localhost:8000/api/auth/{provider}/callback",
        )

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

            oauth_account.access_token = access_token
            setattr(oauth_account, "token_type", token_data.get("token_type", "Bearer"))
            setattr(oauth_account, "scope", token_data.get("scope", ""))
            setattr(oauth_account, "email", email)
            if "refresh_token" in token_data:
                oauth_account.refresh_token = token_data.get("refresh_token")
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
                    _ensure_user_mcp_server(db, user_id, app_info)
            else:
                apps = [
                    app
                    for app in get_all_mcp_apps(db)
                    if app.get("provider") == provider
                ]
                for app_info in apps:
                    _ensure_user_mcp_server(db, user_id, app_info)

            db.commit()

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
        import logging

        logger = logging.getLogger(__name__)
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
