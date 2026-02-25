"""Authentication API endpoints"""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, status
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
    to_encode.update({"exp": expire, "type": "access"})
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
