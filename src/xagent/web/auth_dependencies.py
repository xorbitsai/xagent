"""Authentication dependency for user segregation"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from .auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from .models.database import get_db
from .models.user import User

logger = logging.getLogger(__name__)

# JWT Bearer token authentication
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Get current authenticated user from JWT token

    Args:
        credentials: Bearer token credentials
        db: Database session

    Returns:
        User: Current authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials

        # Validate JWT token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        token_type = payload.get("type")
        if token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        username = payload.get("sub")
        user_id = payload.get("user_id")

        if username is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Type narrowing after null check

        # Get user from database
        user = (
            db.query(User).filter(User.username == username, User.id == user_id).first()
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user

    except JWTError as e:
        logger.error(f"JWT validation error: {e}")
        # Check if it's an expired token specifically
        error_message = str(e)
        if (
            "expired" in error_message.lower()
            or "signature has expired" in error_message.lower()
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer", "Error-Type": "TokenExpired"},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer", "Error-Type": "InvalidToken"},
            )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Get current authenticated user if token is provided

    Args:
        credentials: Optional Bearer token credentials
        db: Database session

    Returns:
        Optional[User]: Current authenticated user or None
    """
    if not credentials:
        return None

    try:
        return get_current_user(credentials, db)
    except HTTPException:
        return None


def get_user_from_token(token: str, db: Session) -> Optional[User]:
    """
    Get user from authentication token

    Args:
        token: Authentication token
        db: Database session

    Returns:
        Optional[User]: User if token is valid, None otherwise
    """
    try:
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]

        # Validate JWT token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            return None
        username = payload.get("sub")
        user_id = payload.get("user_id")

        if username is None or user_id is None:
            return None

        # Type narrowing after null check

        return (
            db.query(User).filter(User.username == username, User.id == user_id).first()
        )

    except JWTError as e:
        logger.error(f"JWT validation error: {e}")
        return None
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return None


def require_user(user: User = Depends(get_current_user)) -> User:
    """
    Require authenticated user (alias for get_current_user)

    Args:
        user: Current authenticated user

    Returns:
        User: Current authenticated user
    """
    return user


def get_user_from_websocket_token(token: str, db: Session) -> Optional[User]:
    """
    Get user from WebSocket authentication token

    Args:
        token: Authentication token from WebSocket
        db: Database session

    Returns:
        Optional[User]: User if token is valid, None otherwise
    """
    return get_user_from_token(token, db)
