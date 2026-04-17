"""
Custom API Management API Endpoints

Provides REST API endpoints for managing Custom API configurations
in the web application.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.utils.encryption import encrypt_value
from ..auth_dependencies import get_current_user
from ..models.custom_api import CustomApi, UserCustomApi
from ..models.database import get_db
from ..models.user import User

logger = logging.getLogger(__name__)


# Pydantic models for API
class CustomApiCreate(BaseModel):
    """Request model for creating a Custom API."""

    name: str = Field(..., min_length=1, max_length=100, description="API name")
    description: Optional[str] = Field(None, description="API description")
    url: Optional[str] = Field(
        None, min_length=1, max_length=500, description="API URL"
    )
    method: Optional[str] = Field("GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables (secrets)"
    )
    is_active: bool = Field(True, description="Whether the API is active")


class CustomApiUpdate(BaseModel):
    """Request model for updating a Custom API."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="API name"
    )
    description: Optional[str] = Field(None, description="API description")
    url: Optional[str] = Field(
        None, min_length=1, max_length=500, description="API URL"
    )
    method: Optional[str] = Field(None, description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables (secrets)"
    )
    is_active: Optional[bool] = Field(None, description="Whether the API is active")


class CustomApiResponse(BaseModel):
    """Response model for Custom API."""

    id: int
    user_id: int
    name: str
    description: Optional[str]
    url: Optional[str]
    method: Optional[str]
    headers: Optional[Dict[str, str]]
    env: Optional[Dict[str, str]]  # Will return masked values
    is_active: bool
    is_default: bool
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


# Create router
custom_api_router = APIRouter(prefix="/api/custom-apis", tags=["Custom API Management"])


def _db_api_to_response(
    api: CustomApi,
    user_api: UserCustomApi,
) -> CustomApiResponse:
    """Convert database CustomApi to response model with masked env values."""

    # Mask env values for frontend
    masked_env = None
    if api.env and isinstance(api.env, dict):
        masked_env = {k: "********" for k in api.env.keys()}

    return CustomApiResponse(
        id=api.id,
        user_id=user_api.user_id,
        name=api.name,
        description=api.description,
        url=api.url,
        method=api.method,
        headers=api.headers,
        env=masked_env,
        is_active=user_api.is_active,
        is_default=user_api.is_default,
        created_at=str(api.created_at.isoformat()),
        updated_at=str(api.updated_at.isoformat()),
    )


def _process_env_vars(
    env: Optional[Dict[str, str]], existing_env: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, str]]:
    """Encrypt environment variables, keeping existing ones if masked."""
    if not env:
        return env

    encrypted_env = {}
    existing_env = existing_env or {}

    for k, v in env.items():
        if v == "********":
            # Retain existing encrypted value if masked
            if k in existing_env:
                encrypted_env[k] = existing_env[k]
            else:
                logger.warning(f"Masked key {k} not found in existing env, skipping")
        else:
            encrypted_env[k] = encrypt_value(v)

    return encrypted_env


@custom_api_router.get("", response_model=List[CustomApiResponse])
async def list_custom_apis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[CustomApiResponse]:
    """List all Custom APIs for the current user."""
    user_apis = (
        db.query(UserCustomApi).filter(UserCustomApi.user_id == current_user.id).all()
    )

    responses = []
    for user_api in user_apis:
        if user_api.custom_api:
            responses.append(_db_api_to_response(user_api.custom_api, user_api))

    return responses


@custom_api_router.post(
    "", response_model=CustomApiResponse, status_code=status.HTTP_201_CREATED
)
async def create_custom_api(
    api_data: CustomApiCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Create a new Custom API."""

    # Check if name already exists
    existing = db.query(CustomApi).filter(CustomApi.name == api_data.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Custom API with name '{api_data.name}' already exists",
        )

    # Process env variables
    encrypted_env = _process_env_vars(api_data.env)

    # Create CustomApi
    new_api = CustomApi(
        name=api_data.name,
        description=api_data.description,
        url=api_data.url,
        method=api_data.method,
        headers=api_data.headers,
        env=encrypted_env,
    )

    db.add(new_api)
    db.flush()

    # Create UserCustomApi link
    user_api = UserCustomApi(
        user_id=current_user.id,
        custom_api_id=new_api.id,
        is_owner=True,
        can_edit=True,
        can_delete=True,
        is_active=api_data.is_active,
    )

    db.add(user_api)
    db.commit()
    db.refresh(new_api)
    db.refresh(user_api)

    return _db_api_to_response(new_api, user_api)


@custom_api_router.get("/{api_id}", response_model=CustomApiResponse)
async def get_custom_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Get a specific Custom API by ID."""

    user_api = (
        db.query(UserCustomApi)
        .filter(
            UserCustomApi.custom_api_id == api_id,
            UserCustomApi.user_id == current_user.id,
        )
        .first()
    )

    if not user_api or not user_api.custom_api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )

    return _db_api_to_response(user_api.custom_api, user_api)


@custom_api_router.put("/{api_id}", response_model=CustomApiResponse)
async def update_custom_api(
    api_id: int,
    api_data: CustomApiUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Update an existing Custom API."""

    user_api = (
        db.query(UserCustomApi)
        .filter(
            UserCustomApi.custom_api_id == api_id,
            UserCustomApi.user_id == current_user.id,
        )
        .first()
    )

    if not user_api or not user_api.custom_api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )

    if not user_api.can_edit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to edit this Custom API",
        )

    api = user_api.custom_api

    # Check name uniqueness if name is changed
    if api_data.name and api_data.name != api.name:
        existing = db.query(CustomApi).filter(CustomApi.name == api_data.name).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Custom API with name '{api_data.name}' already exists",
            )
        api.name = api_data.name

    # Update fields
    if api_data.description is not None:
        api.description = api_data.description
    if api_data.url is not None:
        api.url = api_data.url
    if api_data.method is not None:
        api.method = api_data.method
    if api_data.headers is not None:
        api.headers = api_data.headers

    # Process env variables
    if api_data.env is not None:
        existing_env = api.env if isinstance(api.env, dict) else {}
        api.env = _process_env_vars(api_data.env, existing_env)

    # Update UserCustomApi link
    if api_data.is_active is not None:
        user_api.is_active = api_data.is_active  # type: ignore[assignment]

    db.commit()
    db.refresh(api)

    return _db_api_to_response(api, user_api)


@custom_api_router.delete("/{api_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_custom_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete a Custom API."""

    user_api = (
        db.query(UserCustomApi)
        .filter(
            UserCustomApi.custom_api_id == api_id,
            UserCustomApi.user_id == current_user.id,
        )
        .first()
    )

    if not user_api or not user_api.custom_api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )

    if not user_api.can_delete:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this Custom API",
        )

    db.delete(user_api.custom_api)  # Will cascade to UserCustomApi
    db.commit()

    return None
