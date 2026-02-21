from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, field_validator


class ModelCreate(BaseModel):
    """Model creation request schema"""

    model_id: str
    category: str = "llm"
    model_provider: str
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    dimension: Optional[int] = None
    abilities: Optional[List[str]] = None
    description: Optional[str] = None
    share_with_users: bool = False  # Admin only: share this model with all users

    @field_validator("model_id", "model_name", "base_url", "api_key", mode="before")
    @classmethod
    def strip_string_fields(cls, v: Any) -> Any:
        """Strip whitespace from string fields"""
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("abilities")
    @classmethod
    def validate_abilities(
        cls, v: Optional[List[str]], values: Any
    ) -> Optional[List[str]]:
        """Validate abilities based on model category"""
        if v is None:
            return v

        category = values.data.get("category", "llm")

        if category == "image":
            # Image models can only have "generate" and "edit" abilities
            valid_image_abilities = {"generate", "edit"}
            invalid_abilities = set(v) - valid_image_abilities

            if invalid_abilities:
                raise ValueError(
                    f"Invalid abilities for image model: {invalid_abilities}. "
                    f"Valid abilities are: {valid_image_abilities}"
                )

            if not v:
                raise ValueError("Image model must have at least one ability")
        elif category == "embedding":
            # Embedding models can only have "embedding" ability
            valid_embedding_abilities = {"embedding"}
            invalid_abilities = set(v) - valid_embedding_abilities

            if invalid_abilities:
                raise ValueError(
                    f"Invalid abilities for embedding model: {invalid_abilities}. "
                    f"Valid abilities are: {valid_embedding_abilities}"
                )

            if not v:
                raise ValueError("Embedding model must have at least one ability")

        return v


class ModelUpdate(BaseModel):
    """Model update request schema"""

    category: Optional[str] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    dimension: Optional[int] = None
    description: Optional[str] = None
    abilities: Optional[List[str]] = None
    share_with_users: Optional[bool] = None  # Admin only: update sharing status

    @field_validator("model_name", "base_url", "api_key", mode="before")
    @classmethod
    def strip_string_fields(cls, v: Any) -> Any:
        """Strip whitespace from string fields"""
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("abilities")
    @classmethod
    def validate_abilities(
        cls, v: Optional[List[str]], values: Any
    ) -> Optional[List[str]]:
        """Validate abilities based on model category"""
        if v is None:
            return v

        category = values.data.get("category")

        # If category is being updated to "image" or is already "image"
        if category == "image":
            # Image models can only have "generate" and "edit" abilities
            valid_image_abilities = {"generate", "edit"}
            invalid_abilities = set(v) - valid_image_abilities

            if invalid_abilities:
                raise ValueError(
                    f"Invalid abilities for image model: {invalid_abilities}. "
                    f"Valid abilities are: {valid_image_abilities}"
                )

            if not v:
                raise ValueError("Image model must have at least one ability")
        elif category == "embedding":
            # Embedding models can only have "embedding" ability
            valid_embedding_abilities = {"embedding"}
            invalid_abilities = set(v) - valid_embedding_abilities

            if invalid_abilities:
                raise ValueError(
                    f"Invalid abilities for embedding model: {invalid_abilities}. "
                    f"Valid abilities are: {valid_embedding_abilities}"
                )

            if not v:
                raise ValueError("Embedding model must have at least one ability")

        return v


class ModelResponse(BaseModel):
    """Model response schema"""

    id: int
    model_id: str
    category: str
    model_provider: str
    model_name: str
    base_url: Optional[str]
    temperature: Optional[float]
    dimension: Optional[int]
    is_default: bool
    is_small_fast: bool
    is_visual: bool
    is_compact: bool
    abilities: Optional[List[str]]
    description: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    is_active: bool

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def format_datetime(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    class Config:
        from_attributes = True


class ModelTestRequest(BaseModel):
    """Model test request schema"""

    model_ids: Optional[list[str]] = None


class ModelTestResponse(BaseModel):
    """Model test response schema"""

    model_id: str
    status: str  # passed, failed
    response_time: Optional[float]
    message: Optional[str]
    error: Optional[str]


class UserDefaultModelCreate(BaseModel):
    """User default model configuration creation schema"""

    model_id: int
    config_type: str  # 'general', 'small_fast', 'visual', 'compact', 'embedding', 'image', 'image_edit'

    @field_validator("config_type")
    @classmethod
    def validate_config_type(cls, v: str) -> str:
        valid_types = {
            "general",
            "small_fast",
            "visual",
            "compact",
            "embedding",
            "image",
            "image_edit",
        }
        if v not in valid_types:
            raise ValueError(
                f"Invalid config_type: {v}. Valid types are: {valid_types}"
            )
        return v


class UserDefaultModelResponse(BaseModel):
    """User default model configuration response schema"""

    id: int
    user_id: int
    model_id: int
    config_type: str
    created_at: Optional[str]
    updated_at: Optional[str]

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def format_datetime(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    class Config:
        from_attributes = True


class UserModelResponse(BaseModel):
    """User model relationship response schema"""

    id: int
    user_id: int
    model_id: int
    is_owner: bool
    can_edit: bool
    can_delete: bool
    is_shared: bool
    created_at: Optional[str]
    updated_at: Optional[str]

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def format_datetime(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    class Config:
        from_attributes = True


class ModelWithAccessInfo(BaseModel):
    """Model response with user access information"""

    id: int
    model_id: str
    category: str
    model_provider: str
    model_name: str
    base_url: Optional[str]
    temperature: Optional[float]
    dimension: Optional[int]
    abilities: Optional[List[str]]
    description: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    is_active: bool

    # User access information
    is_owner: bool
    can_edit: bool
    can_delete: bool
    is_shared: bool

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def format_datetime(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    class Config:
        from_attributes = True


class FetchProviderModelsRequest(BaseModel):
    """Request schema for fetching models from a provider."""

    provider: str
    api_key: str
    base_url: Optional[str] = None


class FetchMultipleProvidersRequest(BaseModel):
    """Request schema for fetching models from multiple providers."""

    providers: Optional[List[str]] = None
    """List of providers to fetch from. If None, fetches from all configured providers."""


class ProviderModelInfo(BaseModel):
    """Information about a model from a provider."""

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    created: Optional[int] = None
    owned_by: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None


class FetchProviderModelsResponse(BaseModel):
    """Response schema for fetching models from a provider."""

    provider: str
    models: List[ProviderModelInfo]
    count: int


class ProviderInfo(BaseModel):
    """Information about a model provider."""

    id: str
    name: str
    description: str
    requires_base_url: bool


class SupportedProvidersResponse(BaseModel):
    """Response schema for supported providers."""

    providers: List[ProviderInfo]
