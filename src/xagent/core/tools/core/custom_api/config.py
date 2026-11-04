"""Custom API configuration models."""

import re
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator


class CustomApiConfig(BaseModel):
    """Configuration for a Custom API tool."""

    name: str = Field(..., description="Name of the API tool")
    description: str = Field(..., description="Description of the API tool")
    url: str = Field(..., description="Base URL or full URL template")
    method: str = Field(default="GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Default headers"
    )
    env: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Environment variables/secrets"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate API name format."""
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Name can only contain letters, numbers, hyphens and underscores"
            )

        return v.strip()
