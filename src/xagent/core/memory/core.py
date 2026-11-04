from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class MemoryNote(BaseModel):
    content: Union[str, bytes]
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    category: str = "general"
    timestamp: datetime = Field(default_factory=datetime.now)
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    success: bool
    memory_id: Optional[str] = None
    content: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = Field(default_factory=lambda: {})
    search_results: Optional[list[Any]] = None
