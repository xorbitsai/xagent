from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class UserChannelBase(BaseModel):
    channel_type: str = Field(..., description="e.g. telegram, feishu")
    channel_name: Optional[str] = Field(None, description="User-friendly name")
    config: Dict[str, Any] = Field(..., description="Channel specific configuration")
    is_active: bool = True


class UserChannelCreate(UserChannelBase):
    pass


class UserChannelUpdate(BaseModel):
    channel_name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class UserChannelResponse(UserChannelBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
