from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_serializer


class UserChannelBase(BaseModel):
    channel_type: str = Field(..., description="e.g. telegram, feishu, slack")
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

    @field_serializer("config")
    def serialize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Never return channel credentials to the browser.

        Applies to every channel type and installation mode: clients that
        edit a channel resubmit only the fields they changed, and the update
        endpoint keeps the stored secret when the submitted value is empty,
        so the UI never needs to read a secret back.
        """
        public_config = dict(config)
        for field in ("bot_token", "app_secret", "app_token"):
            if public_config.pop(field, None):
                public_config[f"{field}_configured"] = True
        return public_config
