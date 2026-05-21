from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class BackgroundJobResponse(BaseModel):
    id: str
    user_id: int
    job_type: str
    queue: str
    status: str
    progress: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None
    celery_task_id: str | None = None
    attempts: int
    max_attempts: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
