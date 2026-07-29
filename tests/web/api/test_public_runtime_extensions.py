from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from xagent.web.api.public_chat_access import (
    PublicChatAccessContext,
    ShareChatAccessContext,
    create_public_chat_task,
    create_share_chat_task,
)
from xagent.web.schemas.chat import TaskCreateRequest


@pytest.mark.asyncio
async def test_public_widget_rejects_task_runtime_extensions() -> None:
    request = TaskCreateRequest(
        title="public task",
        runtime_extensions={"browser_runtime": {}},
    )
    context = PublicChatAccessContext(
        user=SimpleNamespace(id=7),  # type: ignore[arg-type]
        channel_id=None,
        guest_id="guest",
        widget_agent_id=1,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_public_chat_task(
            request=request,
            access_context=context,
            db=object(),  # type: ignore[arg-type]
            default_channel_name="Public",
        )

    assert exc_info.value.status_code == 400
    assert "not supported" in exc_info.value.detail


@pytest.mark.asyncio
async def test_shared_link_rejects_task_runtime_extensions() -> None:
    request = TaskCreateRequest(
        title="shared task",
        runtime_extensions={"browser_runtime": {}},
    )
    context = ShareChatAccessContext(
        user=SimpleNamespace(id=7),  # type: ignore[arg-type]
        share_token="share",
        guest_id="guest",
        agent=SimpleNamespace(id=1),  # type: ignore[arg-type]
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_share_chat_task(
            request=request,
            access_context=context,
            db=object(),  # type: ignore[arg-type]
            default_channel_name="Shared",
        )

    assert exc_info.value.status_code == 400
    assert "not supported" in exc_info.value.detail
