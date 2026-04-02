import logging
import os
from typing import Any, List

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from xagent.web.api.auth import get_current_user
from xagent.web.models.database import get_db
from xagent.web.models.user import User
from xagent.web.models.user_channel import UserChannel
from xagent.web.schemas.user_channel import (
    UserChannelCreate,
    UserChannelResponse,
    UserChannelUpdate,
)

router = APIRouter()
logger = logging.getLogger(__name__)


async def trigger_telegram_sync() -> None:
    """Helper to safely trigger telegram bot sync in background"""
    from xagent.web.channels.telegram.bot import get_telegram_channel

    tg = get_telegram_channel()

    try:
        await tg._sync_bots_async()
        logger.info("Successfully triggered telegram sync in main event loop")
    except Exception as e:
        logger.error(f"Failed to trigger telegram sync: {e}")


async def trigger_feishu_sync() -> None:
    """Helper to safely trigger feishu bot sync in background"""
    from xagent.web.channels.feishu.bot import get_feishu_channel

    fs = get_feishu_channel()

    try:
        await fs._sync_bots_async()
        logger.info("Successfully triggered feishu sync in main event loop")
    except Exception as e:
        logger.error(f"Failed to trigger feishu sync: {e}")


def get_telegram_bot_name_sync(token: str) -> str:
    try:
        proxy_url = (
            os.getenv("HTTPS_PROXY")
            or os.getenv("https_proxy")
            or os.getenv("HTTP_PROXY")
            or os.getenv("http_proxy")
        )

        with httpx.Client(proxy=proxy_url) as client:
            resp = client.get(
                f"https://api.telegram.org/bot{token}/getMe", timeout=10.0
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    return str(data["result"].get("first_name", "Telegram Bot"))
    except Exception as e:
        logger.error(f"Failed to fetch telegram bot name: {e}")
    return "Telegram Bot"


def get_feishu_bot_name_sync(app_id: str, app_secret: str) -> str:
    try:
        with httpx.Client() as client:
            url = (
                "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            )
            resp = client.post(
                url, json={"app_id": app_id, "app_secret": app_secret}, timeout=10.0
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 0:
                    token = data["tenant_access_token"]
                    info_url = "https://open.feishu.cn/open-apis/bot/v3/info"
                    info_resp = client.get(
                        info_url,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10.0,
                    )
                    if info_resp.status_code == 200:
                        info_data = info_resp.json()
                        if info_data.get("code") == 0:
                            return str(info_data["bot"].get("app_name", "Feishu Bot"))
    except Exception as e:
        logger.error(f"Failed to fetch feishu bot name: {e}")
    return "Feishu Bot"


@router.get("", response_model=List[UserChannelResponse])
def get_user_channels(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Get all channels configured by the current user."""
    channels = (
        db.query(UserChannel).filter(UserChannel.user_id == current_user.id).all()
    )
    return channels


@router.post("", response_model=UserChannelResponse)
def create_user_channel(
    channel_in: UserChannelCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Create a new channel configuration."""

    # Auto-fetch channel name if not provided
    channel_name = channel_in.channel_name
    if not channel_name or not channel_name.strip():
        if channel_in.channel_type == "telegram":
            token = channel_in.config.get("bot_token", "")
            channel_name = (
                get_telegram_bot_name_sync(token) if token else "Telegram Bot"
            )
        elif channel_in.channel_type == "feishu":
            app_id = channel_in.config.get("app_id", "")
            app_secret = channel_in.config.get("app_secret", "")
            channel_name = (
                get_feishu_bot_name_sync(app_id, app_secret)
                if app_id and app_secret
                else "Feishu Bot"
            )
        else:
            channel_name = "Unknown Bot"

    # Check for duplicate name or token
    existing_channels = (
        db.query(UserChannel)
        .filter(UserChannel.channel_type == channel_in.channel_type)
        .all()
    )

    for ch in existing_channels:
        if ch.user_id == current_user.id and ch.channel_name == channel_name:
            raise HTTPException(status_code=400, detail="Channel name already exists")

        ch_token = ch.config.get("bot_token")
        in_token = channel_in.config.get("bot_token")
        if ch_token and in_token and ch_token == in_token:
            raise HTTPException(status_code=400, detail="Bot token already exists")

    channel = UserChannel(
        user_id=current_user.id,
        channel_type=channel_in.channel_type,
        channel_name=channel_name,
        config=channel_in.config,
        is_active=channel_in.is_active,
    )
    db.add(channel)
    db.commit()
    db.refresh(channel)

    # Trigger bot reload via background task
    if channel.channel_type == "telegram":
        background_tasks.add_task(trigger_telegram_sync)
    elif channel.channel_type == "feishu":
        background_tasks.add_task(trigger_feishu_sync)

    return channel


@router.put("/{channel_id}", response_model=UserChannelResponse)
def update_user_channel(
    channel_id: int,
    channel_in: UserChannelUpdate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Update a channel configuration."""
    channel = (
        db.query(UserChannel)
        .filter(UserChannel.id == channel_id, UserChannel.user_id == current_user.id)
        .first()
    )
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    # Check for duplicate name or token
    existing_channels = (
        db.query(UserChannel)
        .filter(
            UserChannel.channel_type == channel.channel_type,
            UserChannel.id != channel_id,
        )
        .all()
    )

    new_name = (
        channel_in.channel_name
        if channel_in.channel_name is not None
        else str(channel.channel_name)
    )
    new_config = channel_in.config if channel_in.config is not None else channel.config

    if not new_name or not new_name.strip():
        if channel.channel_type == "telegram":
            token = new_config.get("bot_token", "") if new_config else ""
            new_name = get_telegram_bot_name_sync(token) if token else "Telegram Bot"
        elif channel.channel_type == "feishu":
            app_id = new_config.get("app_id", "") if new_config else ""
            app_secret = new_config.get("app_secret", "") if new_config else ""
            new_name = (
                get_feishu_bot_name_sync(app_id, app_secret)
                if app_id and app_secret
                else "Feishu Bot"
            )
        else:
            new_name = "Unknown Bot"

    for ch in existing_channels:
        if ch.user_id == current_user.id and ch.channel_name == new_name:
            raise HTTPException(status_code=400, detail="Channel name already exists")

        ch_token = ch.config.get("bot_token")
        in_token = new_config.get("bot_token") if new_config else None
        if ch_token and in_token and ch_token == in_token:
            raise HTTPException(status_code=400, detail="Bot token already exists")

    update_data = channel_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(channel, field, value)

    channel.channel_name = new_name  # type: ignore[assignment]

    db.commit()
    db.refresh(channel)

    if channel.channel_type == "telegram":
        background_tasks.add_task(trigger_telegram_sync)
    elif channel.channel_type == "feishu":
        background_tasks.add_task(trigger_feishu_sync)

    return channel


@router.delete("/{channel_id}")
def delete_user_channel(
    channel_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Any:
    """Delete a channel configuration."""
    channel = (
        db.query(UserChannel)
        .filter(UserChannel.id == channel_id, UserChannel.user_id == current_user.id)
        .first()
    )
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    channel_type = channel.channel_type

    db.delete(channel)
    db.commit()

    if channel_type == "telegram":
        background_tasks.add_task(trigger_telegram_sync)
    elif channel_type == "feishu":
        background_tasks.add_task(trigger_feishu_sync)

    return {"status": "success"}
