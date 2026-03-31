import asyncio
import logging
from typing import Any, List

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


def trigger_telegram_sync() -> None:
    """Helper to safely trigger telegram bot sync in background"""
    from xagent.web.channels.telegram.bot import get_telegram_channel

    tg = get_telegram_channel()

    # Send a request to the main event loop to sync bots
    # We shouldn't create a new event loop and run aiogram tasks
    # because they need to run in the main thread/event loop
    from xagent.web.app import app

    try:
        if hasattr(app.state, "telegram_task"):
            # Get the running loop where telegram_task was created
            loop = app.state.telegram_task.get_loop()
            asyncio.run_coroutine_threadsafe(tg._sync_bots_async(), loop)
            logger.info("Successfully triggered telegram sync in main event loop")
        else:
            logger.warning("Telegram task not found in app state.")
    except Exception as e:
        logger.error(f"Failed to trigger telegram sync: {e}")


def trigger_feishu_sync() -> None:
    """Helper to safely trigger feishu bot sync in background"""
    from xagent.web.channels.feishu.bot import get_feishu_channel

    fs = get_feishu_channel()

    from xagent.web.app import app

    try:
        if hasattr(app.state, "feishu_task"):
            loop = app.state.feishu_task.get_loop()
            asyncio.run_coroutine_threadsafe(fs._sync_bots_async(), loop)
            logger.info("Successfully triggered feishu sync in main event loop")
        else:
            logger.warning("Feishu task not found in app state.")
    except Exception as e:
        logger.error(f"Failed to trigger feishu sync: {e}")


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
    # Check for duplicate name or token
    existing_channels = (
        db.query(UserChannel)
        .filter(UserChannel.channel_type == channel_in.channel_type)
        .all()
    )

    for ch in existing_channels:
        if ch.user_id == current_user.id and ch.channel_name == channel_in.channel_name:
            raise HTTPException(status_code=400, detail="Channel name already exists")

        ch_token = ch.config.get("bot_token")
        in_token = channel_in.config.get("bot_token")
        if ch_token and in_token and ch_token == in_token:
            raise HTTPException(status_code=400, detail="Bot token already exists")

    channel = UserChannel(
        user_id=current_user.id,
        channel_type=channel_in.channel_type,
        channel_name=channel_in.channel_name,
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
        else channel.channel_name
    )
    new_config = channel_in.config if channel_in.config is not None else channel.config

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
