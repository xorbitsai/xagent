"""Authenticated browser transports for model-layer streaming speech.

These routes own login authentication and resolve the caller's own model
configuration; the streaming sessions themselves run in
:mod:`xagent.web.services.speech_streams`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ...core.model.asr.adapter import get_asr_model_instance
from ...core.model.tts.adapter import get_tts_model_instance
from ..auth_dependencies import get_current_user
from ..models.database import get_db, get_session_local, release_db_connection_if_clean
from ..models.user import User
from ..schemas.speech import ASRStreamStart, TTSStreamRequest
from ..services.db_runtime import run_db_io_cancellation_safe
from ..services.model_service import _passes_model_visibility_hook
from ..services.speech_streams import (
    SpeechStreamPlan,
    _close_asr_socket,
    _close_model,
    _log_provider_error,
    run_asr_stream,
    tts_stream_response,
)
from .model import _resolve_accessible_model
from .websocket_auth import (
    _WebSocketAuthenticationTerminated,
    accept_authenticated_websocket,
    get_authenticated_user,
    revalidate_websocket_session,
)

speech_router = APIRouter(prefix="/api/models/speech", tags=["speech"])


def _load_speech_config(
    user_id: int, model_id: str, ability: Literal["asr", "tts"]
) -> SimpleNamespace:
    """Resolve normal model visibility in a short, worker-owned Session."""
    with get_session_local()() as db:
        user = db.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="Model not found")
        _, model, _ = _resolve_accessible_model(db, user, model_id)
        if not model.is_active or not _passes_model_visibility_hook(db, user_id, model):
            raise HTTPException(status_code=404, detail="Model not found")
        if model.category != "speech" or ability not in (model.abilities or []):
            raise HTTPException(status_code=400, detail="Incompatible speech model")
        return SimpleNamespace(
            **{
                name: getattr(model, name)
                for name in (
                    "model_id",
                    "model_provider",
                    "model_name",
                    "api_key",
                    "base_url",
                    "language",
                    "voice",
                    "format",
                    "sample_rate",
                )
            }
        )


@speech_router.websocket("/asr/stream")
@revalidate_websocket_session
async def stream_asr(
    websocket: WebSocket,
    token: str | None = Query(None, description="Internal header credential carrier"),
) -> None:
    try:
        user = await get_authenticated_user(websocket, token)
    except _WebSocketAuthenticationTerminated:
        return
    if user is None:
        await _close_asr_socket(websocket, 4001, "Authentication required")
        return
    await accept_authenticated_websocket(websocket)

    async def prepare(text: str) -> SpeechStreamPlan[ASRStreamStart]:
        start = ASRStreamStart.model_validate_json(text)
        if not await get_authenticated_user(websocket, token, revalidate=True):
            # The session is gone or already closed; nothing more to send.
            raise _WebSocketAuthenticationTerminated()
        config = await run_db_io_cancellation_safe(
            lambda: _load_speech_config(user.id, start.model_id, "asr")
        )
        return SpeechStreamPlan(
            user_id=user.id,
            config=config,
            options=start,
            usage_source="speech_transcribe",
            audience="member",
        )

    await run_asr_stream(websocket, prepare=prepare)


@speech_router.post("/tts/stream", response_class=StreamingResponse)
async def stream_tts(
    request: TTSStreamRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    user_id = int(user.id)
    release_db_connection_if_clean(db)
    config = await run_db_io_cancellation_safe(
        lambda: _load_speech_config(user_id, request.model_id, "tts")
    )
    return await tts_stream_response(
        plan=SpeechStreamPlan(
            user_id=user_id,
            config=config,
            options=request,
            usage_source="speech_synthesize",
            audience="member",
        )
    )


@speech_router.get("/capabilities")
async def speech_capabilities(
    model_id: str = Query(min_length=1, max_length=512),
    ability: Literal["asr", "tts"] = Query(),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    user_id = int(user.id)
    release_db_connection_if_clean(db)
    config = await run_db_io_cancellation_safe(
        lambda: _load_speech_config(user_id, model_id, ability)
    )
    try:
        model = (
            get_asr_model_instance(config)
            if ability == "asr"
            else get_tts_model_instance(config)
        )
    except ValueError as exc:
        _log_provider_error(config, exc)
        raise HTTPException(
            status_code=400, detail="Invalid speech model configuration"
        ) from None
    try:
        capabilities = await model.get_streaming_capabilities()
        return {
            key: capabilities[key]
            for key in ("streaming", "server_vad", "formats", "sample_rates", "voices")
            if key in capabilities
        }
    except Exception as exc:
        _log_provider_error(model, exc)
        raise HTTPException(
            status_code=504 if isinstance(exc, TimeoutError) else 502,
            detail="Speech capabilities unavailable",
        ) from None
    finally:
        await _close_model(model)
