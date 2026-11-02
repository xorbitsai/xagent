"""Authenticated browser transports for model-layer streaming speech."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Coroutine
from contextlib import aclosing, suppress
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session
from starlette.types import Message, Receive, Scope, Send
from starlette.websockets import WebSocketState

from ...core.model.asr.adapter import get_asr_model_instance
from ...core.model.asr.base import ASRStreamSession, BaseASR
from ...core.model.tts.adapter import get_tts_model_instance
from ...core.model.tts.base import BaseTTS, TTSChunk
from ...core.utils.security import redact_sensitive_text
from ..auth_dependencies import get_current_user
from ..models.database import get_db, get_session_local, release_db_connection_if_clean
from ..models.user import User
from ..schemas.speech import ASRStreamStart, TTSStreamRequest
from ..services.db_runtime import (
    cancel_and_drain_async_task,
    drain_async_task_cancellation_safe,
    run_db_io_cancellation_safe,
)
from ..services.model_service import _passes_model_visibility_hook
from ..tracking.standalone_usage import usage_scope
from .model import _resolve_accessible_model
from .websocket_auth import (
    _WebSocketAuthenticationTerminated,
    accept_authenticated_websocket,
    get_authenticated_user,
    revalidate_websocket_session,
)

logger = logging.getLogger(__name__)
speech_router = APIRouter(prefix="/api/models/speech", tags=["speech"])

# Wire limits keep this short-interaction transport bounded. Audio fragments
# are sent directly to the provider; no recording or unbounded queue is kept.
_AUDIO_FRAME_BYTES = 64 * 1024
_CONTROL_FRAME_BYTES = 4096
_IO_TIMEOUT = 30.0
_SESSION_SECONDS = 600.0
# Terminal frames are best effort, including on a transport under backpressure.
_TERMINAL_TIMEOUT = 1.0


class _ASRSessionTimeout(TimeoutError):
    """A local transport deadline, rather than a provider failure."""


class _ASRWriteTimeout(_ASRSessionTimeout):
    """The outbound transport stalled; do not send another data frame."""


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


async def _race(*operations: Coroutine[Any, Any, None]) -> None:
    """Stop and drain both halves when either finishes, fails, or disconnects."""
    tasks = [asyncio.create_task(operation) for operation in operations]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task
    finally:

        async def cleanup() -> None:
            for task in tasks:
                await cancel_and_drain_async_task(task)

        await drain_async_task_cancellation_safe(asyncio.create_task(cleanup()))


def _log_provider_error(
    model: BaseASR | BaseTTS | SimpleNamespace | None, error: Exception
) -> None:
    detail = str(error) or type(error).__name__
    key = getattr(model, "api_key", None)
    if key:
        detail = detail.replace(key, "[REDACTED]")
    logger.warning("Speech stream failed: %s", redact_sensitive_text(detail))


async def _close_model(model: BaseASR | BaseTTS) -> None:
    async def close() -> None:
        method = getattr(model, "aclose", None)
        if callable(method):
            await method()

    await drain_async_task_cancellation_safe(asyncio.create_task(close()))


async def _receive_asr_message(websocket: WebSocket) -> Message:
    try:
        return await asyncio.wait_for(websocket.receive(), _IO_TIMEOUT)
    except TimeoutError:
        raise _ASRSessionTimeout from None


async def _close_asr_socket(
    websocket: WebSocket, code: int = 1000, reason: str = ""
) -> None:
    if websocket.application_state in {
        WebSocketState.CONNECTING,
        WebSocketState.CONNECTED,
    }:
        with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            await asyncio.wait_for(
                websocket.close(code=code, reason=reason), _TERMINAL_TIMEOUT
            )


async def _send_asr_error(
    websocket: WebSocket, code: str, message: str, close_code: int
) -> None:
    if websocket.application_state is WebSocketState.CONNECTED:
        with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            await asyncio.wait_for(
                websocket.send_json(
                    {"type": "error", "code": code, "message": message}
                ),
                _TERMINAL_TIMEOUT,
            )
        await _close_asr_socket(websocket, close_code, message)


async def _receive_asr_audio(
    websocket: WebSocket, ready: asyncio.Future[ASRStreamSession], start: ASRStreamStart
) -> None:
    audio_bytes = 0
    width = 2 if start.format == "pcm" else 1
    while True:
        message = await _receive_asr_message(websocket)
        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(message.get("code", 1000))
        audio = message.get("bytes")
        if audio is not None:
            if len(audio) > _AUDIO_FRAME_BYTES:
                raise ValueError("Audio frame exceeds 64 KiB")
            if len(audio) % width:
                raise ValueError("PCM fragments must contain complete 16-bit samples")
            audio_bytes += len(audio)
            if audio_bytes > start.sample_rate * width * _SESSION_SECONDS:
                raise ValueError("ASR audio duration limit reached")
            stream = await asyncio.shield(ready)
            await stream.send_audio(audio)
        else:
            text = message.get("text") or ""
            if len(text.encode("utf-8")) > _CONTROL_FRAME_BYTES:
                raise ValueError("Control frame exceeds 4 KiB")
            command = json.loads(text)
            if command == {"type": "close"}:
                return
            if command != {"type": "commit"}:
                raise ValueError("Expected binary audio, commit, or close")
            stream = await asyncio.shield(ready)
            await stream.commit()


async def _relay_asr_events(
    websocket: WebSocket,
    model: BaseASR,
    ready: asyncio.Future[ASRStreamSession],
    start: ASRStreamStart,
) -> None:
    options = start.model_dump(exclude={"type", "model_id"}, exclude_none=True)
    try:
        stream = await model.open_stream(**options)
    except ValueError as exc:
        _log_provider_error(model, exc)
        raise HTTPException(status_code=400, detail="ASR model unavailable") from None
    async with stream:
        ready.set_result(stream)
        async for event in stream:
            try:
                await asyncio.wait_for(websocket.send_json(asdict(event)), _IO_TIMEOUT)
            except TimeoutError:
                raise _ASRWriteTimeout from None


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
    model: BaseASR | None = None
    config: SimpleNamespace | None = None
    model_closed = False
    try:
        message = await _receive_asr_message(websocket)
        if message["type"] == "websocket.disconnect":
            return
        text = message.get("text") or ""
        if len(text.encode("utf-8")) > _CONTROL_FRAME_BYTES:
            raise ValueError("Control frame exceeds 4 KiB")
        start = ASRStreamStart.model_validate_json(text)
        if not await get_authenticated_user(websocket, token, revalidate=True):
            return
        config = await run_db_io_cancellation_safe(
            lambda: _load_speech_config(user.id, start.model_id, "asr")
        )
        try:
            model = get_asr_model_instance(config)
        except ValueError as exc:
            _log_provider_error(config, exc)
            raise HTTPException(
                status_code=400, detail="Invalid ASR model configuration"
            ) from None
        if not model.supports_streaming:
            raise HTTPException(
                status_code=400, detail="Selected ASR model does not support live audio"
            )
        async with usage_scope(user.id, usage_source="speech_transcribe"):
            try:
                deadline = asyncio.timeout(_SESSION_SECONDS)
                try:
                    async with deadline:
                        ready: asyncio.Future[ASRStreamSession] = (
                            asyncio.get_running_loop().create_future()
                        )
                        await _race(
                            _receive_asr_audio(websocket, ready, start),
                            _relay_asr_events(websocket, model, ready, start),
                        )
                except TimeoutError:
                    if not deadline.expired():
                        raise
                    raise _ASRSessionTimeout from None
            finally:
                # Close before the opening scope delivers ASR input duration.
                model_closed = True
                await _close_model(model)
    except (WebSocketDisconnect, _WebSocketAuthenticationTerminated):
        return
    except _ASRWriteTimeout:
        await _close_asr_socket(websocket, 1000, "Speech session timed out")
    except _ASRSessionTimeout:
        await _send_asr_error(
            websocket, "session_timeout", "Speech session timed out", 1000
        )
    except (ValidationError, ValueError) as exc:
        await _send_asr_error(
            websocket, "invalid_request", "Invalid speech stream request", 1008
        )
        logger.debug("Invalid ASR stream request: %s", type(exc).__name__)
    except HTTPException as exc:
        await _send_asr_error(websocket, "model_unavailable", exc.detail, 1008)
    except Exception as exc:
        _log_provider_error(model or config, exc)
        await _send_asr_error(websocket, "stream_failed", "Speech stream failed", 1011)
    finally:
        if model is not None and not model_closed:
            await _close_model(model)
        await _close_asr_socket(websocket)


class _TTSAudioResponse(StreamingResponse):
    """Own first-audio negotiation, disconnect cancellation, and metering."""

    def __init__(self, model: BaseTTS, user_id: int, request: TTSStreamRequest) -> None:
        super().__init__(content=())
        self.model = model
        self.user_id = user_id
        self.options = request.model_dump(exclude={"model_id"}, exclude_none=True)

    async def _audio(
        self, first: TTSChunk, chunks: AsyncIterator[TTSChunk]
    ) -> AsyncIterator[bytes]:
        yield first.audio
        while True:
            try:
                chunk = await asyncio.wait_for(anext(chunks), _IO_TIMEOUT)
            except StopAsyncIteration:
                return
            yield chunk.audio

    async def _serve(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            async with aclosing(self.model.synthesize_stream(**self.options)) as chunks:
                try:
                    first = await asyncio.wait_for(anext(chunks), _IO_TIMEOUT)
                except Exception as exc:
                    _log_provider_error(self.model, exc)
                    status = (
                        400
                        if isinstance(exc, ValueError)
                        else 504
                        if isinstance(exc, TimeoutError)
                        else 502
                    )
                    await JSONResponse(
                        {"detail": "Speech synthesis failed"}, status_code=status
                    )(scope, receive, send)
                    return
                self.media_type = {
                    "mp3": "audio/mpeg",
                    "wav": "audio/wav",
                    "pcm": "audio/pcm",
                }.get(first.format, "application/octet-stream")
                headers = {
                    "Cache-Control": "no-store",
                    "X-Accel-Buffering": "no",
                    "X-Audio-Format": first.format,
                    "Access-Control-Expose-Headers": "X-Audio-Format, X-Audio-Sample-Rate",
                }
                if first.sample_rate is not None:
                    headers["X-Audio-Sample-Rate"] = str(first.sample_rate)
                self.init_headers(headers)
                self.body_iterator = self._audio(first, chunks)
                await self.stream_response(send)
        finally:
            await _close_model(self.model)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Meter on the request owner, outside the cancellable stream/disconnect
        # race. An EOF-triggered disconnect must not detach the quota hook.
        async with usage_scope(self.user_id, usage_source="speech_synthesize"):
            await self._respond(scope, receive, send)

    async def _respond(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = False

        async def send_response(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            async with asyncio.timeout(_SESSION_SECONDS):
                await _race(
                    self._serve(scope, receive, send_response),
                    self.listen_for_disconnect(receive),
                )
        except Exception as exc:
            _log_provider_error(self.model, exc)
            if started:
                # Preserve the truncated-response signal without exposing SDK details
                # through an ASGI exception traceback.
                raise RuntimeError("Speech synthesis stream interrupted") from None
            await JSONResponse(
                {"detail": "Speech synthesis failed"},
                status_code=504 if isinstance(exc, TimeoutError) else 502,
            )(scope, receive, send)


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
    try:
        model = get_tts_model_instance(config)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid TTS model configuration"
        ) from exc
    if not model.supports_streaming:
        await _close_model(model)
        raise HTTPException(
            status_code=400, detail="Selected TTS model does not support streaming"
        )
    return _TTSAudioResponse(model, user_id, request)
