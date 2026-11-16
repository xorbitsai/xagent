"""Streaming speech sessions shared by the speech transport routes.

A route owns authentication and the WebSocket handshake; everything after
that lives here. :func:`run_asr_stream` takes over an accepted transcription
socket and :func:`tts_stream_response` takes over a synthesis request whose
model configuration is already loaded. Both assume the caller has already
authenticated the request and chosen who pays: they never check credentials
themselves, and they bill :attr:`SpeechStreamPlan.user_id`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import aclosing, suppress
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Generic, TypeVar

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.types import Message, Receive, Scope, Send
from starlette.websockets import WebSocketState

from ...core.model.asr.adapter import get_asr_model_instance
from ...core.model.asr.base import ASRStreamSession, BaseASR
from ...core.model.tts.adapter import get_tts_model_instance
from ...core.model.tts.base import BaseTTS, TTSChunk
from ...core.utils.security import redact_sensitive_text
from ..api.websocket_auth import _WebSocketAuthenticationTerminated
from ..schemas.speech import ASRStreamStart, TTSStreamRequest
from ..tracking.standalone_usage import (
    STANDALONE_USAGE_REFUSED,
    StandaloneUsageRefused,
    UsageSource,
    check_standalone_gate_async,
    usage_scope,
)
from .db_runtime import cancel_and_drain_async_task, drain_async_task_cancellation_safe
from .release_hold import WritersRetired, release_hold

logger = logging.getLogger(__name__)

# Wire limits keep this short-interaction transport bounded. Audio fragments
# are sent directly to the provider; no recording or unbounded queue is kept.
_AUDIO_FRAME_BYTES = 64 * 1024
_CONTROL_FRAME_BYTES = 4096
_IO_TIMEOUT = 30.0
_SESSION_SECONDS = 600.0
# Terminal frames are best effort, including on a transport under backpressure.
_TERMINAL_TIMEOUT = 1.0

_OptionsT = TypeVar("_OptionsT", ASRStreamStart, TTSStreamRequest)


@dataclass(frozen=True)
class SpeechStreamPlan(Generic[_OptionsT]):
    """What a route resolved for one speech session before the provider runs.

    ``user_id`` is the account that pays: the gate is asked for it and usage is
    metered on it. ``config`` is the loaded model configuration the provider
    instance is built from. ``options`` is the validated client request: the
    ASR start frame, or the TTS request whose text and output options are
    synthesized.
    """

    user_id: int
    config: SimpleNamespace
    options: _OptionsT
    usage_source: UsageSource


class _ASRSessionTimeout(TimeoutError):
    """A local transport deadline, rather than a provider failure."""


class _ASRWriteTimeout(_ASRSessionTimeout):
    """The outbound transport stalled; do not send another data frame."""


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
    websocket: WebSocket,
    code: str,
    message: str,
    close_code: int,
    close_reason: str | None = None,
) -> None:
    if websocket.application_state is WebSocketState.CONNECTED:
        with suppress(WebSocketDisconnect, RuntimeError, OSError, TimeoutError):
            await asyncio.wait_for(
                websocket.send_json(
                    {"type": "error", "code": code, "message": message}
                ),
                _TERMINAL_TIMEOUT,
            )
        await _close_asr_socket(
            websocket, close_code, message if close_reason is None else close_reason
        )


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


async def run_asr_stream(
    websocket: WebSocket,
    *,
    prepare: Callable[[str], Awaitable[SpeechStreamPlan[ASRStreamStart]]],
) -> None:
    """Serve one live transcription session on an accepted, authenticated socket.

    The caller must have authenticated the client and accepted the handshake.
    From there this owns the whole session: the release-hold write token, the
    first frame (within ``_IO_TIMEOUT``), the provider instance, metering, the
    ``_SESSION_SECONDS`` limit, every error frame and close code, and closing
    the socket. ``prepare`` receives the raw first text frame and returns the
    plan to run; it may raise ``ValidationError`` or ``ValueError`` (invalid
    request), ``HTTPException`` (model unavailable), or
    ``_WebSocketAuthenticationTerminated`` (the client was already answered),
    which are reported exactly as the same failures inside the session are.
    """
    # Transcription meters usage when it ends, so a session is a socket write
    # the host's retire waits for (and winds down when the fence goes up). A
    # retired process cannot write, so the client reconnects to the new one;
    # the close frame needs the handshake accepted first.
    session_task = asyncio.current_task()
    assert session_task is not None
    try:
        write_token = release_hold.acquire_socket_write(session_task.cancel)
    except WritersRetired:
        await _close_asr_socket(websocket, 1012, "Service restart")
        return
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
        plan = await prepare(text)
        config = plan.config
        start = plan.options
        try:
            model = get_asr_model_instance(config)
        except ValueError as exc:
            _log_provider_error(config, exc)
            raise HTTPException(
                status_code=400, detail="Invalid ASR model configuration"
            ) from None
        if not (await model.get_streaming_capabilities()).get("streaming"):
            raise HTTPException(
                status_code=400, detail="Selected ASR model does not support live audio"
            )
        async with usage_scope(
            plan.user_id,
            usage_source=plan.usage_source,
            model_ids=(config.model_id,),
        ):
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
    except asyncio.CancelledError:
        if not release_hold.socket_writes_fenced:
            raise
        # The retire fence wound this session down: tell the client to
        # reconnect to the successor instead of leaking the cancellation. The
        # usage scope has already delivered what was metered (on its own
        # tracked worker), and the token is held until the finally below.
        if session_task.uncancel() > 0:
            # A foreign cancel (shutdown, the revalidate wrapper) landed with
            # the fence cancel: it must not be swallowed.
            raise
        await _close_asr_socket(websocket, 1012, "Service restart")
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
    except StandaloneUsageRefused as exc:
        # The gate's message can outgrow a close frame's 123-byte reason.
        await _send_asr_error(
            websocket,
            STANDALONE_USAGE_REFUSED,
            exc.message,
            1008,
            close_reason="Usage limit reached",
        )
    except HTTPException as exc:
        await _send_asr_error(websocket, "model_unavailable", exc.detail, 1008)
    except Exception as exc:
        _log_provider_error(model or config, exc)
        await _send_asr_error(websocket, "stream_failed", "Speech stream failed", 1011)
    finally:
        try:
            if model is not None and not model_closed:
                await _close_model(model)
            await _close_asr_socket(websocket)
        finally:
            write_token.release()


class _TTSAudioResponse(StreamingResponse):
    """Own first-audio negotiation, disconnect cancellation, and metering."""

    def __init__(
        self,
        model: BaseTTS,
        user_id: int,
        request: TTSStreamRequest,
        usage_source: UsageSource = "speech_synthesize",
    ) -> None:
        super().__init__(content=())
        self.model = model
        self.user_id = user_id
        # Metered under the same source the gate was asked about.
        self.usage_source: UsageSource = usage_source
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
        try:
            async with usage_scope(
                self.user_id,
                usage_source=self.usage_source,
                # Stamped by get_tts_model_instance from the model row.
                model_ids=(getattr(self.model, "model_id", ""),),
            ):
                await self._respond(scope, receive, send)
        except StandaloneUsageRefused as exc:
            # Only entering the scope raises this: _respond turns every failure
            # into a response of its own. Nothing was sent or opened yet.
            await _close_model(self.model)
            await JSONResponse({"detail": exc.detail}, status_code=exc.status_code)(
                scope, receive, send
            )

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


async def tts_stream_response(
    *, plan: SpeechStreamPlan[TTSStreamRequest]
) -> StreamingResponse:
    """Build the streaming response for one synthesis request.

    The caller must have authenticated the request and loaded ``plan.config``.
    In order, this asks the usage gate (``StandaloneUsageRefused``, HTTP 402),
    builds the provider instance, checks that it streams, and returns a
    response that meters ``plan.user_id`` while it is sent. Failures before
    the response exists raise ``HTTPException``; the instance is closed on
    every such path.
    """
    # The metering scope opens once the response is being sent, too late for a
    # status code; refuse here, before the provider is reached. The scope
    # checks again, for a gate that turns in between.
    await check_standalone_gate_async(
        plan.user_id, plan.usage_source, model_ids=(plan.config.model_id,)
    )
    try:
        model = get_tts_model_instance(plan.config)
    except ValueError as exc:
        _log_provider_error(plan.config, exc)
        raise HTTPException(
            status_code=400, detail="Invalid TTS model configuration"
        ) from exc
    try:
        capabilities = await model.get_streaming_capabilities()
    except BaseException as exc:
        await _close_model(model)
        if isinstance(exc, Exception):
            _log_provider_error(model, exc)
            raise HTTPException(
                status_code=504 if isinstance(exc, TimeoutError) else 502,
                detail="Speech capabilities unavailable",
            ) from None
        raise
    if not capabilities.get("streaming"):
        await _close_model(model)
        raise HTTPException(
            status_code=400, detail="Selected TTS model does not support streaming"
        )
    return _TTSAudioResponse(
        model, plan.user_id, plan.options, usage_source=plan.usage_source
    )
