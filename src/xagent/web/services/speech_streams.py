"""Streaming speech sessions shared by the speech transport routes.

A route owns authentication and the WebSocket handshake; everything after
that lives here. :func:`run_asr_stream` takes over an accepted transcription
socket and :func:`tts_stream_response` takes over a synthesis request whose
model configuration is already loaded. Both assume the caller has already
authenticated the request and chosen who pays: they never check credentials
themselves, and they bill :attr:`SpeechStreamPlan.user_id`.

A plan's ``audience`` names who is calling. ``member`` (a signed-in user)
keeps the established responses. ``guest`` (a widget visitor who is not signed
in, with the usage recorded on the widget owner) adds the daily voice budget
and, for transcription, the per-process caps on live guest sessions; every
guest failure is answered with a fixed code and message (a
``{"detail": {"code", "message"}}`` body, or an error frame and close reason)
that carries no provider text, owner configuration, or usage-gate detail.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import aclosing, suppress
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Generic, Literal, TypeVar

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.types import Message, Receive, Scope, Send
from starlette.websockets import WebSocketState

from ...config import (
    get_widget_voice_max_sessions,
    get_widget_voice_max_sessions_per_entity,
    get_widget_voice_max_sessions_per_visitor,
)
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
from .widget_voice_budget import (
    Reservation,
    VoiceBudgetExhausted,
    VoiceBudgetUnavailable,
    refund,
    reserve_asr,
    reserve_tts,
)

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
Audience = Literal["member", "guest"]

# The only text a guest ever receives for a refusal, keyed by its code. A
# refused periodic recheck sends no error frame; its close carries the
# callback's code with the "voice_unavailable" text as the reason.
GUEST_REFUSAL_MESSAGES = {
    "invalid_request": "Invalid speech request",
    "voice_disabled": "Voice is not available for this chat",
    "rate_limited": "Too many requests",
    "voice_unavailable": "Voice is unavailable",
    STANDALONE_USAGE_REFUSED: "Usage limit reached",
    "model_unavailable": "Voice is unavailable",
    "voice_budget_exhausted": "Voice limit reached, try again later",
    "speech_synthesis_failed": "Speech synthesis failed",
    "voice_busy": "Voice is busy",
    "session_timeout": "Speech session timed out",
    "stream_failed": "Speech stream failed",
}


def guest_refusal_detail(code: str) -> dict[str, str]:
    return {"code": code, "message": GUEST_REFUSAL_MESSAGES[code]}


def guest_refusal(status_code: int, code: str) -> HTTPException:
    """A guest-facing refusal: a fixed code and its fixed message."""
    return HTTPException(status_code=status_code, detail=guest_refusal_detail(code))


@dataclass(frozen=True)
class SpeechStreamPlan(Generic[_OptionsT]):
    """What a route resolved for one speech session before the provider runs.

    ``user_id`` is the account that pays: the gate is asked for it and usage is
    metered on it. ``config`` is the loaded model configuration the provider
    instance is built from. ``options`` is the validated client request: the
    ASR start frame, or the TTS request whose text and output options are
    synthesized.

    ``audience`` is ``member`` for a signed-in caller and ``guest`` for a
    widget visitor. A guest plan must name the widget ``entity_key`` and the
    ``visitor_key`` its budget is counted under; a member plan names neither.
    Construction raises ``ValueError`` otherwise.
    """

    user_id: int
    config: SimpleNamespace
    options: _OptionsT
    usage_source: UsageSource
    audience: Audience
    entity_key: str | None = None
    visitor_key: str | None = None

    def __post_init__(self) -> None:
        if self.audience == "guest":
            if not self.entity_key or not self.visitor_key:
                raise ValueError("A guest speech plan needs entity and visitor keys")
        elif self.audience == "member":
            if self.entity_key is not None or self.visitor_key is not None:
                raise ValueError("A member speech plan carries no budget keys")
        else:
            raise ValueError(f"Unknown speech audience: {self.audience!r}")


# Settlement clock for a guest transcription session's budget.
_now = time.monotonic


class _GuestSessions:
    """Live guest transcription sessions in this process: in total, per
    widget entity, and per visitor of an entity. Only touched from the event
    loop."""

    def __init__(self) -> None:
        self.total = 0
        self.by_entity: dict[str, int] = {}
        # Keyed like the per-visitor budget: ``entity_key|visitor_key``.
        self.by_visitor: dict[str, int] = {}

    def admit(self, entity_key: str, visitor_key: str) -> bool:
        """Take a slot when all three caps allow it. There is no await between
        the checks and the increments, so concurrent sessions cannot
        overshoot, and a refused session changes no count."""
        total_cap = get_widget_voice_max_sessions()
        entity_cap = get_widget_voice_max_sessions_per_entity()
        visitor_cap = get_widget_voice_max_sessions_per_visitor()
        visitor = f"{entity_key}|{visitor_key}"
        entity_held = self.by_entity.get(entity_key, 0)
        visitor_held = self.by_visitor.get(visitor, 0)
        if (
            (total_cap and self.total >= total_cap)
            or (entity_cap and entity_held >= entity_cap)
            or (visitor_cap and visitor_held >= visitor_cap)
        ):
            return False
        self.total += 1
        self.by_entity[entity_key] = entity_held + 1
        self.by_visitor[visitor] = visitor_held + 1
        return True

    def release(self, entity_key: str, visitor_key: str) -> None:
        self.total -= 1
        for counts, key in (
            (self.by_entity, entity_key),
            (self.by_visitor, f"{entity_key}|{visitor_key}"),
        ):
            held = counts[key] - 1
            if held:
                counts[key] = held
            else:
                del counts[key]


_guest_sessions = _GuestSessions()


class _RevalidationRefused(Exception):
    """The session's periodic recheck returned a close code."""

    def __init__(self, close_code: int) -> None:
        super().__init__(close_code)
        self.close_code = close_code


class _RevalidationFailed(Exception):
    """The session's periodic recheck itself raised."""


@dataclass
class _ASRMeter:
    """What a transcription session has used, for its budget settlement."""

    audio_bytes: int = 0
    opened: bool = False


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
    websocket: WebSocket,
    ready: asyncio.Future[ASRStreamSession],
    start: ASRStreamStart,
    meter: _ASRMeter,
) -> None:
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
            meter.audio_bytes += len(audio)
            if meter.audio_bytes > start.sample_rate * width * _SESSION_SECONDS:
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
    meter: _ASRMeter,
) -> None:
    options = start.model_dump(exclude={"type", "model_id"}, exclude_none=True)
    try:
        stream = await model.open_stream(**options)
    except ValueError as exc:
        _log_provider_error(model, exc)
        raise HTTPException(status_code=400, detail="ASR model unavailable") from None
    async with stream:
        meter.opened = True
        ready.set_result(stream)
        async for event in stream:
            try:
                await asyncio.wait_for(websocket.send_json(asdict(event)), _IO_TIMEOUT)
            except TimeoutError:
                raise _ASRWriteTimeout from None


async def _recheck(
    revalidate: Callable[[], Awaitable[int | None]], every: float
) -> None:
    """Run ``revalidate`` every ``every`` seconds until it refuses or raises."""
    while True:
        await asyncio.sleep(every)
        try:
            close_code = await revalidate()
        except Exception as exc:
            raise _RevalidationFailed() from exc
        if close_code is not None:
            raise _RevalidationRefused(close_code)


def _settled_seconds(
    meter: _ASRMeter, started: float | None, start: ASRStreamStart | None
) -> int:
    """Seconds a transcription session is charged: 0 when the provider stream
    never opened; otherwise the longer of the audio received and the wall
    time since ``started``, rounded up and capped at the session limit."""
    if started is None or start is None or not meter.opened:
        return 0
    width = 2 if start.format == "pcm" else 1
    audio = meter.audio_bytes / (start.sample_rate * width)
    used = max(audio, _now() - started)
    return min(int(_SESSION_SECONDS), math.ceil(used))


def _log_asr_usage(
    plan: SpeechStreamPlan[ASRStreamStart], seconds: int, result: str
) -> None:
    """One line per transcription session, written when it ends; always on."""
    logger.info(
        "Speech transcription usage: audience=%s user_id=%s entity=%s visitor=%s "
        "model=%s seconds=%d result=%s",
        plan.audience,
        plan.user_id,
        plan.entity_key,
        _loggable_visitor(plan.visitor_key),
        plan.config.model_id,
        seconds,
        result,
    )


async def run_asr_stream(
    websocket: WebSocket,
    *,
    prepare: Callable[[str], Awaitable[SpeechStreamPlan[ASRStreamStart]]],
    audience: Audience,
    revalidate: Callable[[], Awaitable[int | None]] | None = None,
    recheck_seconds: float | None = None,
) -> None:
    """Serve one live transcription session on an accepted, authenticated socket.

    The caller must have authenticated the client and accepted the handshake.
    From there this owns the whole session: the release-hold write token, the
    first frame (within ``_IO_TIMEOUT``), the provider instance, metering, the
    ``_SESSION_SECONDS`` limit, every error frame and close code, and closing
    the socket. ``prepare`` receives the raw first text frame and returns the
    plan to run; it may raise ``ValidationError`` or ``ValueError`` (invalid
    request), ``HTTPException`` (model unavailable: a signed-in member gets a
    ``model_unavailable`` error frame carrying the detail, a guest gets ``voice_unavailable`` with no detail, both closing 1008), or
    ``_WebSocketAuthenticationTerminated`` (the client was already answered),
    which are reported exactly as the same failures inside the session are.

    ``audience`` (required) is the audience of the plans ``prepare`` returns,
    and a plan for another audience ends the session with 1011
    ``stream_failed`` before anything is reserved or opened. It decides,
    from the first frame on, whether failures carry the established texts
    (``member``) or only fixed guest codes and messages (``guest``, where an
    ``HTTPException`` from ``prepare`` is reported as ``voice_unavailable``).
    A guest session also takes a slot under the per-process session caps
    (1008 "Voice is busy" when none is free) and reserves the longest session
    from the daily transcription budget (1008 ``voice_budget_exhausted``, or
    ``voice_unavailable`` when the budget storage fails). When it ends, the
    slot is released and the budget keeps only the settled seconds (see
    :func:`_settled_seconds`), on every exit path including cancellation.

    ``revalidate``, given with ``recheck_seconds``, runs every
    ``recheck_seconds`` while audio is relayed: it returns ``None`` to keep
    the session or a close code to end it with (the socket is closed with
    that code and a fixed reason, and no error frame), and when it raises the
    session ends with 1011 ``stream_failed``. Every session that got a plan
    writes one usage log line.
    """
    if (revalidate is None) != (recheck_seconds is None):
        raise TypeError("revalidate and recheck_seconds are given together")
    guest = audience == "guest"
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

    async def fail(
        code: str, message: str, close_code: int, close_reason: str | None = None
    ) -> None:
        if guest:
            message, close_reason = GUEST_REFUSAL_MESSAGES[code], None
        await _send_asr_error(websocket, code, message, close_code, close_reason)

    model: BaseASR | None = None
    config: SimpleNamespace | None = None
    model_closed = False
    plan: SpeechStreamPlan[ASRStreamStart] | None = None
    admitted: tuple[str, str] | None = None
    reservation: Reservation | None = None
    meter = _ASRMeter()
    started: float | None = None
    result = "incomplete"
    try:
        message = await _receive_asr_message(websocket)
        if message["type"] == "websocket.disconnect":
            result = "disconnected"
            return
        text = message.get("text") or ""
        if len(text.encode("utf-8")) > _CONTROL_FRAME_BYTES:
            raise ValueError("Control frame exceeds 4 KiB")
        try:
            plan = await prepare(text)
        except HTTPException:
            if not guest:
                raise
            # A guest is never told why the route refused.
            result = "voice_unavailable"
            await fail("voice_unavailable", "", 1008)
            return
        if plan.audience != audience:
            raise RuntimeError("The speech plan is for another audience")
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
        if guest:
            entity_key, visitor_key = plan.entity_key, plan.visitor_key
            if not entity_key or not visitor_key:
                raise RuntimeError("A guest speech plan has no budget keys")
            if not _guest_sessions.admit(entity_key, visitor_key):
                result = "busy"
                await fail("voice_busy", GUEST_REFUSAL_MESSAGES["voice_busy"], 1008)
                return
            admitted = (entity_key, visitor_key)
            reservation = await reserve_asr(
                entity_key, visitor_key, int(_SESSION_SECONDS)
            )
        started = _now()
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
                        operations = [
                            _receive_asr_audio(websocket, ready, start, meter),
                            _relay_asr_events(websocket, model, ready, start, meter),
                        ]
                        if revalidate is not None and recheck_seconds is not None:
                            operations.append(_recheck(revalidate, recheck_seconds))
                        await _race(*operations)
                except TimeoutError:
                    if not deadline.expired():
                        raise
                    raise _ASRSessionTimeout from None
            finally:
                # Close before the opening scope delivers ASR input duration.
                model_closed = True
                await _close_model(model)
        result = "ok"
    except (WebSocketDisconnect, _WebSocketAuthenticationTerminated):
        result = "disconnected"
        return
    except asyncio.CancelledError:
        result = "cancelled"
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
        result = "service_restart"
        await _close_asr_socket(websocket, 1012, "Service restart")
    except _ASRWriteTimeout:
        result = "session_timeout"
        await _close_asr_socket(websocket, 1000, "Speech session timed out")
    except _ASRSessionTimeout:
        result = "session_timeout"
        await fail("session_timeout", "Speech session timed out", 1000)
    except _RevalidationRefused as exc:
        result = "revoked"
        # The close code is the only signal, like the routes' own access
        # refusals: an error frame would hide which code the callback chose.
        await _close_asr_socket(
            websocket, exc.close_code, GUEST_REFUSAL_MESSAGES["voice_unavailable"]
        )
    except _RevalidationFailed as exc:
        result = "recheck_failed"
        cause = exc.__cause__
        _log_provider_error(None, cause if isinstance(cause, Exception) else exc)
        await fail("stream_failed", "Speech stream failed", 1011)
    except VoiceBudgetExhausted:
        result = "budget_exhausted"
        await fail(
            "voice_budget_exhausted",
            GUEST_REFUSAL_MESSAGES["voice_budget_exhausted"],
            1008,
        )
    except VoiceBudgetUnavailable:
        result = "budget_unavailable"
        await fail("voice_unavailable", "", 1008)
    except (ValidationError, ValueError) as exc:
        result = "invalid_request"
        await fail("invalid_request", "Invalid speech stream request", 1008)
        logger.debug("Invalid ASR stream request: %s", type(exc).__name__)
    except StandaloneUsageRefused as exc:
        result = "usage_refused"
        # The gate's message can outgrow a close frame's 123-byte reason.
        await fail(
            STANDALONE_USAGE_REFUSED,
            exc.message,
            1008,
            close_reason="Usage limit reached",
        )
    except HTTPException as exc:
        result = "model_unavailable"
        await fail("model_unavailable", exc.detail, 1008)
    except Exception as exc:
        result = "stream_failed"
        _log_provider_error(model or config, exc)
        await fail("stream_failed", "Speech stream failed", 1011)
    finally:
        try:
            try:
                if model is not None and not model_closed:
                    await _close_model(model)
                await _close_asr_socket(websocket)
            finally:
                if admitted is not None:
                    _guest_sessions.release(*admitted)
                seconds = _settled_seconds(
                    meter, started, plan.options if plan is not None else None
                )
                if plan is not None:
                    _log_asr_usage(plan, seconds, result)
                if reservation is not None:
                    # Settle off the cancellable path: a cancellation waits
                    # for the refund and is raised after it.
                    await refund(reservation, reservation.amount - seconds)
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
        *,
        audience: Audience = "member",
        entity_key: str | None = None,
        visitor_key: str | None = None,
    ) -> None:
        super().__init__(content=())
        self.model = model
        self.user_id = user_id
        # Metered under the same source the gate was asked about.
        self.usage_source: UsageSource = usage_source
        self.options = request.model_dump(exclude={"model_id"}, exclude_none=True)
        self.audience: Audience = audience
        self.entity_key = entity_key
        self.visitor_key = visitor_key
        self.chars = len(request.text)
        # The outcome the usage log reports when the response ends.
        self.result = "incomplete"

    def _failure(self, status: int, detail: Any, code: str) -> JSONResponse:
        if self.audience == "guest":
            detail = guest_refusal_detail(code)
        return JSONResponse({"detail": detail}, status_code=status)

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
                    self.result = "provider_failed"
                    await self._failure(
                        status, "Speech synthesis failed", "speech_synthesis_failed"
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
                self.result = "ok"
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
            self.result = "usage_refused"
            await _close_model(self.model)
            await self._failure(exc.status_code, exc.detail, STANDALONE_USAGE_REFUSED)(
                scope, receive, send
            )
        finally:
            _log_tts_usage(
                self.audience,
                self.user_id,
                self.entity_key,
                self.visitor_key,
                getattr(self.model, "model_id", ""),
                self.chars,
                self.result,
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
            self.result = "provider_failed"
            if started:
                # Preserve the truncated-response signal without exposing SDK details
                # through an ASGI exception traceback.
                raise RuntimeError("Speech synthesis stream interrupted") from None
            await self._failure(
                504 if isinstance(exc, TimeoutError) else 502,
                "Speech synthesis failed",
                "speech_synthesis_failed",
            )(scope, receive, send)


def _loggable_visitor(visitor_key: str | None) -> str | None:
    """The visitor key as the usage log records it.

    A caller IP is cut to its network (IPv4 /24, IPv6 /48), so the log never
    holds a full client address; any other key is logged as given.
    """
    if visitor_key is None:
        return None
    try:
        address = ipaddress.ip_address(visitor_key)
    except ValueError:
        return visitor_key
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    prefix = 24 if address.version == 4 else 48
    return str(ipaddress.ip_network(f"{address}/{prefix}", strict=False))


def _log_tts_usage(
    audience: Audience,
    user_id: int,
    entity_key: str | None,
    visitor_key: str | None,
    model_id: str,
    chars: int,
    result: str,
) -> None:
    """One line per synthesis request, written when it ends; always on.

    A caller IP is logged as its network only."""
    logger.info(
        "Speech synthesis usage: audience=%s user_id=%s entity=%s visitor=%s "
        "model=%s chars=%d result=%s",
        audience,
        user_id,
        entity_key,
        _loggable_visitor(visitor_key),
        model_id,
        chars,
        result,
    )


async def tts_stream_response(
    *, plan: SpeechStreamPlan[TTSStreamRequest]
) -> StreamingResponse:
    """Build the streaming response for one synthesis request.

    The caller must have authenticated the request and loaded ``plan.config``.
    In order, this asks the usage gate (``StandaloneUsageRefused``, HTTP 402),
    builds the provider instance, checks that it streams, reserves the daily
    voice budget for a guest plan (429 ``voice_budget_exhausted``, or 503
    ``voice_unavailable`` when the budget storage fails), and returns a
    response that meters ``plan.user_id`` while it is sent. Failures before
    the response exists raise ``HTTPException``, with a fixed guest body for
    a guest plan; the instance is closed on every such path, and every
    request writes one usage log line.
    """
    guest = plan.audience == "guest"
    model_id = str(plan.config.model_id)

    def refused(result: str) -> None:
        _log_tts_usage(
            plan.audience,
            plan.user_id,
            plan.entity_key,
            plan.visitor_key,
            model_id,
            len(plan.options.text),
            result,
        )

    # The metering scope opens once the response is being sent, too late for a
    # status code; refuse here, before the provider is reached. The scope
    # checks again, for a gate that turns in between.
    try:
        await check_standalone_gate_async(
            plan.user_id, plan.usage_source, model_ids=(plan.config.model_id,)
        )
    except StandaloneUsageRefused:
        refused("usage_refused")
        if guest:
            raise guest_refusal(402, STANDALONE_USAGE_REFUSED) from None
        raise
    try:
        model = get_tts_model_instance(plan.config)
    except ValueError as exc:
        _log_provider_error(plan.config, exc)
        refused("model_unavailable")
        if guest:
            raise guest_refusal(400, "model_unavailable") from None
        raise HTTPException(
            status_code=400, detail="Invalid TTS model configuration"
        ) from exc
    try:
        capabilities = await model.get_streaming_capabilities()
    except BaseException as exc:
        await _close_model(model)
        if isinstance(exc, Exception):
            _log_provider_error(model, exc)
            refused("provider_failed")
            status = 504 if isinstance(exc, TimeoutError) else 502
            if guest:
                raise guest_refusal(status, "speech_synthesis_failed") from None
            raise HTTPException(
                status_code=status, detail="Speech capabilities unavailable"
            ) from None
        raise
    if not capabilities.get("streaming"):
        await _close_model(model)
        refused("model_unavailable")
        if guest:
            raise guest_refusal(400, "model_unavailable")
        raise HTTPException(
            status_code=400, detail="Selected TTS model does not support streaming"
        )
    if guest:
        assert plan.entity_key and plan.visitor_key
        try:
            await reserve_tts(plan.entity_key, plan.visitor_key, len(plan.options.text))
        except BaseException as exc:
            await _close_model(model)
            if isinstance(exc, VoiceBudgetExhausted):
                refused("budget_exhausted")
                raise guest_refusal(429, "voice_budget_exhausted") from None
            if isinstance(exc, VoiceBudgetUnavailable):
                refused("budget_unavailable")
                raise guest_refusal(503, "voice_unavailable") from None
            raise
    return _TTSAudioResponse(
        model,
        plan.user_id,
        plan.options,
        usage_source=plan.usage_source,
        audience=plan.audience,
        entity_key=plan.entity_key,
        visitor_key=plan.visitor_key,
    )
