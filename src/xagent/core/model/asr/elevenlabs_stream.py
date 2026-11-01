"""Live ElevenLabs transcription over the existing guarded model transport."""

from __future__ import annotations

import asyncio
import base64
import json
import math
from collections.abc import AsyncIterator
from contextvars import copy_context
from typing import Any

import aiohttp
from yarl import URL

from ...utils.security import redact_sensitive_text
from ..chat.token_context import MediaCallType
from ..media_metering import record_provider_usage
from ..public_http import create_public_model_aiohttp_session
from .base import ASRStreamEvent, ASRStreamSession

REALTIME_ASR_MODEL = "scribe_v2_realtime"
_PCM_RATES = {8000, 16000, 22050, 24000, 44100, 48000}
_ERROR_EVENTS = {
    "error",
    "auth_error",
    "quota_exceeded",
    "commit_throttled",
    "transcriber_error",
    "unaccepted_terms_error",
    "rate_limited",
    "input_error",
    "queue_overflow",
    "resource_exhausted",
    "session_time_limit_exceeded",
    "chunk_size_exceeded",
    "insufficient_audio_activity",
}


class ElevenLabsASRStream(ASRStreamSession):
    def __init__(
        self,
        provider: Any,
        client: aiohttp.ClientSession,
        socket: aiohttp.ClientWebSocketResponse,
        *,
        sample_rate: int,
        format: str,
        vad: bool,
        timeout: float,
        session_id: str | None,
    ) -> None:
        self._provider = provider
        self._client = client
        self._socket = socket
        self.sample_rate = sample_rate
        self.format = format
        self.vad = vad
        self.session_id = session_id
        self._timeout = timeout
        self._closed = False
        self._send_error: Exception | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._audio_bytes = 0
        self._usage_context = copy_context()
        self._iterator = self._events()

    @classmethod
    async def connect(
        cls,
        provider: Any,
        *,
        language: str | None,
        format: str,
        sample_rate: int,
        vad: bool,
        timeout: float = 30,
        vad_silence_threshold_secs: float | None = None,
        vad_threshold: float | None = None,
    ) -> ElevenLabsASRStream:
        if not provider.api_key:
            raise ValueError("ElevenLabs API key is required for live ASR")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("ASR stream timeout must be positive and finite")
        if not (
            (format == "pcm" and sample_rate in _PCM_RATES)
            or (format == "ulaw" and sample_rate == 8000)
        ):
            raise ValueError(
                "Live ASR requires mono PCM16 at a supported rate or ulaw at 8000 Hz"
            )
        if (
            vad_silence_threshold_secs is not None
            and not 0.3 <= vad_silence_threshold_secs <= 3
        ):
            raise ValueError("VAD silence threshold must be between 0.3 and 3 seconds")
        if vad_threshold is not None and not 0.1 <= vad_threshold <= 0.9:
            raise ValueError("VAD threshold must be between 0.1 and 0.9")
        if not vad and (
            vad_silence_threshold_secs is not None or vad_threshold is not None
        ):
            raise ValueError("VAD settings require vad=True")

        params: dict[str, str] = {
            "model_id": provider.model,
            "audio_format": f"{format}_{sample_rate}",
            "commit_strategy": "vad" if vad else "manual",
            "include_timestamps": "false",
        }
        if language:
            params["language_code"] = language
        if vad_silence_threshold_secs is not None:
            params["vad_silence_threshold_secs"] = str(vad_silence_threshold_secs)
        if vad_threshold is not None:
            params["vad_threshold"] = str(vad_threshold)
        # aiohttp accepts HTTP(S) URLs for the WebSocket upgrade. Keeping that
        # spelling lets the shared origin, DNS, redirect, and proxy policy apply.
        url = str(URL(provider.base_url) / "v1" / "speech-to-text" / "realtime")
        client = create_public_model_aiohttp_session(
            timeout=aiohttp.ClientTimeout(total=None, connect=timeout)
        )
        socket = None
        try:
            socket = await asyncio.wait_for(
                client.ws_connect(
                    url,
                    params=params,
                    headers={"xi-api-key": provider.api_key},
                    heartbeat=20,
                    timeout=aiohttp.ClientWSTimeout(ws_close=1),
                ),
                timeout,
            )
            message = await asyncio.wait_for(socket.receive(), timeout)
            data = cls._message_data(message, provider.api_key)
            if data.get("message_type") != "session_started":
                raise RuntimeError("ElevenLabs did not start the ASR session")
            session_id = data.get("session_id")
            return cls(
                provider,
                client,
                socket,
                sample_rate=sample_rate,
                format=format,
                vad=vad,
                timeout=timeout,
                session_id=session_id if isinstance(session_id, str) else None,
            )
        except BaseException:
            try:
                if socket is not None:
                    await socket.close()
            finally:
                await client.close()
            raise

    @staticmethod
    def _message_data(message: aiohttp.WSMessage, api_key: str) -> dict[str, Any]:
        if message.type != aiohttp.WSMsgType.TEXT:
            raise RuntimeError("ElevenLabs ASR connection closed unexpectedly")
        try:
            data = json.loads(message.data)
        except (ValueError, TypeError):
            raise RuntimeError("Invalid ElevenLabs ASR event") from None
        if not isinstance(data, dict):
            raise RuntimeError("Invalid ElevenLabs ASR event")
        if not isinstance(data.get("message_type"), str):
            raise RuntimeError("Invalid ElevenLabs ASR event")
        if data.get("message_type") in _ERROR_EVENTS or data.get("error"):
            detail = str(data.get("error") or data.get("message_type"))
            detail = redact_sensitive_text(detail.replace(api_key, "[REDACTED]"))
            raise RuntimeError(f"ElevenLabs ASR stream failed: {detail}")
        return data

    async def _send(self, payload: dict[str, Any], audio_bytes: int = 0) -> None:
        try:
            async with self._send_lock:
                if self._closed:
                    raise RuntimeError("ASR stream is closed")
                try:
                    await asyncio.wait_for(
                        self._socket.send_json(payload), self._timeout
                    )
                except Exception as exc:
                    if not self._closed:
                        self._send_error = exc
                    raise
                self._audio_bytes += audio_bytes
        except BaseException:
            await self.aclose()
            raise

    async def send_audio(self, audio: bytes) -> None:
        """Send one raw audio fragment, applying transport backpressure."""
        if not isinstance(audio, bytes) or (self.format == "pcm" and len(audio) % 2):
            raise ValueError("Audio must be bytes aligned to mono PCM16 samples")
        if not audio:
            if self._closed:
                raise RuntimeError("ASR stream is closed")
            return
        await self._send(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(audio).decode("ascii"),
                "commit": False,
                "sample_rate": self.sample_rate,
            },
            len(audio),
        )

    async def commit(self) -> None:
        if self.vad:
            raise ValueError("The server VAD owns commits in this session")
        await self._send(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": "",
                "commit": True,
                "sample_rate": self.sample_rate,
            }
        )

    def __aiter__(self) -> AsyncIterator[ASRStreamEvent]:
        return self._iterator

    async def _events(self) -> AsyncIterator[ASRStreamEvent]:
        segment_id = 0
        try:
            if self._closed:
                if self._send_error is not None:
                    raise self._send_error
                return
            yield ASRStreamEvent(type="session_started", session_id=self.session_id)
            while not self._closed:
                message = await self._socket.receive()
                if self._closed:
                    break
                data = self._message_data(message, self._provider.api_key)
                message_type = data.get("message_type")
                if message_type not in {"partial_transcript", "committed_transcript"}:
                    continue
                text = data.get("text")
                if not isinstance(text, str):
                    raise RuntimeError("Invalid ElevenLabs ASR transcript")
                final = message_type == "committed_transcript"
                yield ASRStreamEvent(
                    type="final" if final else "partial",
                    text=text,
                    segment_id=segment_id,
                    session_id=self.session_id,
                )
                if final:
                    segment_id += 1
            if self._send_error is not None:
                raise self._send_error
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        try:
            try:
                await self._socket.close()
            finally:
                await self._client.close()
        finally:
            # Wait for any bounded in-flight send before measuring its input.
            async with self._send_lock:
                if self._audio_bytes:
                    width = 2 if self.format == "pcm" else 1
                    self._usage_context.run(
                        record_provider_usage,
                        self._provider,
                        MediaCallType.ASR,
                        self._audio_bytes / (self.sample_rate * width),
                    )
            self._provider._streams.discard(self)
