"""Xinference realtime transcription with per-connection ownership."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from contextvars import copy_context
from typing import Any

import aiohttp

from ..chat.token_context import MediaCallType
from ..media_metering import record_provider_usage
from ..public_http import create_public_model_aiohttp_session
from ..xinference_speech import audio_headers, audio_model_url
from .base import ASRStreamEvent, ASRStreamSession

_IO_TIMEOUT = 10.0
_CLOSE_TIMEOUT = 1.0


class XinferenceASRStream(ASRStreamSession):
    def __init__(self, provider: Any, client: Any, socket: Any, session_id: str):
        self._provider, self._client, self._socket = provider, client, socket
        self.session_id = session_id
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._audio_bytes = 0
        self._usage_context = copy_context()
        self._iterator = self._events()

    @staticmethod
    def _data(message: aiohttp.WSMessage) -> dict[str, Any]:
        if message.type != aiohttp.WSMsgType.TEXT:
            raise RuntimeError("Xinference transcription connection ended")
        try:
            data = json.loads(message.data)
        except ValueError as exc:
            raise RuntimeError("Invalid Xinference transcription event") from exc
        if not isinstance(data, dict) or data.get("event") == "error":
            raise RuntimeError("Xinference transcription failed")
        return data

    @classmethod
    async def connect(cls, provider: Any, language: str | None) -> XinferenceASRStream:
        client = create_public_model_aiohttp_session(
            timeout=aiohttp.ClientTimeout(total=None, connect=10)
        )
        socket = None
        try:
            socket = await asyncio.wait_for(
                client.ws_connect(
                    audio_model_url(provider, "realtime"),
                    headers=audio_headers(provider),
                    heartbeat=20,
                    timeout=aiohttp.ClientWSTimeout(ws_close=1),
                    max_msg_size=2 * 1024**2,
                ),
                _IO_TIMEOUT,
            )
            await asyncio.wait_for(
                socket.send_json(
                    {
                        "event": "session.start",
                        "protocol_version": 1,
                        "task": "audio.transcription",
                        "input": {
                            "audio": {
                                "encoding": "pcm_s16le",
                                "sample_rate": 16000,
                                "channels": 1,
                            }
                        },
                        "parameters": {"language": language or "en"},
                    }
                ),
                _IO_TIMEOUT,
            )
            data = cls._data(await asyncio.wait_for(socket.receive(), _IO_TIMEOUT))
            session_id = data.get("session_id")
            if data.get("event") != "session.started" or not isinstance(
                session_id, str
            ):
                raise RuntimeError("Xinference did not start transcription")
            return cls(provider, client, socket, session_id)
        except BaseException:
            try:
                if socket is not None:
                    with suppress(Exception):
                        await asyncio.wait_for(socket.close(), _CLOSE_TIMEOUT)
            finally:
                await client.close()
            raise

    async def send_audio(self, audio: bytes) -> None:
        if not isinstance(audio, bytes) or len(audio) % 2 or len(audio) > 65536:
            raise ValueError("Expected at most 64 KiB of aligned mono PCM16 audio")
        try:
            async with self._send_lock:
                if self._closed:
                    raise RuntimeError("ASR stream is closed")
                await asyncio.wait_for(self._socket.send_bytes(audio), _IO_TIMEOUT)
                self._audio_bytes += len(audio)
        except BaseException:
            await self.aclose()
            raise

    async def commit(self) -> None:
        raise ValueError("Xinference server VAD owns segment commits")

    def __aiter__(self) -> AsyncIterator[ASRStreamEvent]:
        return self._iterator

    async def _events(self) -> AsyncIterator[ASRStreamEvent]:
        sentences: list[str] = []
        partial = ""
        seq = 0
        try:
            if self._closed:
                return
            yield ASRStreamEvent(type="session_started", session_id=self.session_id)
            while not self._closed:
                message = await self._socket.receive()
                if self._closed:
                    return
                event = self._data(message)
                if event.get("event") != "output":
                    continue
                current = event.get("seq")
                if (
                    type(current) is not int
                    or current <= seq
                    or event.get("session_id") != self.session_id
                ):
                    raise RuntimeError("Invalid Xinference transcript sequence")
                seq = current
                data = event.get("data", {})
                if not isinstance(data, dict):
                    raise RuntimeError("Invalid Xinference transcript snapshot")
                committed = data.get("sentences")
                next_partial = data.get("partial")
                if not isinstance(committed, list) or not isinstance(next_partial, str):
                    raise RuntimeError("Invalid Xinference transcript snapshot")
                texts = [
                    item.get("text") if isinstance(item, dict) else None
                    for item in committed
                ]
                if (
                    any(not isinstance(text, str) for text in texts)
                    or texts[: len(sentences)] != sentences
                ):
                    raise RuntimeError("Invalid Xinference committed segments")
                for text in texts[len(sentences) :]:
                    assert isinstance(text, str)
                    index = len(sentences)
                    sentences.append(text)
                    partial = ""
                    yield ASRStreamEvent(
                        type="final",
                        text=text,
                        segment_id=index,
                        session_id=self.session_id,
                    )
                    if self._closed:
                        return
                if next_partial != partial:
                    partial = next_partial
                    yield ASRStreamEvent(
                        type="partial",
                        text=partial,
                        segment_id=len(sentences),
                        session_id=self.session_id,
                    )
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
                if not self._socket.closed:
                    with suppress(Exception):
                        await asyncio.wait_for(
                            self._socket.send_json({"event": "session.cancel"}),
                            _CLOSE_TIMEOUT,
                        )
            finally:
                with suppress(Exception):
                    await asyncio.wait_for(self._socket.close(), _CLOSE_TIMEOUT)
        finally:
            try:
                await self._client.close()
            finally:
                async with self._send_lock:
                    if self._audio_bytes:
                        self._usage_context.run(
                            record_provider_usage,
                            self._provider,
                            MediaCallType.ASR,
                            self._audio_bytes / 32000,
                        )
                self._provider._streams.discard(self)
