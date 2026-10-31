"""Live ASR protocol, cancellation, and owned-transport coverage."""

import asyncio
import base64
import json
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

from xagent.core.model.asr import ASRStreamEvent, ElevenLabsASR, XinferenceASR
from xagent.core.model.asr import elevenlabs_stream as stream_module
from xagent.core.utils.security import PrivateNetworkHostError


def message(data):
    return aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(data), "")


class Socket:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.messages.put_nowait(
            message(
                {
                    "message_type": "session_started",
                    "session_id": "session-1",
                }
            )
        )
        self.waiting = asyncio.Event()
        self.sent = []
        self.close_count = 0

    async def receive(self):
        if self.messages.empty():
            self.waiting.set()
        return await self.messages.get()

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.close_count += 1
        self.messages.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, None, ""))


@pytest.fixture
def transport(monkeypatch):
    socket = Socket()
    client = SimpleNamespace(
        ws_connect=AsyncMock(return_value=socket),
        close=AsyncMock(),
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr(stream_module, "create_public_model_aiohttp_session", factory)
    usage = Mock()
    monkeypatch.setattr(stream_module, "record_provider_usage", usage)
    model = ElevenLabsASR(
        model="scribe_v2_realtime", api_key="test-secret", language="en"
    )
    model.model_id = "configured-asr"
    return SimpleNamespace(
        socket=socket, client=client, factory=factory, model=model, usage=usage
    )


async def test_vad_stream_sends_pcm_and_commits_distinct_segments(transport):
    t = transport
    async with await t.model.open_stream(
        vad_silence_threshold_secs=0.8, vad_threshold=0.4
    ) as stream:
        events = aiter(stream)
        assert await anext(events) == ASRStreamEvent(
            type="session_started", session_id="session-1"
        )
        audio = b"\x00\x00" * 16000
        await stream.send_audio(audio)
        sent = t.socket.sent[0]
        assert base64.b64decode(sent["audio_base_64"]) == audio
        assert (sent["commit"], sent["sample_rate"]) == (False, 16000)
        params = t.client.ws_connect.call_args.kwargs["params"]
        assert params == {
            "model_id": "scribe_v2_realtime",
            "audio_format": "pcm_16000",
            "language_code": "en",
            "commit_strategy": "vad",
            "include_timestamps": "false",
            "vad_silence_threshold_secs": "0.8",
            "vad_threshold": "0.4",
        }
        for kind, text in [
            ("partial_transcript", "Hello"),
            ("committed_transcript", "Hello world"),
            ("committed_transcript", "Hello world"),
        ]:
            t.socket.messages.put_nowait(message({"message_type": kind, "text": text}))
        updates = [await anext(events) for _ in range(3)]
        assert [(event.type, event.text, event.segment_id) for event in updates] == [
            ("partial", "Hello", 0),
            ("final", "Hello world", 0),
            ("final", "Hello world", 1),
        ]
        with pytest.raises(ValueError, match="server VAD"):
            await stream.commit()
    assert t.socket.close_count == 1
    t.client.close.assert_awaited_once()
    assert not t.model._streams
    assert t.usage.call_args.args == (t.model, stream_module.MediaCallType.ASR, 1.0)
    await stream.aclose()
    t.usage.assert_called_once()


async def test_manual_commit_and_cancel_do_not_submit_extra_audio(transport):
    t = transport
    stream = await t.model.open_stream(vad=False, format="ulaw", sample_rate=8000)
    await stream.send_audio(b"\xff" * 8000)
    await stream.commit()
    assert t.socket.sent[-1] == {
        "message_type": "input_audio_chunk",
        "audio_base_64": "",
        "commit": True,
        "sample_rate": 8000,
    }
    await stream.cancel()
    assert len(t.socket.sent) == 2
    with pytest.raises(StopAsyncIteration):
        await anext(aiter(stream))
    with pytest.raises(RuntimeError, match="closed"):
        await stream.send_audio(b"\xff")
    assert t.usage.call_args.args[2] == 1.0
    t.usage.assert_called_once()


@pytest.mark.parametrize(
    "options",
    [
        {"sample_rate": 12345},
        {"format": "mp3"},
        {"format": "ulaw", "sample_rate": 16000},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"vad_threshold": 2},
        {"vad_silence_threshold_secs": 4},
        {"vad": False, "vad_threshold": 0.4},
    ],
)
async def test_invalid_audio_or_vad_options_fail_before_connect(transport, options):
    with pytest.raises(ValueError):
        await transport.model.open_stream(**options)
    transport.factory.assert_not_called()


async def test_batch_and_realtime_models_use_their_own_endpoints(
    transport, monkeypatch
):
    batch = ElevenLabsASR(api_key="test-secret")
    assert not batch.supports_streaming
    assert not batch.supports_vad
    with pytest.raises(ValueError, match="scribe_v2_realtime"):
        await batch.open_stream()
    monkeypatch.setattr(
        transport.model, "_ensure_async_client", Mock(side_effect=AssertionError)
    )
    with pytest.raises(ValueError, match="open_stream"):
        await transport.model.transcribe(b"audio")
    assert transport.model.supports_streaming
    assert transport.model.supports_vad
    assert transport.model.abilities == ["asr"]
    local = XinferenceASR()
    assert not local.supports_streaming
    with pytest.raises(NotImplementedError):
        await local.open_stream()


@pytest.mark.parametrize(
    "data",
    [
        {"message_type": "rate_limited", "error": "upstream rejected test-secret"},
        {"message_type": "committed_transcript", "text": 42},
        {"message_type": []},
        [],
    ],
)
async def test_provider_errors_and_malformed_events_close_connection(transport, data):
    t = transport
    stream = await t.model.open_stream()
    events = aiter(stream)
    await anext(events)
    t.socket.messages.put_nowait(message(data))
    with pytest.raises(RuntimeError) as error:
        await anext(events)
    assert "test-secret" not in str(error.value)
    assert t.socket.close_count == 1
    t.client.close.assert_awaited_once()
    assert not t.model._streams


async def test_cancelling_pending_receive_closes_and_meters_once(transport):
    t = transport
    stream = await t.model.open_stream()
    events = aiter(stream)
    await anext(events)
    await stream.send_audio(b"\x00\x00" * 8000)
    pending = asyncio.create_task(anext(events))
    await t.socket.waiting.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert t.socket.close_count == 1
    t.client.close.assert_awaited_once()
    assert t.usage.call_args.args[2] == 0.5
    await t.model.aclose()
    t.usage.assert_called_once()


async def test_cancelling_pending_send_releases_transport(transport):
    t = transport
    sending = asyncio.Event()

    async def blocked_send(payload):
        sending.set()
        await asyncio.Event().wait()

    t.socket.send_json = blocked_send
    stream = await t.model.open_stream()
    pending = asyncio.create_task(stream.send_audio(b"\x00\x00"))
    await sending.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert t.socket.close_count == 1
    t.client.close.assert_awaited_once()
    assert not t.model._streams


async def test_startup_timeout_closes_transport(transport):
    t = transport
    t.socket.messages.get_nowait()
    with pytest.raises(TimeoutError):
        await t.model.open_stream(timeout=0.01)
    assert t.socket.close_count == 1
    t.client.close.assert_awaited_once()


async def test_model_close_discards_session_that_is_still_opening(transport):
    t = transport
    started = t.socket.messages.get_nowait()
    pending = asyncio.create_task(t.model.open_stream())
    await t.socket.waiting.wait()
    await t.model.aclose()
    t.socket.messages.put_nowait(started)
    with pytest.raises(RuntimeError, match="closed while opening"):
        await pending
    assert t.socket.close_count == 1
    t.client.close.assert_awaited_once()


async def test_usage_keeps_the_context_that_opened_the_session(transport):
    owner = ContextVar("speech_usage_owner", default="none")
    seen = []
    transport.usage.side_effect = lambda *args, **kwargs: seen.append(owner.get())
    token = owner.set("original")
    try:
        stream = await transport.model.open_stream()
        await stream.send_audio(b"\x00\x00")
        owner.set("different")
        await stream.aclose()
        assert seen == ["original"]
    finally:
        owner.reset(token)


async def test_real_asr_upgrade_keeps_model_destination_policy(monkeypatch):
    original_factory = stream_module.create_public_model_aiohttp_session
    clients = []

    def factory(**kwargs):
        client = original_factory(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(stream_module, "create_public_model_aiohttp_session", factory)
    model = ElevenLabsASR(
        model="scribe_v2_realtime",
        api_key="test-secret",
        base_url="http://169.254.169.254",
    )
    with pytest.raises(PrivateNetworkHostError):
        await model.open_stream()
    assert clients[0].closed
    assert not model._streams
