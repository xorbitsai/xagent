"""Real HTTP/WebSocket contracts between Xagent and Xinference audio adapters."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web

from xagent.core.model.asr.xinference import XinferenceASR
from xagent.core.model.tts.xinference import XinferenceTTS


@pytest.fixture
async def service(monkeypatch):
    state = SimpleNamespace(
        audio=[],
        speech=[],
        cancelled=0,
        probes=0,
        caps_status=200,
        caps_body=None,
        outputs=None,
        expected_language="English",
        release=asyncio.Event(),
        caps={
            "asr": {"streaming": True, "server_vad": True},
            "tts": {
                "streaming": True,
                "formats": ["pcm"],
                "sample_rates": [24000],
                "voices": [{"voice_id": "ryan", "name": "Ryan"}],
            },
        },
    )

    async def capabilities(request):
        state.probes += 1
        if state.caps_status != 200 or state.caps_body is not None:
            return web.Response(status=state.caps_status, body=state.caps_body)
        return web.json_response(state.caps)

    async def realtime(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        start = await ws.receive_json()
        assert start["task"] == "audio.transcription"
        assert start["parameters"]["language"] == "en"
        await ws.send_json({"event": "session.started", "session_id": "session-1"})
        async for message in ws:
            if message.type == web.WSMsgType.BINARY:
                state.audio.append(message.data)
                if state.outputs is not None:
                    for output in state.outputs:
                        await ws.send_json(output)
                    continue
                for seq, data in enumerate(
                    [
                        {"sentences": [], "partial": "Hello"},
                        {"sentences": [{"text": "Hello."}], "partial": ""},
                        {"sentences": [{"text": "Hello."}], "partial": ""},
                        {
                            "sentences": [{"text": "Hello."}, {"text": "Bye."}],
                            "partial": "",
                        },
                    ],
                    1,
                ):
                    await ws.send_json(
                        {
                            "event": "output",
                            "session_id": "session-1",
                            "seq": seq,
                            "data": data,
                        }
                    )
            elif message.type == web.WSMsgType.TEXT:
                assert json.loads(message.data) == {"event": "session.cancel"}
                state.cancelled += 1
                await ws.close()
        return ws

    async def speech(request):
        body = await request.json()
        state.speech.append(body)
        expected = (
            {"language": state.expected_language} if state.expected_language else {}
        )
        assert json.loads(body.get("kwargs", "{}")) == expected
        response = web.StreamResponse(
            headers={"Content-Type": "application/octet-stream"}
        )
        await response.prepare(request)
        await response.write(b"\x01\x00\x02\x00")
        await state.release.wait()
        try:
            await response.write(b"\x03\x00")
            await response.write_eof()
        except ConnectionResetError:
            pass
        return response

    app = web.Application()
    app.router.add_get("/v1/models/{uid}/audio/capabilities", capabilities)
    app.router.add_get("/v1/models/{uid}/realtime", realtime)
    app.router.add_post("/v1/audio/speech", speech)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    state.url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_ENDPOINTS", state.url)
    monkeypatch.setenv("XAGENT_MODEL_PRIVATE_CIDRS", "127.0.0.1/32")
    try:
        yield state
    finally:
        state.release.set()
        await runner.cleanup()


async def test_asr_cumulative_snapshots_become_individual_commits(service, monkeypatch):
    charges = []
    monkeypatch.setattr(
        "xagent.core.model.asr.xinference_stream.record_provider_usage",
        lambda *args: charges.append(args[-1]),
    )
    provider = XinferenceASR(model_uid="arbitrary-uid", base_url=service.url)
    assert not provider.supports_streaming
    stream = await provider.open_stream()
    assert provider.supports_streaming and provider.supports_vad
    async with stream:
        events = aiter(stream)
        assert (await anext(events)).type == "session_started"
        await stream.send_audio(b"\x01\x00" * 512)
        results = [await asyncio.wait_for(anext(events), 1) for _ in range(3)]
        assert [(r.type, r.text, r.segment_id) for r in results] == [
            ("partial", "Hello", 0),
            ("final", "Hello.", 0),
            ("final", "Bye.", 1),
        ]
        with pytest.raises(ValueError, match="VAD"):
            await stream.commit()
    await provider.aclose()
    assert charges == [512 / 16000]
    assert service.cancelled == 1 and not provider._streams


async def test_asr_unsupported_engine_and_audio_options(service):
    provider = XinferenceASR(base_url=service.url)
    for options in ({"vad": False}, {"sample_rate": 48000}, {"vad_threshold": 0.8}):
        with pytest.raises(ValueError):
            await provider.open_stream(**options)
    service.caps = {}
    with pytest.raises(ValueError, match="does not support"):
        await provider.open_stream()
    assert not provider.supports_streaming
    await provider.aclose()


async def test_tts_streams_first_audio_and_closes_on_interruption(service, monkeypatch):
    charges = []
    monkeypatch.setattr(
        "xagent.core.model.tts.xinference.record_provider_usage",
        lambda *args: charges.append(args[-1]),
    )
    provider = XinferenceTTS(
        model_uid="arbitrary-uid", base_url=service.url, format="pcm", language="en"
    )
    assert await provider.list_available_voices() == [
        {"voice_id": "ryan", "name": "Ryan"}
    ]
    stream = provider.synthesize_stream("Hello.", voice="ryan")
    first = await asyncio.wait_for(anext(stream), 1)
    assert (
        first.audio == b"\x01\x00\x02\x00"
        and first.sample_rate == 24000
        and first.format == "pcm"
    )
    assert not service.release.is_set() and provider._stream_clients
    await stream.aclose()
    assert not provider._stream_clients
    assert charges == [6]
    assert service.speech[0]["model"] == "arbitrary-uid"
    await provider.aclose()


async def test_tts_rejects_mislabeled_formats_before_request(service):
    provider = XinferenceTTS(base_url=service.url, format="pcm")
    for options in ({"format": "mp3"}, {"sample_rate": 48000}):
        stream = provider.synthesize_stream("Hello", **options)
        with pytest.raises(ValueError, match="native format"):
            await anext(stream)
    assert not service.speech
    await provider.aclose()


@pytest.mark.parametrize(
    "language,expected",
    [
        ("en", "English"),
        ("en-US", "English"),
        ("zh", "Chinese"),
        ("zh_CN", "Chinese"),
        ("ja", "Japanese"),
        ("ko", "Korean"),
        ("de", "German"),
        ("fr", "French"),
        ("ru", "Russian"),
        ("pt", "Portuguese"),
        ("es", "Spanish"),
        ("it", "Italian"),
        ("Chinese", "Chinese"),
        ("auto", None),
        ("unknown", None),
    ],
)
async def test_tts_normalizes_saved_languages(service, language, expected):
    service.expected_language = expected
    service.release.set()
    provider = XinferenceTTS(base_url=service.url, format="pcm", language=language)
    assert [chunk async for chunk in provider.synthesize_stream("Hello")]
    await provider.aclose()


async def test_capability_probe_reused_within_each_provider_instance(service):
    provider = XinferenceASR(base_url=service.url)
    await provider.get_streaming_capabilities()
    stream = await provider.open_stream()
    assert service.probes == 1
    await stream.aclose()
    await provider.aclose()
    speaker = XinferenceTTS(base_url=service.url, language="en", format="pcm")
    assert not speaker.supports_voice_listing
    await speaker.get_streaming_capabilities()
    assert await speaker.list_available_voices()
    assert speaker.supports_voice_listing
    service.release.set()
    assert [chunk async for chunk in speaker.synthesize_stream("Hello")]
    assert service.probes == 2
    await speaker.aclose()


@pytest.mark.parametrize("status", [404, 405])
async def test_missing_capabilities_disable_streaming_and_voice_listing(
    service, status
):
    service.caps_status = status
    provider = XinferenceTTS(base_url=service.url)
    assert await provider.get_streaming_capabilities() == {}
    assert await provider.list_available_voices() == []
    assert not provider.supports_streaming and not provider.supports_voice_listing
    assert service.probes == 1
    await provider.aclose()


@pytest.mark.parametrize("body", [b"{", b"\xff", b"[]", b'{"asr":[]}', b" " * 65537])
async def test_malformed_or_oversized_capabilities_are_upstream_errors(service, body):
    service.caps_body = body
    provider = XinferenceTTS(base_url=service.url)
    with pytest.raises(RuntimeError, match="capabilities"):
        await provider.get_streaming_capabilities()
    assert provider._stream_capabilities is None
    await provider.aclose()


async def test_failed_capability_probe_is_not_cached(service):
    provider = XinferenceASR(base_url=service.url)
    service.caps_status = 503
    with pytest.raises(aiohttp.ClientResponseError):
        await provider.get_streaming_capabilities()
    service.caps_status = 200
    assert (await provider.get_streaming_capabilities())["streaming"]
    assert service.probes == 2
    await provider.aclose()


@pytest.mark.parametrize(
    "invalid",
    [
        {"event": "error", "message": "provider-secret"},
        {"event": "output", "seq": 0, "session_id": "session-1"},
        {"event": "output", "seq": True, "session_id": "session-1"},
        {"event": "output", "seq": 2, "session_id": "other"},
        {"event": "output", "seq": 1, "session_id": "session-1"},
    ],
)
async def test_asr_rejects_upstream_error_or_invalid_sequence_and_session(
    service, invalid
):
    data = {"sentences": [], "partial": "Hello"}
    service.outputs = [
        {"event": "output", "seq": 1, "session_id": "session-1", "data": data},
        {**invalid, "data": data},
    ]
    provider = XinferenceASR(base_url=service.url)
    stream = await provider.open_stream()
    await stream.send_audio(b"\x00\x00")
    with pytest.raises(RuntimeError) as exc:
        async for _ in stream:
            pass
    assert "provider-secret" not in str(exc.value)
    assert not provider._streams and service.cancelled == 1
    await provider.aclose()


@pytest.mark.parametrize("phase", ["connect", "start", "send", "cancel"])
async def test_asr_transport_deadlines_close_owned_resources(monkeypatch, phase):
    from xagent.core.model.asr import xinference_stream as module

    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    socket = SimpleNamespace(
        closed=False,
        close=AsyncMock(),
        send_json=AsyncMock(),
        send_bytes=AsyncMock(),
        receive=AsyncMock(
            return_value=aiohttp.WSMessage(
                aiohttp.WSMsgType.TEXT,
                json.dumps({"event": "session.started", "session_id": "session-1"}),
                "",
            )
        ),
    )
    client = SimpleNamespace(
        close=AsyncMock(), ws_connect=AsyncMock(return_value=socket)
    )
    monkeypatch.setattr(
        module, "create_public_model_aiohttp_session", lambda **_: client
    )
    monkeypatch.setattr(module, "_IO_TIMEOUT", 0.01)
    monkeypatch.setattr(module, "_CLOSE_TIMEOUT", 0.01)
    provider = XinferenceASR()
    if phase == "connect":
        client.ws_connect.side_effect = blocked
    if phase == "start":
        socket.send_json.side_effect = blocked
    if phase in {"connect", "start"}:
        with pytest.raises(TimeoutError):
            await module.XinferenceASRStream.connect(provider, "en")
    else:
        stream = await module.XinferenceASRStream.connect(provider, "en")
        provider._streams.add(stream)
        if phase == "send":
            socket.send_bytes.side_effect = blocked
            with pytest.raises(TimeoutError):
                await stream.send_audio(b"\x00\x00")
        else:
            socket.send_json.side_effect = blocked
            await asyncio.wait_for(stream.aclose(), 0.5)
        assert not provider._streams
    client.close.assert_awaited_once()
    if phase != "connect":
        socket.close.assert_awaited_once()
