"""Real HTTP/WebSocket contracts between Xagent and Xinference audio adapters."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

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
        assert json.loads(body["kwargs"]) == {"language": "English"}
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
