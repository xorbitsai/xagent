"""Audio must reach the consumer before completion and release on interruption."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from elevenlabs.client import AsyncElevenLabs

from xagent.core.model import media_metering
from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.tts import ElevenLabsTTS, TTSChunk, XinferenceTTS
from xagent.core.model.tts import elevenlabs as tts_module


@pytest.fixture
def provider(monkeypatch):
    model = ElevenLabsTTS(
        model="eleven_flash_v2_5",
        api_key="test-secret",
        voice="configured-voice",
        language="en",
        format="pcm",
        sample_rate=16000,
    )
    model.model_id = "configured-tts"
    usage = Mock()
    monkeypatch.setattr(tts_module, "record_provider_usage", usage)
    return SimpleNamespace(model=model, usage=usage)


def install_stream(monkeypatch, provider, audio, *, error=None):
    closed = asyncio.Event()
    requests = []

    @asynccontextmanager
    async def stream(**kwargs):
        requests.append(kwargs)
        try:
            if error:
                raise error
            yield SimpleNamespace(headers={"character-cost": "7"}, data=audio)
        finally:
            closed.set()

    client = SimpleNamespace(
        text_to_speech=SimpleNamespace(
            with_raw_response=SimpleNamespace(stream=stream),
        )
    )
    monkeypatch.setattr(provider.model, "_ensure_async_client", lambda: client)
    return requests, closed


async def test_first_audio_is_yielded_before_provider_finishes_and_close_releases(
    monkeypatch, provider
):
    gate = asyncio.Event()

    async def audio():
        yield b"first"
        await gate.wait()
        yield b"second"

    requests, closed = install_stream(monkeypatch, provider, audio())
    stream = provider.model.synthesize_stream(
        "UN", pronunciation_aliases={"UN": "United Nations"}
    )
    assert await anext(stream) == TTSChunk(
        audio=b"first", format="pcm", sample_rate=16000
    )
    assert not gate.is_set()
    assert not closed.is_set()
    assert requests[0]["text"] == "United Nations"
    assert requests[0]["output_format"] == "pcm_16000"
    assert requests[0]["voice_id"] == "configured-voice"
    assert requests[0]["language_code"] == "en"
    assert requests[0]["model_id"] == "eleven_flash_v2_5"
    assert provider.usage.call_args.args[2] == len("United Nations")
    assert provider.usage.call_args.kwargs["provider_billed_units"] == "7"
    await stream.aclose()
    assert closed.is_set()
    provider.usage.assert_called_once()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_cancel_while_waiting_for_first_audio_closes_and_bills_acceptance(
    monkeypatch, provider
):
    waiting = asyncio.Event()

    async def audio():
        waiting.set()
        await asyncio.Event().wait()
        yield b"unreachable"

    _, closed = install_stream(monkeypatch, provider, audio())
    stream = provider.model.synthesize_stream("Hello")
    pending = asyncio.create_task(anext(stream))
    await waiting.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert closed.is_set()
    provider.usage.assert_called_once()
    assert provider.usage.call_args.args[2] == 5


async def test_complete_multichunk_stream_uses_cloned_voice_and_bills_once(
    monkeypatch, provider
):
    async def audio():
        yield b"first"
        yield b"second"

    requests, closed = install_stream(monkeypatch, provider, audio())
    clone = AsyncMock(return_value="cloned-voice")
    monkeypatch.setattr(provider.model, "_get_or_create_cloned_voice", clone)
    chunks = [
        chunk.audio
        async for chunk in provider.model.synthesize_stream(
            "Hello", reference_audio="reference.wav"
        )
    ]
    assert chunks == [b"first", b"second"]
    clone.assert_awaited_once_with("reference.wav")
    assert requests[0]["voice_id"] == "cloned-voice"
    assert "reference_audio" not in requests[0]
    assert closed.is_set()
    provider.usage.assert_called_once()


@pytest.mark.parametrize("finish", ["iterate", "close"])
async def test_stream_usage_remains_in_the_originating_context(
    monkeypatch, provider, finish
):
    async def audio():
        yield b"first"
        yield b"second"

    install_stream(monkeypatch, provider, audio())
    monkeypatch.setattr(
        tts_module, "record_provider_usage", media_metering.record_provider_usage
    )
    with TokenContextManager() as opening:
        stream = provider.model.synthesize_stream("Hello")
        await anext(stream)
        with TokenContextManager() as consuming:
            if finish == "iterate":
                assert [chunk.audio async for chunk in stream] == [b"second"]
            else:
                await stream.aclose()
    rows = [row for row in opening.get_usage().details if row.get("type") == "media"]
    assert len(rows) == 1
    assert rows[0]["quantity"] == 5
    assert not consuming.get_usage().details


async def test_stream_invalid_voice_uses_batch_guidance_and_logs_safely(
    monkeypatch, provider, caplog
):
    install_stream(
        monkeypatch, provider, None, error=RuntimeError("invalid_uid test-secret")
    )
    with pytest.raises(RuntimeError, match="list_tts_voices") as error:
        await anext(provider.model.synthesize_stream("Hello"))
    assert "test-secret" not in str(error.value)
    assert "test-secret" not in caplog.text
    assert "provided voice ID" in caplog.text


@pytest.mark.parametrize("chunks", [[], [b""], ["invalid audio"]])
async def test_empty_or_malformed_accepted_audio_is_metered_and_closed(
    monkeypatch, provider, chunks
):
    async def audio():
        for chunk in chunks:
            yield chunk

    _, closed = install_stream(monkeypatch, provider, audio())
    with pytest.raises(RuntimeError):
        async for _ in provider.model.synthesize_stream("Hello"):
            pytest.fail("Invalid audio reached the consumer")
    assert closed.is_set()
    provider.usage.assert_called_once()


async def test_rejected_stream_does_not_bill_and_redacts_credentials(
    monkeypatch, provider
):
    _, closed = install_stream(
        monkeypatch, provider, None, error=RuntimeError("rejected test-secret")
    )
    with pytest.raises(RuntimeError) as error:
        await anext(provider.model.synthesize_stream("Hello"))
    assert "test-secret" not in str(error.value)
    assert closed.is_set()
    provider.usage.assert_not_called()


@pytest.mark.parametrize("coroutine_response", [False, True])
async def test_sdk_iterator_compatibility_and_per_call_overrides(
    monkeypatch, provider, coroutine_response
):
    closed = asyncio.Event()
    calls = []

    async def audio():
        try:
            yield b"first"
            await asyncio.Event().wait()
        finally:
            closed.set()

    def stream(**kwargs):
        calls.append(kwargs)
        return audio()

    async def legacy_stream(**kwargs):
        return stream(**kwargs)

    client = SimpleNamespace(
        text_to_speech=SimpleNamespace(
            stream=legacy_stream if coroutine_response else stream,
        )
    )
    monkeypatch.setattr(provider.model, "_ensure_async_client", lambda: client)
    chunks = provider.model.synthesize_stream(
        "Hello",
        voice="override-voice",
        language="fr",
        format="pcm",
        sample_rate=24000,
    )
    assert (await anext(chunks)).sample_rate == 24000
    assert calls[0]["voice_id"] == "override-voice"
    assert calls[0]["language_code"] == "fr"
    await chunks.aclose()
    assert closed.is_set()
    provider.usage.assert_called_once()


async def test_stream_reader_does_not_accumulate_audio():
    produced = []
    closed = asyncio.Event()

    async def audio(**kwargs):
        try:
            for number in range(1000):
                produced.append(number)
                yield b"audio"
        finally:
            closed.set()

    state = media_metering.BilledStream()
    stream = media_metering.iter_billed_stream(
        SimpleNamespace(stream=audio),
        "stream",
        {},
        bytes,
        state,
    )
    assert await anext(stream) == b"audio"
    assert produced == [0]
    assert state.accepted
    assert state.chunks == []
    await stream.aclose()
    assert closed.is_set()


def test_only_implemented_adapters_advertise_streaming(provider):
    assert provider.model.supports_streaming
    local = XinferenceTTS()
    assert not local.supports_streaming
    with pytest.raises(NotImplementedError):
        local.synthesize_stream("Hello")


async def test_real_sdk_stream_yields_before_response_completion(monkeypatch, provider):
    released = asyncio.Event()
    gate = asyncio.Event()
    requests = []

    class Audio(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"\x00\x00" * 2048
            await gate.wait()
            yield b"later"

        async def aclose(self):
            released.set()

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/pcm", "character-cost": "5"},
            stream=Audio(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        sdk = AsyncElevenLabs(api_key="test-secret", httpx_client=http)
        monkeypatch.setattr(provider.model, "_ensure_async_client", lambda: sdk)
        stream = provider.model.synthesize_stream("Hello")
        chunk = await asyncio.wait_for(anext(stream), 1)
        assert chunk.audio
        assert not gate.is_set()
        assert requests[0].url.path.endswith("/text-to-speech/configured-voice/stream")
        await stream.aclose()
        assert released.is_set()
    provider.usage.assert_called_once()
