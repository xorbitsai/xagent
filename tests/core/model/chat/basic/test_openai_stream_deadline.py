"""First-token and token-interval limits are deadlines, not arrival checks (#3056).

Before #3056 both limits were evaluated only when a chunk arrived, so a stream
that never delivered one was bounded only by the transport, and a late first
chunk was rejected on arrival instead of at the deadline.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from xagent.core.model.chat.basic.openai import OpenAILLM
from xagent.core.model.chat.exceptions import LLMTimeoutError
from xagent.core.model.chat.timeout_config import TimeoutConfig

FIRST_TOKEN_TIMEOUT = 0.2
TOKEN_INTERVAL_TIMEOUT = 0.2
# Scheduling margin for a loaded CI runner. FIRST_TOKEN_TIMEOUT + MARGIN stays
# well below the 1.5 s late chunk, so "at the deadline, not on arrival" holds.
MARGIN = 0.6
# Outer bound so a regression fails the test instead of hanging the suite.
HANG_GUARD = 5.0


def _content_chunk(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=ChoiceDelta(content=text), finish_reason=None)],
        usage=None,
    )


class FakeStream:
    """Shaped like ``openai.AsyncStream``: async-iterable with ``close()``.

    ``script`` is a list of ``(delay_seconds, chunk)``; after it runs out the
    stream either ends or, with ``then_block``, stays silent forever.
    """

    def __init__(self, script, *, then_block=False):
        self._script = script
        self._then_block = then_block
        self.closed = False

    def __aiter__(self):
        return self._chunks()

    async def _chunks(self):
        for delay, chunk in self._script:
            await asyncio.sleep(delay)
            yield chunk
        if self._then_block:
            await asyncio.Event().wait()

    async def close(self):
        self.closed = True


@pytest.fixture
def make_llm(openai_llm_config, mocker):
    def _make(create_side_effect=None, stream=None):
        mock_client = mocker.AsyncMock()
        if create_side_effect is not None:
            mock_client.chat.completions.create.side_effect = create_side_effect
        else:
            mock_client.chat.completions.create.return_value = stream
        mocker.patch(
            "xagent.core.model.chat.basic.openai.AsyncOpenAI",
            return_value=mock_client,
        )
        return OpenAILLM(
            **openai_llm_config,
            timeout_config=TimeoutConfig(
                first_token_timeout=FIRST_TOKEN_TIMEOUT,
                token_interval_timeout=TOKEN_INTERVAL_TIMEOUT,
            ),
        )

    return _make


async def _drain(llm, received):
    async with asyncio.timeout(HANG_GUARD):
        async for chunk in llm.stream_chat([{"role": "user", "content": "Hi"}]):
            received.append(chunk)


@pytest.mark.asyncio
async def test_silent_stream_times_out_at_first_token_deadline_and_closes(make_llm):
    stream = FakeStream([], then_block=True)
    llm = make_llm(stream=stream)
    loop = asyncio.get_running_loop()

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])
    elapsed = loop.time() - started

    assert elapsed < FIRST_TOKEN_TIMEOUT + MARGIN
    assert stream.closed


@pytest.mark.asyncio
async def test_late_first_chunk_times_out_at_the_deadline_not_on_arrival(make_llm):
    late = 1.5
    stream = FakeStream([(late, _content_chunk("late"))])
    llm = make_llm(stream=stream)
    loop = asyncio.get_running_loop()

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])
    elapsed = loop.time() - started

    assert elapsed < FIRST_TOKEN_TIMEOUT + MARGIN
    assert stream.closed


@pytest.mark.asyncio
async def test_first_token_deadline_covers_opening_the_stream(make_llm):
    async def hanging_create(**_kwargs):
        await asyncio.Event().wait()

    llm = make_llm(create_side_effect=hanging_create)
    loop = asyncio.get_running_loop()

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])

    assert loop.time() - started < FIRST_TOKEN_TIMEOUT + MARGIN


@pytest.mark.asyncio
async def test_stall_after_first_chunk_times_out_at_interval_deadline(make_llm):
    stream = FakeStream([(0.0, _content_chunk("hello"))], then_block=True)
    llm = make_llm(stream=stream)
    loop = asyncio.get_running_loop()
    received = []

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
        await _drain(llm, received)
    elapsed = loop.time() - started

    assert [c.content for c in received] == ["hello"]
    assert elapsed < TOKEN_INTERVAL_TIMEOUT + MARGIN
    assert stream.closed


@pytest.mark.asyncio
async def test_chunks_inside_each_limit_are_all_delivered(make_llm):
    # Each gap is inside the interval limit, while the whole stream runs well
    # past the first-token limit: neither limit is a cap on total duration.
    gap = TOKEN_INTERVAL_TIMEOUT / 4
    texts = ["a", "b", "c", "d", "e", "f", "g", "h"]
    stream = FakeStream([(gap, _content_chunk(t)) for t in texts])
    llm = make_llm(stream=stream)
    received = []

    await _drain(llm, received)

    assert [c.content for c in received if c.is_token()] == texts


@pytest.mark.asyncio
async def test_consumer_time_between_chunks_is_not_provider_silence(make_llm):
    # The interval limit bounds how long the provider stays silent while we
    # wait; time the caller spends on a chunk before asking for the next one
    # is not provider silence.
    texts = ["a", "b", "c"]
    stream = FakeStream([(0.0, _content_chunk(t)) for t in texts])
    llm = make_llm(stream=stream)
    received = []

    async for chunk in llm.stream_chat([{"role": "user", "content": "Hi"}]):
        received.append(chunk)
        await asyncio.sleep(TOKEN_INTERVAL_TIMEOUT * 1.5)

    assert [c.content for c in received if c.is_token()] == texts


class KeepaliveOnlyBody(httpx.AsyncByteStream):
    """An SSE body that sends only comment keepalives, never an event.

    Bytes keep arriving, so a transport read timeout never fires, while the
    SDK's SSE decoder drops comment lines and never yields a chunk.
    """

    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        while True:
            yield b": keepalive\n\n"
            await asyncio.sleep(0.02)

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_keepalive_only_sdk_stream_times_out_and_closes_response(
    openai_llm_config, mocker
):
    body = KeepaliveOnlyBody()
    requests_seen = 0

    def handler(request):
        nonlocal requests_seen
        requests_seen += 1
        if requests_seen == 1:
            # Warm-up, sent through a lenient instance: the first request
            # through a fresh SDK client has been seen to take longer than
            # the deadline under test on a loaded runner.
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"data: [DONE]\n\n",
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    sdk_client = AsyncOpenAI(
        api_key="test-api-key",
        base_url="https://provider.test/v1",
        http_client=http_client,
        max_retries=0,
    )
    mocker.patch(
        "xagent.core.model.chat.basic.openai.AsyncOpenAI",
        return_value=sdk_client,
    )
    # Both instances get the patched ``sdk_client``, so the warm-up primes the
    # same client the timed call uses.
    warm_up_llm = OpenAILLM(
        **openai_llm_config,
        timeout_config=TimeoutConfig(
            first_token_timeout=HANG_GUARD,
            token_interval_timeout=HANG_GUARD,
        ),
    )
    llm = OpenAILLM(
        **openai_llm_config,
        timeout_config=TimeoutConfig(
            first_token_timeout=FIRST_TOKEN_TIMEOUT,
            token_interval_timeout=TOKEN_INTERVAL_TIMEOUT,
        ),
    )
    loop = asyncio.get_running_loop()
    try:
        await _drain(warm_up_llm, [])

        started = loop.time()
        with pytest.raises(LLMTimeoutError, match="First token timeout"):
            await _drain(llm, [])

        assert loop.time() - started < FIRST_TOKEN_TIMEOUT + MARGIN
        assert body.closed
    finally:
        await http_client.aclose()
