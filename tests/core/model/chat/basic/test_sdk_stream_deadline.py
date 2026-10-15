"""Claude and Gemini stream limits are deadlines, not arrival checks (#3058).

Both adapters used to evaluate the first-token and token-interval limits only
when an event arrived, so a stream that never delivered one was bounded only by
the transport, and a late first event was rejected on arrival instead of at the
deadline. The OpenAI-compatible path is covered by
``test_openai_stream_deadline.py``.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from xagent.core.model.chat.basic.claude import ClaudeLLM
from xagent.core.model.chat.basic.gemini import GeminiLLM
from xagent.core.model.chat.exceptions import LLMTimeoutError
from xagent.core.model.chat.timeout_config import TimeoutConfig

FIRST_TOKEN_TIMEOUT = 0.2
TOKEN_INTERVAL_TIMEOUT = 0.2
# Scheduling margin for a loaded CI runner. FIRST_TOKEN_TIMEOUT + MARGIN stays
# well below the 1.5 s late event, so "at the deadline, not on arrival" holds.
MARGIN = 0.6
# Outer bound so a regression fails the test instead of hanging the suite.
HANG_GUARD = 5.0

TIMEOUT_CONFIG = TimeoutConfig(
    first_token_timeout=FIRST_TOKEN_TIMEOUT,
    token_interval_timeout=TOKEN_INTERVAL_TIMEOUT,
)


class FakeStream:
    """An SDK event stream: ``script`` is a list of ``(delay_seconds, event)``.

    After the script runs out the stream ends, after ``end_delay`` seconds of
    silence, or with ``then_block`` stays silent forever.
    """

    def __init__(self, script, *, then_block=False, end_delay=0.0):
        self._script = list(script)
        self._then_block = then_block
        self._end_delay = end_delay
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._script:
            delay, event = self._script.pop(0)
            await asyncio.sleep(delay)
            return event
        if self._then_block:
            await asyncio.Event().wait()
        await asyncio.sleep(self._end_delay)
        raise StopAsyncIteration


class ClaudeStream(FakeStream):
    """Shaped like ``anthropic.AsyncStream``, which is closed with ``close()``."""

    async def close(self):
        self.closed = True


class RaisingCloseClaudeStream(ClaudeStream):
    """A stream whose ``close()`` fails, as a broken transport can."""

    async def close(self):
        raise RuntimeError("close failed")


class GeminiStream(FakeStream):
    """Shaped like the async generator ``generate_content_stream`` returns."""

    async def aclose(self):
        self.closed = True


class RaisingCloseGeminiStream(GeminiStream):
    """A generator whose ``aclose()`` fails, as a broken transport can."""

    async def aclose(self):
        raise RuntimeError("aclose failed")


async def _hang(*_args, **_kwargs):
    await asyncio.Event().wait()


def _claude_text(text):
    return SimpleNamespace(
        type="content_block_delta",
        index=0,
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _make_claude(mocker, config, script, block, end_delay, close_raises, timeouts):
    stream_cls = RaisingCloseClaudeStream if close_raises else ClaudeStream
    stream = stream_cls(script or [], then_block=block, end_delay=end_delay)
    client = mocker.AsyncMock()
    if script is not None:
        client.messages.create.return_value = stream
    else:
        client.messages.create.side_effect = _hang
    mocker.patch(
        "xagent.core.model.chat.basic.claude.AsyncAnthropic", return_value=client
    )
    return ClaudeLLM(**config, timeout_config=timeouts), stream


def _gemini_text(text):
    part = SimpleNamespace(text=text, function_call=None)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    return SimpleNamespace(candidates=[candidate], usage_metadata=None)


def _make_gemini(mocker, config, script, block, end_delay, close_raises, timeouts):
    stream_cls = RaisingCloseGeminiStream if close_raises else GeminiStream
    stream = stream_cls(script or [], then_block=block, end_delay=end_delay)
    client = mocker.MagicMock()
    if script is not None:

        async def generate_content_stream(**_kwargs):
            return stream

        client.aio.models.generate_content_stream = generate_content_stream
    else:
        client.aio.models.generate_content_stream = _hang
    llm = GeminiLLM(**config, timeout_config=timeouts)
    mocker.patch.object(llm, "_ensure_client")
    llm._client = client
    return llm, stream


ADAPTERS = {
    "claude": (_make_claude, _claude_text),
    "gemini": (_make_gemini, _gemini_text),
}


@pytest.fixture(params=sorted(ADAPTERS))
def adapter(request, mocker):
    make, text_event = ADAPTERS[request.param]
    config = request.getfixturevalue(f"{request.param}_llm_config")

    def _make(
        script=None,
        *,
        then_block=False,
        end_delay=0.0,
        close_raises=False,
        timeout_config=TIMEOUT_CONFIG,
    ):
        """``script=None`` makes opening the stream hang forever."""
        return make(
            mocker,
            config,
            script,
            then_block,
            end_delay,
            close_raises,
            timeout_config,
        )

    return SimpleNamespace(make=_make, text=text_event)


async def _drain(llm, received):
    async with asyncio.timeout(HANG_GUARD):
        async for chunk in llm.stream_chat([{"role": "user", "content": "Hi"}]):
            received.append(chunk)


def _token_deltas(chunks):
    return [chunk.delta for chunk in chunks if chunk.is_token()]


@pytest.mark.asyncio
async def test_silent_stream_times_out_at_first_token_deadline_and_closes(adapter):
    llm, stream = adapter.make([], then_block=True)
    loop = asyncio.get_running_loop()

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])

    assert loop.time() - started < FIRST_TOKEN_TIMEOUT + MARGIN
    assert stream.closed


@pytest.mark.asyncio
async def test_late_first_event_times_out_at_the_deadline_not_on_arrival(adapter):
    llm, stream = adapter.make([(1.5, adapter.text("late"))])
    loop = asyncio.get_running_loop()

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])

    assert loop.time() - started < FIRST_TOKEN_TIMEOUT + MARGIN
    assert stream.closed


@pytest.mark.asyncio
async def test_first_token_deadline_covers_opening_the_stream(adapter):
    llm, _ = adapter.make(None)
    loop = asyncio.get_running_loop()

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])

    assert loop.time() - started < FIRST_TOKEN_TIMEOUT + MARGIN


@pytest.mark.asyncio
async def test_stall_after_first_event_times_out_at_interval_deadline(adapter):
    llm, stream = adapter.make([(0.0, adapter.text("hello"))], then_block=True)
    loop = asyncio.get_running_loop()
    received = []

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
        await _drain(llm, received)

    assert _token_deltas(received) == ["hello"]
    assert loop.time() - started < TOKEN_INTERVAL_TIMEOUT + MARGIN
    assert stream.closed


@pytest.mark.asyncio
async def test_late_second_event_times_out_at_interval_deadline(adapter):
    llm, stream = adapter.make(
        [(0.0, adapter.text("hello")), (1.5, adapter.text("late"))]
    )
    loop = asyncio.get_running_loop()
    received = []

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
        await _drain(llm, received)

    assert _token_deltas(received) == ["hello"]
    assert loop.time() - started < TOKEN_INTERVAL_TIMEOUT + MARGIN
    assert stream.closed


@pytest.mark.asyncio
async def test_silence_past_interval_then_clean_end_times_out(adapter):
    # Before #3058 the loop exited on end-of-stream without checking, so a
    # stream that went silent past the limit and then ended was accepted. The
    # wait for the end is now bounded like any other wait.
    llm, stream = adapter.make([(0.0, adapter.text("hello"))], end_delay=1.5)

    with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
        await _drain(llm, [])

    assert stream.closed


@pytest.mark.asyncio
async def test_events_inside_each_limit_are_all_delivered(adapter):
    # The first event arrives at once and each later gap is well inside the
    # interval limit, while the whole stream (~0.4 s) runs past the 0.2 s
    # first-token limit: neither limit is a cap on total duration.
    texts = ["a", "b", "c", "d", "e"]
    llm, stream = adapter.make(
        [(0.0, adapter.text(texts[0]))] + [(0.1, adapter.text(t)) for t in texts[1:]],
        timeout_config=TimeoutConfig(
            first_token_timeout=FIRST_TOKEN_TIMEOUT,
            token_interval_timeout=1.0,
        ),
    )
    received = []

    await _drain(llm, received)

    assert _token_deltas(received) == texts
    assert not stream.closed


@pytest.mark.asyncio
async def test_consumer_time_between_events_is_not_provider_silence(adapter):
    # The interval limit bounds how long the provider stays silent while we
    # wait; time the caller spends on a chunk before asking for the next one
    # is not provider silence.
    texts = ["a", "b", "c"]
    llm, _ = adapter.make([(0.0, adapter.text(t)) for t in texts])
    received = []

    async with asyncio.timeout(HANG_GUARD):
        async for chunk in llm.stream_chat([{"role": "user", "content": "Hi"}]):
            received.append(chunk)
            await asyncio.sleep(TOKEN_INTERVAL_TIMEOUT * 1.5)

    assert _token_deltas(received) == texts


@pytest.mark.asyncio
async def test_timeout_is_raised_even_when_closing_the_stream_fails(adapter):
    llm, _ = adapter.make([], then_block=True, close_raises=True)

    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])


@pytest.mark.asyncio
async def test_first_token_timeout_is_logged_once_at_error_level(adapter, caplog):
    llm, _ = adapter.make([], then_block=True)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(LLMTimeoutError):
            await _drain(llm, [])

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, [r.getMessage() for r in errors]
