"""Zhipu stream limits are deadlines, and the producer thread stops (#3058).

``ZhipuLLM`` reads the synchronous zai SDK stream in a producer thread and
consumes it through a queue. The limits used to be checked only when a chunk
came off the queue, and the producer kept reading the whole stream after the
consumer had gone. The Claude and Gemini paths are covered by
``test_sdk_stream_deadline.py``.
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from xagent.core.model.chat.basic.zhipu import ZhipuLLM
from xagent.core.model.chat.exceptions import LLMTimeoutError
from xagent.core.model.chat.timeout_config import TimeoutConfig

FIRST_TOKEN_TIMEOUT = 0.2
TOKEN_INTERVAL_TIMEOUT = 0.2
# Scheduling margin for a loaded CI runner. FIRST_TOKEN_TIMEOUT + MARGIN stays
# well below the 1.5 s late chunk, so "at the deadline, not on arrival" holds.
MARGIN = 0.6
# Outer bound so a regression fails the test instead of hanging the suite.
HANG_GUARD = 5.0

TIMEOUT_CONFIG = TimeoutConfig(
    first_token_timeout=FIRST_TOKEN_TIMEOUT,
    token_interval_timeout=TOKEN_INTERVAL_TIMEOUT,
)
# For tests about what happens after the first chunk: the producer thread has
# to start and deliver it first, which must not race the first-token limit.
AFTER_FIRST_CHUNK_CONFIG = TimeoutConfig(
    first_token_timeout=1.0,
    token_interval_timeout=TOKEN_INTERVAL_TIMEOUT,
)


class FakeResponse:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeZhipuStream:
    """Shaped like zai's ``StreamResponse``: a sync iterator with ``.response``.

    ``script`` is a list of ``(delay_seconds, chunk)``; after it runs out the
    stream either ends or, with ``then_block``, stays silent until released.
    ``release`` ends every wait early, so teardown never leaves a thread behind.
    """

    def __init__(self, script, *, then_block=False, end_delay=0.0, fail=None):
        self._script = list(script)
        self._then_block = then_block
        self._end_delay = end_delay
        self._fail = fail
        self.response = FakeResponse()
        self.release = threading.Event()
        self.pulled = 0

    def __iter__(self):
        for delay, chunk in self._script:
            if delay:
                self.release.wait(delay)
            self.pulled += 1
            yield chunk
        if self._fail is not None:
            raise self._fail
        if self._then_block:
            self.release.wait()
        elif self._end_delay:
            self.release.wait(self._end_delay)


def _text(text):
    delta = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason=None)
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.fixture
async def make_llm():
    # Async so it is torn down before the runner's teardown
    # (``asyncio.Runner.close()`` -> ``loop.shutdown_default_executor()``, which
    # pytest-asyncio uses) joins the default executor's threads: a producer
    # still blocked in a fake read would hang that join until released here.
    streams = []

    def _make(
        script,
        *,
        then_block=False,
        end_delay=0.0,
        fail=None,
        hang_on_open=False,
        timeouts=None,
        creates=None,
    ):
        stream = FakeZhipuStream(
            script, then_block=then_block, end_delay=end_delay, fail=fail
        )
        streams.append(stream)

        def create(**kwargs):
            if creates is not None:
                creates.append(kwargs)
            if hang_on_open:
                stream.release.wait()
            return stream

        llm = ZhipuLLM(api_key="test-key", timeout_config=timeouts or TIMEOUT_CONFIG)
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return llm, stream

    yield _make
    for stream in streams:
        stream.release.set()


async def _drain(llm, received):
    async with asyncio.timeout(HANG_GUARD):
        async for chunk in llm.stream_chat([{"role": "user", "content": "Hi"}]):
            received.append(chunk)


async def _wait_until(predicate):
    async with asyncio.timeout(HANG_GUARD):
        while not predicate():
            await asyncio.sleep(0.01)


def _token_deltas(chunks):
    return [chunk.delta for chunk in chunks if chunk.is_token()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_kwargs",
    [
        pytest.param({"script": [], "then_block": True}, id="silent"),
        pytest.param({"script": [(1.5, _text("late"))]}, id="late_first_chunk"),
        pytest.param(
            {"script": [(0.0, _text("never"))], "hang_on_open": True},
            id="hangs_on_open",
        ),
    ],
)
async def test_first_token_deadline(make_llm, llm_kwargs):
    llm, _ = make_llm(**llm_kwargs)
    loop = asyncio.get_running_loop()

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])

    assert loop.time() - started < FIRST_TOKEN_TIMEOUT + MARGIN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_kwargs",
    [
        pytest.param(
            {"script": [(0.0, _text("hello"))], "then_block": True}, id="stall"
        ),
        pytest.param(
            {"script": [(0.0, _text("hello")), (1.5, _text("late"))]},
            id="late_second_chunk",
        ),
        # Before #3058 the loop exited on end-of-stream without checking, so a
        # stream that went silent past the limit and then ended was accepted.
        # The wait for the end is now bounded like any other wait.
        pytest.param(
            {"script": [(0.0, _text("hello"))], "end_delay": 1.5},
            id="silence_then_clean_end",
        ),
    ],
)
async def test_interval_deadline(make_llm, llm_kwargs):
    llm, _ = make_llm(**llm_kwargs, timeouts=AFTER_FIRST_CHUNK_CONFIG)
    loop = asyncio.get_running_loop()
    received = []

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
        await _drain(llm, received)

    assert _token_deltas(received) == ["hello"]
    assert loop.time() - started < TOKEN_INTERVAL_TIMEOUT + MARGIN


@pytest.mark.asyncio
async def test_chunks_inside_each_limit_are_all_delivered(make_llm):
    # The first chunk arrives at once and each later gap is well inside the
    # interval limit, while the whole stream (~1.2 s) runs past the 1 s
    # first-token limit: neither limit is a cap on total duration. Both limits
    # are wide because the producer thread has to start before the first chunk.
    texts = ["a", "b", "c", "d", "e"]
    llm, stream = make_llm(
        [(0.0, _text(texts[0]))] + [(0.3, _text(t)) for t in texts[1:]],
        timeouts=TimeoutConfig(first_token_timeout=1.0, token_interval_timeout=1.0),
    )
    received = []

    await _drain(llm, received)

    assert _token_deltas(received) == texts
    assert stream.pulled == len(texts)


@pytest.mark.asyncio
async def test_consumer_time_between_chunks_is_not_provider_silence(make_llm):
    # The interval limit bounds how long the provider stays silent while we
    # wait; time the caller spends on a chunk before asking for the next one
    # is not provider silence.
    texts = ["a", "b", "c"]
    llm, _ = make_llm(
        [(0.0, _text(t)) for t in texts], timeouts=AFTER_FIRST_CHUNK_CONFIG
    )
    received = []

    async with asyncio.timeout(HANG_GUARD):
        async for chunk in llm.stream_chat([{"role": "user", "content": "Hi"}]):
            received.append(chunk)
            await asyncio.sleep(TOKEN_INTERVAL_TIMEOUT * 1.5)

    assert _token_deltas(received) == texts


@pytest.mark.asyncio
async def test_producer_stops_and_closes_response_after_a_timeout(make_llm):
    # The read the producer is blocked in cannot be interrupted, but the next
    # chunk it receives makes it stop reading and close the response.
    script = [(1.5, _text("late")), (0.0, _text("more")), (0.0, _text("rest"))]
    llm, stream = make_llm(script)

    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])

    await _wait_until(lambda: stream.response.closed)
    assert stream.pulled == 1


@pytest.mark.asyncio
async def test_producer_stops_when_the_stream_opens_after_a_timeout(make_llm):
    llm, stream = make_llm([(0.0, _text("unread"))], hang_on_open=True)

    with pytest.raises(LLMTimeoutError, match="First token timeout"):
        await _drain(llm, [])
    stream.release.set()

    await _wait_until(lambda: stream.response.closed)
    assert stream.pulled == 0


@pytest.mark.asyncio
async def test_producer_started_after_a_timeout_sends_no_request(make_llm):
    creates = []
    llm, _ = make_llm([(0.0, _text("unread"))], creates=creates)
    loop = asyncio.get_running_loop()
    # One worker, kept busy, so the producer only starts after the deadline.
    # The executor is not restored: each test has its own loop, and the
    # runner's teardown joins whatever default executor is installed.
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    gate = threading.Event()
    try:
        blocker = loop.run_in_executor(None, gate.wait)
        with pytest.raises(LLMTimeoutError, match="First token timeout"):
            await _drain(llm, [])
        gate.set()
        await blocker
        # FIFO on the single worker: this runs after the queued producer returned.
        await loop.run_in_executor(None, lambda: None)
    finally:
        gate.set()

    assert creates == []


@pytest.mark.asyncio
async def test_caller_abandoning_the_stream_stops_the_producer(make_llm):
    script = [(0.0, _text("a")), (1.0, _text("b"))] + [
        (0.0, _text(t)) for t in ["c", "d", "e"]
    ]
    llm, stream = make_llm(script, timeouts=AFTER_FIRST_CHUNK_CONFIG)

    stream_iter = llm.stream_chat([{"role": "user", "content": "Hi"}])
    async with asyncio.timeout(HANG_GUARD):
        first = await anext(stream_iter)
        await stream_iter.aclose()

    assert first.delta == "a"
    await _wait_until(lambda: stream.response.closed)
    assert stream.pulled == 2


@pytest.mark.asyncio
async def test_producer_closes_response_when_the_stream_fails(make_llm):
    llm, stream = make_llm(
        [(0.0, _text("hello"))],
        fail=RuntimeError("cut off"),
        timeouts=AFTER_FIRST_CHUNK_CONFIG,
    )
    received = []

    await _drain(llm, received)

    assert received[-1].is_error()
    # The producer queues the error before it closes the response, so the
    # close can land after the consumer is done.
    await _wait_until(lambda: stream.response.closed)


@pytest.mark.asyncio
async def test_first_token_timeout_is_logged_once_at_error_level(make_llm, caplog):
    llm, _ = make_llm([], then_block=True)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(LLMTimeoutError):
            await _drain(llm, [])

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, [r.getMessage() for r in errors]
