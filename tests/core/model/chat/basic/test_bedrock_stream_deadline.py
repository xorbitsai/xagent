"""Native Bedrock stream limits are deadlines, and expiry frees the reader (#3059).

``BedrockLLM`` reads boto3's synchronous event stream one event per worker
thread hop. It used to apply neither ``first_token_timeout`` nor
``token_interval_timeout``: the only bound was botocore's per-read socket
timeout.
"""

import asyncio
import logging
import socket
import threading
import time

import pytest
import urllib3
from botocore.eventstream import EventStream

from xagent.core.model.chat.basic.bedrock import BedrockLLM, _shutdown_event_stream
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
# For tests about what happens after the first event: the worker thread has to
# start and deliver it first, which must not race the first-token limit.
AFTER_FIRST_EVENT_CONFIG = TimeoutConfig(
    first_token_timeout=1.0,
    token_interval_timeout=TOKEN_INTERVAL_TIMEOUT,
)


class FakeRawStream:
    """Stands in for the urllib3 response botocore's ``EventStream`` wraps.

    Without ``supports_shutdown`` it has no ``shutdown()``, as urllib3 < 2.3.
    """

    def __init__(self, wake, *, supports_shutdown=True):
        self._wake = wake
        self.shut_down = False
        self.closed = False
        if not supports_shutdown:
            self.shutdown = None

    def shutdown(self):
        self.shut_down = True
        self._wake.set()

    def close(self):
        self.closed = True


class FakeEventStream:
    """Shaped like botocore's ``EventStream``: iterable, ``close()``, ``_raw_stream``.

    ``script`` is a list of ``(delay_seconds, event)``. After it runs out the
    stream ends, after ``end_delay`` seconds of silence, or with ``then_block``
    stays silent. A read blocked in a wait returns early when the raw stream is
    shut down, as a socket read does, and raises like urllib3 then does. With
    ``close_blocks``, ``close()`` waits as urllib3's does behind a blocked read.
    ``close_raises`` makes ``close()`` fail the way urllib3's does when the
    woken reader's cleanup races it. ``close_thread`` records which thread ran
    ``close()``. ``release`` ends every wait at teardown so no thread is left
    behind.
    """

    def __init__(
        self,
        script,
        *,
        then_block=False,
        end_delay=0.0,
        supports_shutdown=True,
        close_blocks=False,
        close_raises=False,
    ):
        self._script = list(script)
        self._then_block = then_block
        self._end_delay = end_delay
        self._close_blocks = close_blocks
        self._close_raises = close_raises
        self.release = threading.Event()
        self._raw_stream = FakeRawStream(
            self.release, supports_shutdown=supports_shutdown
        )
        self.pulled = 0
        self.reader_woken = threading.Event()
        self.close_called = threading.Event()
        self.close_thread = None

    def _wait(self, timeout=None):
        self.release.wait(timeout)
        if self._raw_stream.shut_down:
            self.reader_woken.set()
            raise ConnectionError("socket shut down")

    def __iter__(self):
        for delay, event in self._script:
            if delay:
                self._wait(delay)
            self.pulled += 1
            yield event
        if self._then_block:
            self._wait()
        elif self._end_delay:
            self._wait(self._end_delay)

    def close(self):
        self.close_thread = threading.current_thread().name
        self.close_called.set()
        if self._close_blocks:
            self.release.wait()
        if self._close_raises:
            raise AttributeError("'NoneType' object has no attribute 'close'")
        self._raw_stream.close()


class FakeClient:
    def __init__(self, stream, *, hang_on_open=False):
        self._stream = stream
        self._hang_on_open = hang_on_open

    def converse_stream(self, **_request):
        if self._hang_on_open:
            self._stream.release.wait()
        return {"stream": self._stream}


def _text(text):
    return {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}


@pytest.fixture
async def make_llm():
    # Async so it is torn down before the runner's teardown
    # (``asyncio.Runner.close()`` -> ``loop.shutdown_default_executor()``, which
    # pytest-asyncio uses) joins the default executor's threads: a worker still
    # blocked in a fake read would hang that join until released here.
    streams = []

    def _make(
        script,
        *,
        then_block=False,
        end_delay=0.0,
        hang_on_open=False,
        supports_shutdown=True,
        close_blocks=False,
        close_raises=False,
        timeouts=TIMEOUT_CONFIG,
    ):
        stream = FakeEventStream(
            script,
            then_block=then_block,
            end_delay=end_delay,
            supports_shutdown=supports_shutdown,
            close_blocks=close_blocks,
            close_raises=close_raises,
        )
        streams.append(stream)
        llm = BedrockLLM(
            "anthropic.claude-test",
            region_name="us-east-1",
            client=FakeClient(stream, hang_on_open=hang_on_open),
            timeout_config=timeouts,
        )
        return llm, stream

    yield _make
    for stream in streams:
        stream.release.set()


async def _drain(llm, received):
    async with asyncio.timeout(HANG_GUARD):
        async for chunk in llm.stream_chat([{"role": "user", "content": "Hi"}]):
            received.append(chunk)


def _token_deltas(chunks):
    return [chunk.delta for chunk in chunks if chunk.is_token()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_kwargs",
    [
        pytest.param({"script": [], "then_block": True}, id="silent"),
        pytest.param({"script": [(1.5, _text("late"))]}, id="late_first_event"),
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
            id="late_second_event",
        ),
        # A stream that went silent past the limit and then ended used to be
        # accepted; the wait for the end is now bounded like any other wait.
        pytest.param(
            {"script": [(0.0, _text("hello"))], "end_delay": 1.5},
            id="silence_then_clean_end",
        ),
    ],
)
async def test_interval_deadline(make_llm, llm_kwargs):
    llm, _ = make_llm(**llm_kwargs, timeouts=AFTER_FIRST_EVENT_CONFIG)
    loop = asyncio.get_running_loop()
    received = []

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
        await _drain(llm, received)

    assert _token_deltas(received) == ["hello"]
    assert loop.time() - started < TOKEN_INTERVAL_TIMEOUT + MARGIN


@pytest.mark.asyncio
async def test_events_inside_each_limit_are_all_delivered(make_llm):
    # The first event arrives at once and each later gap is well inside the
    # interval limit, while the whole stream (~1.2 s) runs past the 1 s
    # first-token limit: neither limit is a cap on total duration.
    texts = ["a", "b", "c", "d", "e"]
    llm, stream = make_llm(
        [(0.0, _text(texts[0]))] + [(0.3, _text(t)) for t in texts[1:]],
        timeouts=TimeoutConfig(first_token_timeout=1.0, token_interval_timeout=1.0),
    )
    received = []

    await _drain(llm, received)

    assert _token_deltas(received) == texts
    assert not stream._raw_stream.shut_down


@pytest.mark.asyncio
async def test_consumer_time_between_events_is_not_provider_silence(make_llm):
    # The interval limit bounds how long the provider stays silent while we
    # wait; time the caller spends on a chunk before asking for the next one
    # is not provider silence.
    texts = ["a", "b", "c"]
    llm, _ = make_llm(
        [(0.0, _text(t)) for t in texts], timeouts=AFTER_FIRST_EVENT_CONFIG
    )
    received = []

    async with asyncio.timeout(HANG_GUARD):
        async for chunk in llm.stream_chat([{"role": "user", "content": "Hi"}]):
            received.append(chunk)
            await asyncio.sleep(TOKEN_INTERVAL_TIMEOUT * 1.5)

    assert _token_deltas(received) == texts


@pytest.mark.asyncio
async def test_expiry_shuts_the_socket_down_and_frees_the_reader(make_llm):
    llm, stream = make_llm(
        [(0.0, _text("hello"))], then_block=True, timeouts=AFTER_FIRST_EVENT_CONFIG
    )

    with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
        await _drain(llm, [])

    assert stream._raw_stream.shut_down
    assert stream._raw_stream.closed
    # The worker blocked in the read returned at once, not at botocore's read
    # timeout or at teardown.
    assert await asyncio.to_thread(stream.reader_woken.wait, HANG_GUARD)


@pytest.mark.asyncio
async def test_expiry_without_shutdown_does_not_wait_for_close(make_llm):
    # urllib3 < 2.3 has no shutdown(), and close() then waits for the blocked
    # read. The timeout must still reach the caller at the deadline, with the
    # close left to finish in the background.
    llm, stream = make_llm(
        [(0.0, _text("hello"))],
        then_block=True,
        supports_shutdown=False,
        close_blocks=True,
        timeouts=AFTER_FIRST_EVENT_CONFIG,
    )
    loop = asyncio.get_running_loop()

    started = loop.time()
    with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
        await _drain(llm, [])

    assert loop.time() - started < TOKEN_INTERVAL_TIMEOUT + MARGIN
    assert await asyncio.to_thread(stream.close_called.wait, HANG_GUARD)


@pytest.mark.asyncio
async def test_close_error_after_shutdown_does_not_replace_the_timeout(make_llm):
    llm, stream = make_llm(
        [(0.0, _text("hello"))],
        then_block=True,
        close_raises=True,
        timeouts=AFTER_FIRST_EVENT_CONFIG,
    )

    with pytest.raises(LLMTimeoutError, match="Token interval timeout"):
        await _drain(llm, [])

    assert stream._raw_stream.shut_down


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_shutdown", [True, False])
async def test_caller_closing_the_stream_between_events_closes_it_inline(
    make_llm, supports_shutdown
):
    # No read is pending when the caller stops at a yield, so there is nothing
    # to wake: the stream is closed before aclose() returns, without shutting
    # the socket down and without falling back to the background close.
    llm, stream = make_llm(
        [(0.0, _text("a")), (0.0, _text("b"))],
        supports_shutdown=supports_shutdown,
        timeouts=AFTER_FIRST_EVENT_CONFIG,
    )

    async with asyncio.timeout(HANG_GUARD):
        stream_iter = llm.stream_chat([{"role": "user", "content": "Hi"}])
        first = await anext(stream_iter)
        await stream_iter.aclose()

    assert first.delta == "a"
    assert stream._raw_stream.closed
    assert not stream._raw_stream.shut_down
    assert stream.close_thread != "bedrock-stream-close"


@pytest.mark.asyncio
async def test_cancelling_the_caller_frees_the_blocked_reader(make_llm):
    # A user stop cancels the consuming task while a worker is blocked in the
    # read; the socket is shut down on that path too, not only on a deadline.
    llm, stream = make_llm(
        [(0.0, _text("hello"))],
        then_block=True,
        timeouts=TimeoutConfig(first_token_timeout=30.0, token_interval_timeout=30.0),
    )
    received = []
    first_chunk = asyncio.Event()

    async def consume():
        async for chunk in llm.stream_chat([{"role": "user", "content": "Hi"}]):
            received.append(chunk)
            first_chunk.set()

    task = asyncio.create_task(consume())
    async with asyncio.timeout(HANG_GUARD):
        await first_chunk.wait()
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stream._raw_stream.shut_down
    assert stream._raw_stream.closed
    assert await asyncio.to_thread(stream.reader_woken.wait, HANG_GUARD)


@pytest.mark.asyncio
async def test_first_token_timeout_is_logged_once_at_error_level(make_llm, caplog):
    llm, _ = make_llm([], then_block=True)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(LLMTimeoutError):
            await _drain(llm, [])

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, [r.getMessage() for r in errors]


def test_shutdown_wakes_a_botocore_event_stream_read_blocked_in_another_thread():
    # The adapter relies on botocore's EventStream wrapping the urllib3
    # response as ``_raw_stream``, and on urllib3's shutdown() waking a socket
    # read blocked in another thread. Both are pinned here, through the
    # production helper, against a real socket.
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    hold = threading.Event()
    response = None
    try:

        def serve():
            conn, _ = server.accept()
            conn.recv(65536)
            # Headers only: the body never arrives.
            conn.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            hold.wait(HANG_GUARD)
            conn.close()

        threading.Thread(target=serve, daemon=True).start()
        pool = urllib3.PoolManager(timeout=urllib3.Timeout(connect=5, read=30))
        port = server.getsockname()[1]
        response = pool.request(
            "GET", f"http://127.0.0.1:{port}/", preload_content=False
        )
        event_stream = EventStream(response, None, None, "ConverseStream")
        returned = threading.Event()

        def read():
            try:
                for _ in event_stream:
                    pass
            except Exception:
                pass
            returned.set()

        threading.Thread(target=read, daemon=True).start()
        time.sleep(0.2)

        assert _shutdown_event_stream(event_stream)
        assert returned.wait(HANG_GUARD)
    finally:
        hold.set()
        if response is not None:
            response.close()
        server.close()
