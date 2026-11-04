"""Wall-clock deadlines for waiting on a provider stream (#3056).

A first-token or token-interval limit checked only when a chunk arrives
cannot bound a stream that never delivers one: the check never runs, and the
only remaining bound is the transport's read timeout, which a provider that
keeps the connection alive (SSE comment keepalives are skipped by the SDK's
decoder) never trips. The limit has to sit on the wait itself.

``XinferenceLLM`` bounds its stream with a similar private helper
(``_await_before_deadline``). That one also relabels a transport timeout as
``LLMTimeoutError``; this one leaves it to the caller's own handling.
"""

import asyncio
import inspect
import logging
from typing import Awaitable, TypeVar

from .exceptions import LLMTimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def await_before_deadline(
    awaitable: Awaitable[T], *, deadline: float, timeout_message: str
) -> T:
    """Await ``awaitable``, raising :class:`LLMTimeoutError` at ``deadline``.

    ``deadline`` is an absolute ``loop.time()`` value, so one deadline can
    span several waits (opening a stream, then its first chunk).

    On expiry the wait is cancelled, which lets an SDK stream iterator run its
    own cleanup; closing the stream object stays the caller's job. A
    ``TimeoutError`` raised by the awaitable itself (a transport timeout) is
    not relabelled: only this deadline produces ``timeout_message``.
    """
    if deadline <= asyncio.get_running_loop().time():
        # Do not start a request (or a read) that is already out of time.
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        else:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
        logger.error(timeout_message)
        raise LLMTimeoutError(timeout_message)

    scope = asyncio.timeout_at(deadline)
    try:
        async with scope:
            return await awaitable
    except TimeoutError:
        if not scope.expired():
            raise
        logger.error(timeout_message)
        raise LLMTimeoutError(timeout_message) from None
