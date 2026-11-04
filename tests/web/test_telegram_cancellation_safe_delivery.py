"""Cancellation-safe delivery for Telegram output.

Every send is an awaited round trip, so a /stop, /new, or /switch can land
while one is in flight. Checking only before the call would leave that output
visible in a conversation the user already left, so these tests cover the
success-path race rather than only fallback failures.
"""

from types import SimpleNamespace

import pytest

from xagent.web.channels.telegram.handler import TelegramTraceHandler
from xagent.web.channels.telegram.utils import (
    CancelledDelivery,
    deliver_cancellation_safe,
)


@pytest.mark.asyncio
async def test_late_successful_delivery_is_removed() -> None:
    deleted: list[int] = []
    cancelled = False

    async def send() -> SimpleNamespace:
        nonlocal cancelled
        # The latch flips while the request is in flight.
        cancelled = True
        return SimpleNamespace(message_id=42)

    async def delete(msg: SimpleNamespace) -> None:
        deleted.append(msg.message_id)

    with pytest.raises(CancelledDelivery):
        await deliver_cancellation_safe(
            send,
            is_cancelled=lambda: cancelled,
            delete=delete,
            description="stream message",
        )

    assert deleted == [42]


@pytest.mark.asyncio
async def test_delivery_is_skipped_when_already_cancelled() -> None:
    sent = False

    async def send() -> None:
        nonlocal sent
        sent = True

    result = await deliver_cancellation_safe(
        send,
        is_cancelled=lambda: True,
        description="stream message",
    )

    assert result is None
    assert sent is False


@pytest.mark.asyncio
async def test_uncancelled_delivery_returns_its_result() -> None:
    async def send() -> str:
        return "sent"

    result = await deliver_cancellation_safe(
        send,
        is_cancelled=lambda: False,
        description="stream message",
    )

    assert result == "sent"


@pytest.mark.asyncio
async def test_undeletable_late_delivery_still_stops_the_sequence() -> None:
    """A message past Telegram's delete window cannot be removed, but the rest
    of the output sequence must still be abandoned."""

    cancelled = False

    async def send() -> SimpleNamespace:
        nonlocal cancelled
        cancelled = True
        return SimpleNamespace(message_id=7)

    async def failing_delete(_msg: SimpleNamespace) -> None:
        raise RuntimeError("message can't be deleted")

    with pytest.raises(CancelledDelivery):
        await deliver_cancellation_safe(
            send,
            is_cancelled=lambda: cancelled,
            delete=failing_delete,
            description="stream message",
        )


@pytest.mark.asyncio
async def test_trace_handler_removes_a_message_sent_after_cancellation() -> None:
    handler = TelegramTraceHandler(7, bot=None, chat_id=5, message_id=None)  # type: ignore[arg-type]
    deleted: list[tuple[int, int]] = []

    class _Bot:
        async def send_message(self, **_kwargs: object) -> SimpleNamespace:
            # /switch lands while this request is in flight.
            handler.cancel()
            return SimpleNamespace(message_id=99)

        async def delete_message(self, *, chat_id: int, message_id: int) -> None:
            deleted.append((chat_id, message_id))

    handler.bot = _Bot()  # type: ignore[assignment]

    await handler._update_message("streamed output")

    assert deleted == [(5, 99)]
    # The handler must not adopt a message it just removed.
    assert handler.message_id is None


@pytest.mark.asyncio
async def test_trace_handler_removes_an_edit_completed_after_cancellation() -> None:
    handler = TelegramTraceHandler(7, bot=None, chat_id=5, message_id=31)  # type: ignore[arg-type]
    deleted: list[tuple[int, int]] = []

    class _Bot:
        async def edit_message_text(self, **_kwargs: object) -> None:
            handler.cancel()

        async def delete_message(self, *, chat_id: int, message_id: int) -> None:
            deleted.append((chat_id, message_id))

    handler.bot = _Bot()  # type: ignore[assignment]

    await handler._update_message("streamed output")

    # An edit writes into an existing message, so that message is removed.
    assert deleted == [(5, 31)]


@pytest.mark.asyncio
async def test_stop_keeps_a_message_sent_after_cancellation() -> None:
    """/stop must not delete the streamed message the user is still reading."""

    handler = TelegramTraceHandler(7, bot=None, chat_id=5, message_id=None)  # type: ignore[arg-type]
    deleted: list[tuple[int, int]] = []

    class _Bot:
        async def send_message(self, **_kwargs: object) -> SimpleNamespace:
            # /stop lands while this request is in flight.
            handler.cancel(discard_output=False)
            return SimpleNamespace(message_id=99)

        async def delete_message(self, *, chat_id: int, message_id: int) -> None:
            deleted.append((chat_id, message_id))

    handler.bot = _Bot()  # type: ignore[assignment]

    await handler._update_message("streamed output")

    assert deleted == []
    assert handler.message_id == 99
    # Streaming still stops; only the output is kept.
    assert handler.cancelled is True
    assert handler.discard_output is False


@pytest.mark.asyncio
async def test_stop_keeps_an_edit_completed_after_cancellation() -> None:
    handler = TelegramTraceHandler(7, bot=None, chat_id=5, message_id=31)  # type: ignore[arg-type]
    deleted: list[tuple[int, int]] = []

    class _Bot:
        async def edit_message_text(self, **_kwargs: object) -> None:
            handler.cancel(discard_output=False)

        async def delete_message(self, *, chat_id: int, message_id: int) -> None:
            deleted.append((chat_id, message_id))

    handler.bot = _Bot()  # type: ignore[assignment]

    await handler._update_message("streamed output")

    assert deleted == []


@pytest.mark.asyncio
async def test_skipped_delivery_is_not_recorded_as_the_current_text() -> None:
    """current_text is the dedup key, so it must only record real deliveries.

    Recording text whose send was skipped would make the dedup check suppress a
    later retry of that same text, losing it permanently.
    """

    handler = TelegramTraceHandler(7, bot=None, chat_id=5, message_id=None)  # type: ignore[arg-type]
    deleted: list[int] = []

    class _Bot:
        async def send_message(self, **_kwargs: object) -> SimpleNamespace:
            # /switch lands while this send is in flight, so the primitive
            # compensates by deleting it and raises CancelledDelivery.
            handler.cancel()
            return SimpleNamespace(message_id=42)

        async def delete_message(self, *, chat_id: int, message_id: int) -> None:
            deleted.append(message_id)

    handler.bot = _Bot()  # type: ignore[assignment]

    await handler._update_message("streamed output")

    # The message was removed again, so it was never really delivered.
    assert deleted == [42]
    assert handler.current_text == ""


@pytest.mark.asyncio
async def test_trace_handler_keeps_output_when_not_cancelled() -> None:
    handler = TelegramTraceHandler(7, bot=None, chat_id=5, message_id=None)  # type: ignore[arg-type]
    deleted: list[tuple[int, int]] = []

    class _Bot:
        async def send_message(self, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(message_id=12)

        async def delete_message(self, *, chat_id: int, message_id: int) -> None:
            deleted.append((chat_id, message_id))

    handler.bot = _Bot()  # type: ignore[assignment]

    await handler._update_message("streamed output")

    assert deleted == []
    assert handler.message_id == 12
