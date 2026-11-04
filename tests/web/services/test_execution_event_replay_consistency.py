"""Reconnect/replay delivery consistency for V2 task sockets.

Each V2 socket records the display frames it was actually sent after its
replay horizon. On disconnect the frames committed after the horizon are
compared with what was sent, off the event loop. Bridge duplicates and gaps
are counted where the receiver drops or resyncs. No signal line carries
payload content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from tests.web.services.test_execution_event_signals import (
    exported,
    key,
)
from tests.web.services.test_execution_event_signals import (
    metric_reader as metric_reader_fixture,
)
from tests.web.services.test_task_execution_event_writer import (
    canonical as canonical_fixture,
)
from tests.web.services.test_task_execution_event_writer import engine as engine_fixture
from tests.web.services.test_task_execution_event_writer import (
    task_id as task_id_fixture,
)
from xagent.web.api import websocket as websocket_api
from xagent.web.api.websocket import ConnectionManager, _HistoricalStreamSnapshot
from xagent.web.models.task import Task
from xagent.web.services import (
    task_event_bridge,
    task_event_display,
    task_stream_snapshot,
)
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)
from xagent.web.services.task_event_display import (
    convert_display_event,
    expected_display_frames,
    load_event_display_snapshot,
    load_live_display_event,
)
from xagent.web.services.task_execution_event_writer import append_fact_no_commit
from xagent.web.services.task_socket_writer import TaskSocketWriter

canonical = canonical_fixture
engine = engine_fixture
task_id = task_id_fixture
metric_reader = metric_reader_fixture

SENTINEL = "T3A-SENTINEL"
CHECK_TIMEOUT = 10
GRACE = websocket_api._DELIVERY_REORDER_GRACE_SECONDS


class Socket:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.fail_event_ids: dict[str, BaseException] = {}

    async def send_text(self, text: str) -> None:
        frame = json.loads(text)
        error = self.fail_event_ids.get(frame.get("event_id"))
        if error is not None:
            raise error
        self.frames.append(frame)

    async def close(self, code: int = 1000) -> None:
        return None


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(websocket_api, "_delivery_clock", fake)
    return fake


@pytest.fixture
def manager(monkeypatch, clock):
    """A fresh process-wide manager with control-state enrichment disabled."""
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "false")

    async def unchanged(message, **_kwargs):
        return message

    monkeypatch.setattr(websocket_api, "_with_current_task_control_state", unchanged)
    fresh = ConnectionManager()
    monkeypatch.setattr(websocket_api, "manager", fresh)
    return fresh


@pytest.fixture
def shared(monkeypatch):
    monkeypatch.setenv("XAGENT_SHARED_TASK_EXECUTION_ENABLED", "true")
    monkeypatch.setattr(
        task_event_bridge,
        "_bridge",
        SimpleNamespace(discard_recipient=lambda _socket: None),
    )


def fact(db, task_id, kind, key, data, *, flat=False, **kwargs):
    return append_fact_no_commit(
        db,
        task_id=task_id,
        kind=kind,
        key=key,
        payload=data if flat else {"data": data, "protocol_event_id": key},
        occurred_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        **kwargs,
    )


def commit(factory, build):
    """Append the facts ``build(db)`` returns and return the committed rows."""
    with factory() as db:
        rows = build(db)
        db.commit()
        for row in rows:
            db.refresh(row)
        db.expunge_all()
        return rows


def write_history(factory, task_id):
    """Display facts interleaved with facts that never produce a frame.

    Returns the committed rows in sequence order.
    """
    return commit(
        factory,
        lambda db: [
            fact(
                db,
                task_id,
                "input_accepted",
                "turn-1",
                {"content": f"hello {SENTINEL}", "message_type": "user"},
                flat=True,
                turn_id="turn-1",
            ),
            fact(db, task_id, "agent_message", "before", {"message": SENTINEL}),
            # Horizon of the replay below ends here.
            fact(db, task_id, "recovery_state", "state", {"snapshot": SENTINEL}),
            fact(db, task_id, "agent_message", "live-1", {"message": SENTINEL}),
            fact(
                db,
                task_id,
                "agent_message",
                "child",
                {"message": SENTINEL},
                scope_id="child",
            ),
            fact(
                db,
                task_id,
                "llm_call_end",
                "audit",
                {"__audit_only__": True, "response": SENTINEL},
            ),
            fact(db, task_id, "control_state_changed", "ctl", {"state": "x"}),
            fact(db, task_id, "agent_message", "live-2", {"message": SENTINEL}),
            fact(db, task_id, "agent_message", "live-3", {"message": SENTINEL}),
        ],
    )


def by_key(rows):
    return {row.idempotency_key: row for row in rows}


def live_frame(row):
    frame = convert_display_event(row)
    assert frame is not None
    return frame


def replay_snapshot(factory, task_id, horizon):
    with factory() as db:
        events = load_event_display_snapshot(
            db, task_id, through_sequence=horizon
        ).events
    return _HistoricalStreamSnapshot(
        events=tuple(events), horizon=horizon, display_events=len(events)
    )


async def replay(monkeypatch, socket, task_id, snapshot):
    monkeypatch.setattr(
        websocket_api, "_load_historical_stream_snapshot_sync", lambda *a, **k: snapshot
    )
    await websocket_api.send_historical_data_as_stream(
        socket, task_id, SimpleNamespace(id=1, is_admin=True)
    )


async def settle_checks(manager):
    pending = list(manager._delivery_checks)
    if pending:
        await asyncio.wait_for(asyncio.gather(*pending), timeout=CHECK_TIMEOUT)


async def close(manager, socket, clock):
    """Disconnect once every frame sent so far is outside the grace window."""
    clock.advance(GRACE + 1)
    manager.disconnect(socket)
    await settle_checks(manager)


async def drain(manager, socket):
    """Wait until the shared-mode writer sent everything queued before now.

    The writer sends in order and records a frame before acknowledging the
    next one, so the acknowledgement of a marker frame is deterministic on
    every Python version (unlike polling the queue).
    """
    await manager.send_personal_message({"type": "heartbeat"}, socket)


def consistency(reader):
    return exported(reader, "xagent.websocket.replay.consistency")


def outcome(reader, name):
    return consistency(reader) == {key(outcome=name, storage="v2"): 1}


def signal_lines(caplog, prefix="execution_event_"):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


def fields(line: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in line.split()[1:])


def horizon_of(rows):
    return by_key(rows)["state"].sequence


async def connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket):
    manager.register_connection(socket, task_id)
    await replay(
        monkeypatch,
        socket,
        task_id,
        replay_snapshot(factory, task_id, horizon_of(rows)),
    )


async def broadcast_history(manager, rows, task_id):
    for name in ("live-1", "live-2", "live-3"):
        await manager.broadcast_to_task(live_frame(by_key(rows)[name]), task_id)


def display(task_id, sequence, event_id=None):
    return {
        "type": "trace_event",
        "event_id": event_id or f"e{sequence}",
        "task_id": task_id,
        "execution_sequence": sequence,
    }


@pytest.mark.asyncio
async def test_replay_and_live_frames_covering_everything_are_complete(
    canonical, manager, clock, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    rows_by_key = by_key(rows)
    # Alternate the send paths so each one delivers a frame below the
    # highest sequence another path delivered.
    await manager.broadcast_to_task(live_frame(rows_by_key["live-1"]), task_id)
    await manager.send_personal_message(live_frame(rows_by_key["live-2"]), socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-3"]), task_id)
    ledger = manager._delivery_ledgers[socket]
    assert ledger.horizon == horizon_of(rows)
    assert set(ledger.sent) == {
        live_frame(rows_by_key[name])["event_id"]
        for name in ("live-1", "live-2", "live-3")
    }

    caplog.set_level(logging.INFO)
    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")
    assert not signal_lines(caplog, "execution_event_delivery_gap")


@pytest.mark.asyncio
async def test_internal_kinds_and_other_scopes_are_not_missing(
    canonical, manager, clock, monkeypatch, caplog, metric_reader
):
    """Sequence gaps between delivered frames are normal and never missing."""
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-1"]), task_id)
    # Child scope, audit-only and control facts sit between live-1 and live-2.
    await manager.broadcast_to_task(live_frame(rows_by_key["live-2"]), task_id)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-3"]), task_id)
    assert rows_by_key["live-2"].sequence - rows_by_key["live-1"].sequence == 4
    assert rows_by_key["live-1"].sequence - horizon_of(rows) == 1

    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_withheld_frame_is_missing_with_its_sequence(
    canonical, manager, clock, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    other = Socket()
    manager.register_connection(other, task_id)
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-1"]), task_id)
    # A lost send: live-2 reaches the other socket only.
    manager._remove_from_task(socket, task_id)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-2"]), task_id)
    manager.active_connections[task_id].append(socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-3"]), task_id)

    caplog.set_level(logging.INFO)
    await close(manager, socket, clock)

    assert outcome(metric_reader, "missing")
    (line,) = signal_lines(caplog, "execution_event_delivery_gap")
    assert fields(line) == {
        "task_id": str(task_id),
        "storage": "v2",
        "component": "execution-events",
        "horizon": str(horizon_of(rows)),
        "missing": "1",
        "first_missing_sequence": str(rows_by_key["live-2"].sequence),
    }
    assert [r.levelno for r in caplog.records if r.getMessage() == line] == [
        logging.WARNING
    ]
    for line in signal_lines(caplog):
        assert SENTINEL not in line
    assert SENTINEL in json.dumps(socket.frames)


@pytest.mark.asyncio
async def test_frame_reordered_within_the_grace_window_is_not_missing(
    canonical, manager, clock, monkeypatch, metric_reader
):
    """live-3 overtook live-2 just before the disconnect: not a lost frame."""
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-1"]), task_id)
    clock.advance(GRACE * 2)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-3"]), task_id)
    clock.advance(GRACE - 1)
    manager.disconnect(socket)
    await settle_checks(manager)

    # The range ends at live-1, the last frame sent before the window.
    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_frame_withheld_beyond_the_grace_window_is_missing(
    canonical, manager, clock, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-1"]), task_id)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-3"]), task_id)
    clock.advance(GRACE)
    manager.disconnect(socket)
    await settle_checks(manager)

    assert outcome(metric_reader, "missing")


@pytest.mark.asyncio
async def test_rejected_input_without_a_live_frame_is_not_missing(
    canonical, manager, clock, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await broadcast_history(manager, rows, task_id)
    # An input accepted into the log but never injected gets no live frame.
    (_input, after) = commit(
        factory,
        lambda db: [
            fact(
                db,
                task_id,
                "input_accepted",
                "turn-2",
                {"content": "rejected", "message_type": "user"},
                flat=True,
                turn_id="turn-2",
            ),
            fact(db, task_id, "agent_message", "after-input", {"message": "x"}),
        ],
    )
    await manager.broadcast_to_task(live_frame(after), task_id)

    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_cancelled_settlement_message_without_a_broadcast_is_not_missing(
    canonical, manager, clock, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await broadcast_history(manager, rows, task_id)
    # A cancellation settlement commits the assistant row and publishes none.
    (_message, after) = commit(
        factory,
        lambda db: [
            fact(
                db,
                task_id,
                "assistant_message",
                "cancelled",
                {"content": "task execution cancelled", "message_type": "error"},
                flat=True,
            ),
            fact(db, task_id, "agent_message", "after-cancel", {"message": "x"}),
        ],
    )
    await manager.broadcast_to_task(live_frame(after), task_id)

    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_live_frames_during_the_first_replay_load_are_recorded(
    canonical, manager, clock, monkeypatch, metric_reader
):
    """Registered before the replay: overlap below the horizon is benign.

    A frame at or below the horizon sent live while the replay loads is sent
    again by the replay and is not a duplicate; a frame above the horizon sent
    in that window is recorded, so it is not reported missing.
    """
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    manager.register_connection(socket, task_id)
    snapshot = replay_snapshot(factory, task_id, horizon_of(rows))
    loading = threading.Event()
    release = threading.Event()

    def load(*_args, **_kwargs):
        loading.set()
        release.wait(CHECK_TIMEOUT)
        return snapshot

    monkeypatch.setattr(websocket_api, "_load_historical_stream_snapshot_sync", load)
    replaying = asyncio.create_task(
        websocket_api.send_historical_data_as_stream(
            socket, task_id, SimpleNamespace(id=1, is_admin=True)
        )
    )
    await asyncio.to_thread(loading.wait, CHECK_TIMEOUT)
    # Sent live twice while the horizon is unknown: the replay covers it, so
    # it is no duplicate once the horizon prunes it.
    await manager.broadcast_to_task(live_frame(rows_by_key["before"]), task_id)
    await manager.broadcast_to_task(live_frame(rows_by_key["before"]), task_id)
    await manager.broadcast_to_task(live_frame(rows_by_key["live-1"]), task_id)
    release.set()
    await asyncio.wait_for(replaying, timeout=CHECK_TIMEOUT)

    ledger = manager._delivery_ledgers[socket]
    assert ledger.horizon == horizon_of(rows)
    assert set(ledger.sent) == {live_frame(rows_by_key["live-1"])["event_id"]}
    sent_ids = [frame.get("event_id") for frame in socket.frames]
    assert sent_ids.count(live_frame(rows_by_key["before"])["event_id"]) == 3
    assert not ledger.duplicated
    for name in ("live-2", "live-3"):
        await manager.broadcast_to_task(live_frame(rows_by_key[name]), task_id)

    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_live_frame_sent_twice_is_duplicate_without_warning(
    canonical, manager, clock, monkeypatch, caplog, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    for name in ("live-1", "live-1", "live-2", "live-3"):
        await manager.broadcast_to_task(live_frame(rows_by_key[name]), task_id)

    caplog.set_level(logging.INFO)
    await close(manager, socket, clock)

    assert outcome(metric_reader, "duplicate")
    assert not signal_lines(caplog, "execution_event_delivery_gap")


@pytest.mark.asyncio
async def test_live_duplicate_inside_the_grace_window_is_reported(
    canonical, manager, clock, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    frame = live_frame(by_key(rows)["live-1"])
    await manager.broadcast_to_task(frame, task_id)
    await manager.broadcast_to_task(frame, task_id)

    # Disconnect at once: no frame is outside the grace window yet, so there
    # is no range to compare, but the duplicate is still known.
    manager.disconnect(socket)
    await settle_checks(manager)

    assert not manager._delivery_checks
    assert outcome(metric_reader, "duplicate")


class ReplayBlockingSocket(Socket):
    """Holds the first replay frame until released."""

    def __init__(self) -> None:
        super().__init__()
        self.replaying = asyncio.Event()
        self.resume = asyncio.Event()

    async def send_text(self, text: str) -> None:
        if websocket_api._replay_pass.get() and not self.replaying.is_set():
            self.replaying.set()
            await self.resume.wait()
        await super().send_text(text)


@pytest.mark.asyncio
async def test_live_frames_sent_during_the_replay_send_are_live(
    canonical, manager, clock, monkeypatch, metric_reader
):
    """Only the replay coroutine's own sends are replay sends.

    A live frame sent from another coroutine while the replay is sending is
    live, so sending it twice there is a duplicate.
    """
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = ReplayBlockingSocket()
    manager.register_connection(socket, task_id)
    replaying = asyncio.create_task(
        replay(
            monkeypatch,
            socket,
            task_id,
            replay_snapshot(factory, task_id, horizon_of(rows)),
        )
    )
    await asyncio.wait_for(socket.replaying.wait(), timeout=CHECK_TIMEOUT)
    frame = live_frame(by_key(rows)["live-1"])
    await manager.send_personal_message(frame, socket)
    await manager.send_personal_message(frame, socket)
    socket.resume.set()
    await asyncio.wait_for(replaying, timeout=CHECK_TIMEOUT)

    ledger = manager._delivery_ledgers[socket]
    assert ledger.live == {frame["event_id"]}
    assert ledger.duplicated == {frame["event_id"]}


@pytest.mark.asyncio
async def test_repeated_replay_on_request_is_not_a_duplicate(
    canonical, manager, clock, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    for name in ("live-1", "live-2", "live-3"):
        await manager.broadcast_to_task(live_frame(by_key(rows)[name]), task_id)
    # A status request replays the whole history up to a later horizon.
    await replay(
        monkeypatch,
        socket,
        task_id,
        replay_snapshot(factory, task_id, rows[-1].sequence),
    )

    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_frames_missed_live_but_sent_by_a_later_replay_are_complete(
    canonical, manager, clock, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    await replay(
        monkeypatch,
        socket,
        task_id,
        replay_snapshot(factory, task_id, rows[-1].sequence),
    )
    # A late live frame the replay already sent is overlap, not a duplicate.
    await manager.broadcast_to_task(live_frame(by_key(rows)["live-3"]), task_id)

    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_public_audience_projection_is_not_missing(
    canonical, manager, clock, monkeypatch, metric_reader
):
    """Guest sockets receive a projection of every frame, with the same ids.

    Broadcasts filter no V2 frame per recipient; facts that produce no frame
    for anyone (audit-only, internal kinds, child scope) are not expected.
    """
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    manager.register_connection(socket, task_id, public_audience=True)
    await replay(
        monkeypatch,
        socket,
        task_id,
        replay_snapshot(factory, task_id, horizon_of(rows)),
    )
    for name in ("live-1", "live-2", "live-3"):
        await manager.broadcast_to_task(live_frame(by_key(rows)[name]), task_id)

    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_frame_built_by_the_live_path_matches_the_expected_id(
    canonical, manager, clock, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    (tool,) = commit(
        factory,
        lambda db: [
            fact(
                db,
                task_id,
                "tool_execution_start",
                "tool-1",
                {"tool_name": "calculator", "args": {"expression": "1+1"}},
            )
        ],
    )
    with factory() as db:
        frame = load_live_display_event(db, task_id, "tool-1")
        expected = expected_display_frames(
            db,
            task_id,
            after_sequence=horizon_of(rows),
            through_sequence=tool.sequence,
        )
    assert frame is not None
    assert expected[frame["event_id"]] == tool.sequence
    await broadcast_history(manager, rows, task_id)
    await manager.broadcast_to_task(frame, task_id)

    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")


def test_unconvertible_fact_is_left_out_of_the_expected_frames(canonical, monkeypatch):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    convert = task_event_display.convert_display_event

    def flaky(row):
        if row.idempotency_key == "live-2":
            raise TypeError("cannot serialize")
        return convert(row)

    monkeypatch.setattr(task_event_display, "convert_display_event", flaky)
    with factory() as db:
        expected = expected_display_frames(
            db,
            task_id,
            after_sequence=horizon_of(rows),
            through_sequence=rows_by_key["live-3"].sequence,
        )
    assert expected == {
        live_frame(rows_by_key[name])["event_id"]: rows_by_key[name].sequence
        for name in ("live-1", "live-3")
    }


@pytest.mark.asyncio
async def test_shared_mode_records_frames_when_the_writer_sends_them(
    canonical, manager, clock, shared, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    writer = manager._writers[socket]
    rows_by_key = by_key(rows)
    await manager.deliver_shared_event(live_frame(rows_by_key["live-1"]), task_id)
    # Queued is not sent: nothing is recorded until the writer sends.
    ledger = manager._delivery_ledgers[socket]
    assert not ledger.sent
    # A personal message waits for the writer, so live-1 is sent by now.
    await manager.send_personal_message(live_frame(rows_by_key["live-2"]), socket)
    assert set(ledger.sent) == {
        live_frame(rows_by_key[name])["event_id"] for name in ("live-1", "live-2")
    }
    await manager.deliver_shared_event(live_frame(rows_by_key["live-3"]), task_id)
    await drain(manager, socket)
    assert set(ledger.sent) == {
        live_frame(rows_by_key[name])["event_id"]
        for name in ("live-1", "live-2", "live-3")
    }

    await close(manager, socket, clock)
    await asyncio.gather(writer.task, return_exceptions=True)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_reconcile_resends_are_not_duplicates(
    canonical, manager, clock, shared, monkeypatch, metric_reader
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    rows_by_key = by_key(rows)
    socket = Socket()
    await connect_and_replay(manager, monkeypatch, factory, task_id, rows, socket)
    writer = manager._writers[socket]
    settled = live_frame(rows_by_key["live-3"])
    monkeypatch.setattr(
        task_stream_snapshot, "load_task_stream_snapshots", lambda _ids: [settled]
    )
    for name in ("live-1", "live-2", "live-3"):
        await manager.deliver_shared_event(live_frame(rows_by_key[name]), task_id)
    # Every reconcile pass re-sends the settlement frames.
    await manager._reconcile_streams_once()
    await manager._reconcile_streams_once()
    await drain(manager, socket)
    sent_ids = [frame.get("event_id") for frame in socket.frames]
    assert sent_ids.count(settled["event_id"]) == 3

    await close(manager, socket, clock)
    await asyncio.gather(writer.task, return_exceptions=True)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_failing_send_callback_never_stops_the_writer():
    socket = Socket()
    writer = TaskSocketWriter(socket, disconnect=lambda: None)

    def broken():
        raise RuntimeError("ledger bug")

    writer.enqueue(json.dumps({"event_id": "e1"}), on_sent=broken)
    await asyncio.wait_for(
        writer.enqueue(json.dumps({"event_id": "e2"}), acknowledge=True),
        timeout=CHECK_TIMEOUT,
    )
    assert [frame["event_id"] for frame in socket.frames] == ["e1", "e2"]
    assert not writer.closed
    writer.stop()
    await asyncio.gather(writer.task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_or_timed_out_sends_are_not_recorded(manager, monkeypatch, shared):
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )
    writer = manager._writers[socket]
    ledger = manager._delivery_ledgers[socket]
    socket.fail_event_ids["e2"] = asyncio.TimeoutError()
    await manager.deliver_shared_event(display(7, 1), 7)
    await manager.deliver_shared_event(display(7, 2), 7)
    await asyncio.wait_for(
        asyncio.gather(writer.task, return_exceptions=True), timeout=CHECK_TIMEOUT
    )
    assert set(ledger.sent) == {"e1"}
    assert socket not in manager._delivery_ledgers


@pytest.mark.asyncio
async def test_failed_broadcast_send_is_not_recorded(manager, monkeypatch):
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )
    ledger = manager._delivery_ledgers[socket]
    socket.fail_event_ids["e2"] = ConnectionError("gone")
    await manager.broadcast_to_task(display(7, 1), 7)
    await manager.broadcast_to_task(display(7, 2), 7)
    assert set(ledger.sent) == {"e1"}
    assert socket not in manager._delivery_ledgers


@pytest.mark.asyncio
async def test_writer_overflow_is_not_recorded(manager, monkeypatch, shared):
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )
    ledger = manager._delivery_ledgers[socket]
    oversized = {**display(7, 1), "data": {"blob": "x" * (2 * 1024 * 1024 + 1)}}
    await manager.deliver_shared_event(oversized, 7)
    assert not ledger.sent
    assert socket not in manager._delivery_ledgers


@pytest.mark.asyncio
async def test_bookkeeping_failure_never_changes_delivery(manager, monkeypatch):
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )

    def broken(*_args, **_kwargs):
        raise RuntimeError("ledger bug")

    monkeypatch.setattr(manager._delivery_ledgers[socket], "record", broken)
    await manager.broadcast_to_task(display(7, 1), 7)
    await manager.send_personal_message(display(7, 2), socket)
    assert [frame["event_id"] for frame in socket.frames[1:]] == ["e1", "e2"]
    assert manager.is_connection_registered(socket, 7)


@pytest.mark.asyncio
async def test_frame_for_another_task_is_ignored(manager, monkeypatch):
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )
    await manager.send_personal_message(display(8, 1), socket)
    await manager.send_personal_message({**display(7, 2), "task_id": None}, socket)
    assert not manager._delivery_ledgers[socket].sent


@pytest.mark.asyncio
async def test_move_connection_resets_the_ledger(
    manager, clock, monkeypatch, metric_reader
):
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )
    await manager.send_personal_message(display(7, 1), socket)
    manager.move_connection(socket, 8)
    assert socket not in manager._delivery_ledgers
    await manager.send_personal_message(display(8, 2), socket)
    assert socket not in manager._delivery_ledgers

    await close(manager, socket, clock)

    assert consistency(metric_reader) == {}


@pytest.mark.asyncio
async def test_connection_without_frames_after_the_horizon_is_idle(
    manager, clock, monkeypatch, metric_reader
):
    def unexpected(*_args):
        raise AssertionError("an idle ledger has nothing to compare")

    monkeypatch.setattr(websocket_api, "_expected_display_frames_sync", unexpected)
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(
            events=(display(7, 1), display(7, 2)), horizon=2, display_events=2
        ),
    )
    # A late live frame below the horizon is covered by the replay.
    await manager.send_personal_message(display(7, 2), socket)

    await close(manager, socket, clock)

    assert not manager._delivery_checks
    assert outcome(metric_reader, "idle")


@pytest.mark.asyncio
async def test_check_dropped_by_the_pending_cap_is_skipped(
    manager, clock, monkeypatch, caplog, metric_reader
):
    monkeypatch.setattr(websocket_api, "_DELIVERY_CHECK_PENDING_CAP", 0)
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )
    await manager.send_personal_message(display(7, 1), socket)

    caplog.set_level(logging.DEBUG, logger=websocket_api.logger.name)
    await close(manager, socket, clock)

    assert not manager._delivery_checks
    assert outcome(metric_reader, "skipped")
    assert any("Delivery check skipped" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_long_history_below_the_horizon_does_not_overflow(
    manager, clock, monkeypatch, metric_reader
):
    cap = websocket_api._DELIVERY_LEDGER_CAP
    horizon = cap + 10
    monkeypatch.setattr(
        websocket_api,
        "_expected_display_frames_sync",
        lambda *_args: {f"e{horizon + 1}": horizon + 1},
    )
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(
            events=tuple(display(7, seq) for seq in range(1, horizon + 1)),
            horizon=horizon,
            display_events=horizon,
        ),
    )
    await manager.send_personal_message(display(7, horizon + 1), socket)
    ledger = manager._delivery_ledgers[socket]
    assert not ledger.overflow and set(ledger.sent) == {f"e{horizon + 1}"}

    await close(manager, socket, clock)

    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_more_frames_than_the_cap_overflow_without_a_check(
    manager, clock, monkeypatch, metric_reader
):
    def unexpected(*_args):
        raise AssertionError("an overflowed ledger must not be checked")

    monkeypatch.setattr(websocket_api, "_expected_display_frames_sync", unexpected)
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
    )
    for sequence in range(1, websocket_api._DELIVERY_LEDGER_CAP + 2):
        await manager.send_personal_message(display(7, sequence), socket)
    ledger = manager._delivery_ledgers[socket]
    assert ledger.overflow and not ledger.sent

    await close(manager, socket, clock)

    assert not manager._delivery_checks
    assert outcome(metric_reader, "overflow")


@pytest.mark.asyncio
async def test_v1_socket_records_nothing(manager, monkeypatch, metric_reader):
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(
            events=(
                {"type": "task_info"},
                {"type": "trace_event", "event_id": "v1", "data": {}},
            )
        ),
    )
    await manager.broadcast_to_task(
        {"type": "trace_event", "event_id": "v1-live", "data": {}}, 7
    )
    assert manager._delivery_ledgers == {}

    manager.disconnect(socket)
    assert not manager._delivery_checks
    assert consistency(metric_reader) == {}


@pytest.mark.asyncio
async def test_sockets_that_never_replay_get_no_ledger(manager, monkeypatch):
    """SSE sinks and preview sockets register without a replay."""
    socket = Socket()
    manager.register_connection(socket, 7)
    await manager.broadcast_to_task(display(7, 1), 7)
    await manager.send_personal_message(display(7, 2), socket)
    assert manager._delivery_ledgers == {}


@pytest.mark.asyncio
async def test_check_runs_off_loop_and_a_failing_check_only_warns(
    manager, clock, monkeypatch, caplog, metric_reader
):
    loop_thread = threading.get_ident()
    threads: list[int] = []
    release = threading.Event()

    def failing(*_args):
        threads.append(threading.get_ident())
        release.wait(CHECK_TIMEOUT)
        raise OperationalError("SELECT", {}, Exception(SENTINEL))

    monkeypatch.setattr(websocket_api, "_expected_display_frames_sync", failing)
    socket = Socket()
    manager.register_connection(socket, 7)
    await replay(
        monkeypatch,
        socket,
        7,
        _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=3),
    )
    await manager.send_personal_message(display(7, 4), socket)

    caplog.set_level(logging.INFO)
    clock.advance(GRACE + 1)
    # Disconnect returns at once while the check is still blocked in a thread.
    manager.disconnect(socket)
    assert not manager.has_connections_for_task(7)
    assert socket not in manager._delivery_ledgers
    (check,) = manager._delivery_checks
    assert not check.done()
    release.set()
    await settle_checks(manager)

    assert threads and loop_thread not in threads
    assert consistency(metric_reader) == {}
    (line,) = signal_lines(caplog, "execution_event_delivery_check_failed")
    assert fields(line) == {
        "task_id": "7",
        "storage": "v2",
        "component": "execution-events",
        "horizon": "3",
        "error_type": "OperationalError",
    }
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
async def test_pending_checks_are_cancelled_on_shutdown(
    manager, clock, monkeypatch, metric_reader
):
    started = threading.Event()
    release = threading.Event()

    def blocked(*_args):
        started.set()
        release.wait(CHECK_TIMEOUT)
        return {}

    monkeypatch.setattr(websocket_api, "_expected_display_frames_sync", blocked)

    async def replayed_socket():
        socket = Socket()
        manager.register_connection(socket, 7)
        await replay(
            monkeypatch,
            socket,
            7,
            _HistoricalStreamSnapshot(events=({"type": "task_info"},), horizon=0),
        )
        await manager.send_personal_message(display(7, 1), socket)
        return socket

    first = await replayed_socket()
    late = await replayed_socket()
    clock.advance(GRACE + 1)
    manager.disconnect(first)
    (check,) = manager._delivery_checks
    await asyncio.to_thread(started.wait, CHECK_TIMEOUT)
    stopping = asyncio.create_task(manager.stop_delivery_checks())
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(stopping, timeout=CHECK_TIMEOUT)
    assert check.cancelled()
    assert not manager._delivery_checks

    # After shutdown, a disconnect schedules no database read and no outcome.
    manager.disconnect(late)
    assert not manager._delivery_checks
    assert consistency(metric_reader) == {}

    # A registration starts a new lifespan: checks run again.
    again = await replayed_socket()
    await close(manager, again, clock)
    assert outcome(metric_reader, "complete")


@pytest.mark.asyncio
async def test_replay_logs_one_payload_free_line(
    canonical, manager, monkeypatch, caplog
):
    factory, task_id = canonical
    rows = write_history(factory, task_id)
    with factory() as db:
        task = db.get(Task, task_id)
        owner = int(task.user_id)
    socket = Socket()
    manager.register_connection(socket, task_id)
    caplog.set_level(logging.INFO)
    user = SimpleNamespace(id=owner, is_admin=False)
    set_cache_backend_for_testing(InMemoryTTLCache())
    try:
        await websocket_api.send_historical_data_as_stream(socket, task_id, user)
        await websocket_api.send_historical_data_as_stream(socket, task_id, user)
    finally:
        set_cache_backend_for_testing(None)

    lines = signal_lines(caplog, "execution_event_replay ")
    assert len(lines) == 2
    expected = {
        "task_id": str(task_id),
        "storage": "v2",
        "component": "execution-events",
        "horizon": str(rows[-1].sequence),
        # input, before, live-1, live-2, live-3 (child/audit/internal excluded)
        "events": "5",
    }
    first, second = (fields(line) for line in lines)
    assert float(first["duration_ms"]) >= 0
    assert {k: v for k, v in first.items() if k != "duration_ms"} == {
        **expected,
        "cache_hit": "False",
    }
    # The cached replay reports the same horizon and display event count.
    assert {k: v for k, v in second.items() if k != "duration_ms"} == {
        **expected,
        "cache_hit": "True",
    }
    assert manager._delivery_ledgers[socket].horizon == rows[-1].sequence
    for line in signal_lines(caplog):
        assert SENTINEL not in line
    assert SENTINEL in json.dumps(socket.frames)


def bridge_frames(reader):
    return exported(reader, "xagent.task_event.bridge.frames")


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.setenv("XAGENT_REDIS_URL", "redis://localhost:6379/0")
    return task_event_bridge.TaskEventBridge()


@pytest.mark.asyncio
async def test_bridge_counts_duplicates_and_gaps_by_ids_only(
    bridge, caplog, metric_reader
):
    async def deliver(message, task_id):
        return None

    async def status(kind):
        return None

    bridge.deliver = deliver
    bridge.stream_status = status
    body = {
        "version": 1,
        "kind": "event",
        "host": "worker-1",
        "sequence": 1,
        "task_id": 42,
        "message": {"type": "trace_event", "data": {"message": SENTINEL}},
    }
    await bridge._receive(body)
    await bridge._receive(body)
    await bridge._receive({**body, "sequence": 4, "task_id": 43})
    await bridge.close()

    assert bridge_frames(metric_reader) == {
        key(outcome="duplicate_dropped"): 1,
        key(outcome="gap_resync"): 1,
    }
    (record,) = [
        r for r in caplog.records if r.getMessage().startswith("task_event_bridge_gap")
    ]
    assert record.levelno == logging.WARNING
    assert fields(record.getMessage()) == {
        "task_id": "43",
        "host": "worker-1",
        "previous_sequence": "1",
        "sequence": "4",
    }
    assert SENTINEL not in caplog.text


@pytest.mark.asyncio
async def test_bridge_gap_signal_never_blocks_the_resync(bridge, caplog, metric_reader):
    statuses = []

    async def deliver(message, task_id):
        return None

    async def status(kind):
        statuses.append(kind)

    bridge.deliver = deliver
    bridge.stream_status = status
    body = {
        "version": 1,
        "kind": "event",
        "host": "worker-1",
        "sequence": 1,
        "task_id": 42,
        "message": {"type": "heartbeat"},
    }
    await bridge._receive(body)
    gapped = {key: value for key, value in body.items() if key != "task_id"}
    with pytest.raises(KeyError):
        await bridge._receive({**gapped, "sequence": 3})
    await bridge.close()

    assert statuses == ["stream_resync_required"]
    assert bridge._seen["worker-1"] == 3
    assert bridge_frames(metric_reader) == {key(outcome="gap_resync"): 1}


@pytest.mark.asyncio
async def test_bridge_counts_publish_failures(bridge, monkeypatch, metric_reader):
    async def fail(channel, body):
        raise ConnectionError("redis down")

    monkeypatch.setattr(bridge, "_publish", fail)
    await bridge.publish({"type": "heartbeat"}, 42)
    await bridge.close()

    assert bridge_frames(metric_reader) == {key(outcome="publish_failed"): 1}
