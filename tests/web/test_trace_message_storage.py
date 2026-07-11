from __future__ import annotations

import copy
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from scripts.convert_trace_checkpoint_messages import convert_trace_checkpoint_messages
from xagent.core.agent.checkpoint import CHECKPOINT_EVENT_TYPE, CHECKPOINT_TYPE
from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    TraceScope,
)
from xagent.core.tools.adapters.vibe.connector_runtime import REDACTED_RUNTIME_SECRET
from xagent.web.api.trace_handlers import DatabaseTraceHandler
from xagent.web.models.database import Base
from xagent.web.models.task import (
    Task,
    TaskStatus,
    TraceCheckpointBlob,
)
from xagent.web.models.task import TraceEvent as DatabaseTraceEvent
from xagent.web.models.task import (
    TraceMessageBlob,
)
from xagent.web.models.user import User
from xagent.web.services.trace_message_storage import (
    CHECKPOINT_BLOB_REF_ENCODING,
    LEDGER_REFS_ENCODING,
    MESSAGE_REFS_DECODE_ERROR_KEY,
    MESSAGE_REFS_ENCODING,
    TOOL_LEDGER_RECORD_KIND,
    CheckpointMessageDecodeError,
    canonical_json_hash,
    decode_trace_event_data,
    decode_trace_events_data,
    encode_checkpoint_data_for_storage,
    encode_checkpoint_messages_for_storage,
)


def _session_factory() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _sqlite_driver_connection(db: Session) -> sqlite3.Connection:
    raw_connection = db.connection().connection
    return (
        getattr(raw_connection, "driver_connection", None)
        or getattr(raw_connection, "connection", None)
        or raw_connection
    )


def _create_task(db: Session) -> Task:
    user = User(username="tester", password_hash="hashed_password", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    task = Task(
        user_id=int(user.id),
        title="Checkpoint task",
        description="Checkpoint task",
        status=TaskStatus.PENDING,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def _checkpoint_data(execution_id: str, messages: Any) -> dict[str, Any]:
    return {
        "checkpoint_type": CHECKPOINT_TYPE,
        "root_execution_id": execution_id,
        "execution_id": execution_id,
        "label": "after_llm",
        "snapshot": {
            "type": "checkpoint",
            "label": "after_llm",
            "execution_id": execution_id,
            "context": {"messages": messages},
            "pattern": "ReActPattern",
            "pattern_state": {"current_iteration": 1},
        },
    }


def _checkpoint_data_with_large_fields(
    execution_id: str,
    messages: Any,
    *,
    tool_ledger: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = _checkpoint_data(execution_id, messages)
    data["snapshot"]["pattern_state"]["tool_ledger"] = tool_ledger or {
        "tool-1": {"result": "large tool output"}
    }
    data["snapshot"]["context"]["metadata"] = metadata or {
        "retrieved_memories": {"react_memory": "large memory context"}
    }
    return data


def _get_db_factory(SessionLocal: sessionmaker[Session]) -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_message_refs_codec_round_trips_and_dedupes_blobs() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
        ]

        encoded = encode_checkpoint_messages_for_storage(
            db,
            task_id=int(task.id),
            data=_checkpoint_data("exec-1", messages),
        )
        db.flush()

        refs_payload = encoded["snapshot"]["context"]["messages"]
        assert refs_payload["__encoding"] == MESSAGE_REFS_ENCODING
        assert refs_payload["count"] == 3
        assert refs_payload["hash"] == canonical_json_hash(refs_payload["refs"])
        assert len(refs_payload["refs"]) == 3
        assert db.query(TraceMessageBlob).count() == 2

        decoded = decode_trace_event_data(
            db,
            task_id=int(task.id),
            data=encoded,
            strict=True,
        )
        assert decoded["snapshot"]["context"]["messages"] == messages
    finally:
        db.close()


def test_message_refs_decoder_passes_old_inline_messages_through() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        data = _checkpoint_data("exec-old", [{"role": "user", "content": "old"}])

        decoded = decode_trace_event_data(
            db,
            task_id=int(task.id),
            data=data,
            strict=True,
        )

        assert decoded == data
    finally:
        db.close()


def test_message_refs_decoder_rejects_missing_blob() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        data = _checkpoint_data(
            "exec-missing",
            {
                "__encoding": MESSAGE_REFS_ENCODING,
                "count": 1,
                "hash": canonical_json_hash(["sha256:missing"]),
                "refs": ["sha256:missing"],
            },
        )

        with pytest.raises(CheckpointMessageDecodeError):
            decode_trace_event_data(db, task_id=int(task.id), data=data, strict=True)

        fallback = decode_trace_event_data(
            db,
            task_id=int(task.id),
            data=data,
            strict=False,
        )
        assert MESSAGE_REFS_DECODE_ERROR_KEY in fallback
    finally:
        db.close()


def test_message_refs_decoder_rejects_malformed_refs_payload() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        data = _checkpoint_data(
            "exec-malformed",
            {
                "__encoding": MESSAGE_REFS_ENCODING,
                "count": "1",
                "refs": ["sha256:missing"],
            },
        )

        with pytest.raises(CheckpointMessageDecodeError):
            decode_trace_event_data(db, task_id=int(task.id), data=data, strict=True)
    finally:
        db.close()


def test_message_refs_decoder_rejects_count_and_sequence_mismatch() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        count_mismatch = _checkpoint_data(
            "exec-bad-count",
            {
                "__encoding": MESSAGE_REFS_ENCODING,
                "count": 2,
                "hash": canonical_json_hash(["sha256:missing"]),
                "refs": ["sha256:missing"],
            },
        )
        sequence_mismatch = _checkpoint_data(
            "exec-bad-sequence",
            {
                "__encoding": MESSAGE_REFS_ENCODING,
                "count": 1,
                "hash": canonical_json_hash(["sha256:different"]),
                "refs": ["sha256:missing"],
            },
        )

        with pytest.raises(CheckpointMessageDecodeError):
            decode_trace_event_data(
                db,
                task_id=int(task.id),
                data=count_mismatch,
                strict=True,
            )
        with pytest.raises(CheckpointMessageDecodeError):
            decode_trace_event_data(
                db,
                task_id=int(task.id),
                data=sequence_mismatch,
                strict=True,
            )
    finally:
        db.close()


def test_checkpoint_blob_refs_round_trip_and_dedupe() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        messages = [{"role": "user", "content": "hello"}]
        tool_ledger = {"call-1": {"result": "same output"}}
        metadata = {"retrieved_memories": {"react_memory": "same memory"}}
        data = _checkpoint_data_with_large_fields(
            "exec-fields",
            messages,
            tool_ledger=tool_ledger,
            metadata=metadata,
        )

        encoded = encode_checkpoint_data_for_storage(
            db,
            task_id=int(task.id),
            data=data,
        )
        db.flush()

        stored_messages = encoded["snapshot"]["context"]["messages"]
        stored_tool_ledger = encoded["snapshot"]["pattern_state"]["tool_ledger"]
        stored_metadata = encoded["snapshot"]["context"]["metadata"]
        assert stored_messages["__encoding"] == MESSAGE_REFS_ENCODING
        assert stored_tool_ledger == {
            "__encoding": LEDGER_REFS_ENCODING,
            "count": 1,
            "records": [["call-1", canonical_json_hash(tool_ledger["call-1"])]],
        }
        assert stored_metadata == {
            "__encoding": CHECKPOINT_BLOB_REF_ENCODING,
            "kind": "context.metadata",
            "hash": canonical_json_hash(metadata),
        }
        assert db.query(TraceMessageBlob).count() == 1
        assert db.query(TraceCheckpointBlob).count() == 2

        encoded_again = encode_checkpoint_data_for_storage(
            db,
            task_id=int(task.id),
            data=data,
        )
        db.flush()
        assert (
            encoded_again["snapshot"]["pattern_state"]["tool_ledger"]
            == stored_tool_ledger
        )
        assert db.query(TraceCheckpointBlob).count() == 2

        decoded = decode_trace_event_data(
            db,
            task_id=int(task.id),
            data=encoded,
            strict=True,
        )
        assert decoded["snapshot"]["context"]["messages"] == messages
        assert decoded["snapshot"]["pattern_state"]["tool_ledger"] == tool_ledger
        assert decoded["snapshot"]["context"]["metadata"] == metadata
    finally:
        db.close()


def test_decode_trace_events_data_batches_blob_queries() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        first_messages = [{"role": "user", "content": "hello"}]
        second_messages = [
            *first_messages,
            {"role": "assistant", "content": "hi"},
        ]
        shared_tool_ledger = {"call-1": {"result": "same output"}}
        first = encode_checkpoint_data_for_storage(
            db,
            task_id=task_id,
            data=_checkpoint_data_with_large_fields(
                "exec-batch",
                first_messages,
                tool_ledger=shared_tool_ledger,
            ),
        )
        second = encode_checkpoint_data_for_storage(
            db,
            task_id=task_id,
            data=_checkpoint_data_with_large_fields(
                "exec-batch",
                second_messages,
                tool_ledger=shared_tool_ledger,
            ),
        )
        db.flush()

        statements: list[str] = []
        engine = SessionLocal.kw["bind"]

        def capture_statement(
            conn: Any,
            cursor: Any,
            statement: str,
            parameters: Any,
            context: Any,
            executemany: bool,
        ) -> None:
            if (
                "trace_message_blobs" in statement
                or "trace_checkpoint_blobs" in statement
            ):
                statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture_statement)
        try:
            decoded = decode_trace_events_data(
                db,
                task_id=task_id,
                data_items=[first, second],
                strict=True,
            )
        finally:
            event.remove(engine, "before_cursor_execute", capture_statement)

        assert decoded[0]["snapshot"]["context"]["messages"] == first_messages
        assert decoded[1]["snapshot"]["context"]["messages"] == second_messages
        assert len(statements) == 3
    finally:
        db.close()


def test_decode_trace_events_data_degrades_when_bulk_lookup_fails() -> None:
    from unittest.mock import patch

    import xagent.web.services.trace_message_storage as tms

    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        messages = [{"role": "user", "content": "hello"}]
        item = encode_checkpoint_data_for_storage(
            db,
            task_id=task_id,
            data=_checkpoint_data_with_large_fields("exec-degrade", messages),
        )
        db.flush()

        boom = RuntimeError("bulk lookup boom")

        # Non-strict: a failed bulk lookup must not drop the event; it degrades to
        # per-ref blob queries and still resolves the messages.
        with patch.object(tms, "_load_trace_blob_lookup", side_effect=boom):
            decoded = decode_trace_events_data(
                db, task_id=task_id, data_items=[item], strict=False
            )
        assert len(decoded) == 1
        assert decoded[0]["snapshot"]["context"]["messages"] == messages

        # Strict: the failure surfaces instead of being swallowed.
        with patch.object(tms, "_load_trace_blob_lookup", side_effect=boom):
            with pytest.raises(RuntimeError):
                decode_trace_events_data(
                    db, task_id=task_id, data_items=[item], strict=True
                )
    finally:
        db.close()


def test_ledger_records_decode_without_bulk_lookup() -> None:
    """The no-prefetch fallback resolves per-record ledger refs correctly."""
    from unittest.mock import patch

    import xagent.web.services.trace_message_storage as tms

    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        tool_ledger = {
            f"call-{index}": {
                "tool_call_id": f"call-{index}",
                "result": f"output-{index}",
            }
            for index in range(5)
        }
        item = encode_checkpoint_data_for_storage(
            db,
            task_id=task_id,
            data=_checkpoint_data_with_large_fields(
                "exec-ledger-fallback",
                [{"role": "user", "content": "hello"}],
                tool_ledger=tool_ledger,
            ),
        )
        db.flush()
        assert (
            item["snapshot"]["pattern_state"]["tool_ledger"]["__encoding"]
            == LEDGER_REFS_ENCODING
        )

        boom = RuntimeError("bulk lookup boom")
        with patch.object(tms, "_load_trace_blob_lookup", side_effect=boom):
            decoded = decode_trace_events_data(
                db, task_id=task_id, data_items=[item], strict=False
            )

        assert decoded[0]["snapshot"]["pattern_state"]["tool_ledger"] == tool_ledger
        assert list(decoded[0]["snapshot"]["pattern_state"]["tool_ledger"]) == list(
            tool_ledger
        )
    finally:
        db.close()


def test_blob_lookups_respect_sqlite_bind_parameter_limit() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    raw_connection = _sqlite_driver_connection(db)
    previous_limit = raw_connection.setlimit(
        sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER,
        50,
    )
    try:
        task = _create_task(db)
        task_id = int(task.id)
        big_messages = [
            {"role": "user", "content": f"large-history-{index}"} for index in range(60)
        ]
        big_checkpoint = encode_checkpoint_data_for_storage(
            db,
            task_id=task_id,
            data=_checkpoint_data_with_large_fields(
                "exec-limit-big",
                big_messages,
                metadata={"memory": "big"},
            ),
        )

        checkpoints = [big_checkpoint]
        for index in range(60):
            checkpoints.append(
                encode_checkpoint_data_for_storage(
                    db,
                    task_id=task_id,
                    data=_checkpoint_data_with_large_fields(
                        f"exec-limit-{index}",
                        [{"role": "user", "content": f"small-history-{index}"}],
                        tool_ledger={f"tool-{index}": {"result": f"value-{index}"}},
                        metadata={"memory": f"value-{index}"},
                    ),
                )
            )
        db.flush()

        decoded = decode_trace_events_data(
            db,
            task_id=task_id,
            data_items=checkpoints,
            strict=True,
        )
        assert decoded[0]["snapshot"]["context"]["messages"] == big_messages

        single_decoded = decode_trace_event_data(
            db,
            task_id=task_id,
            data=big_checkpoint,
            strict=True,
        )
        assert single_decoded["snapshot"]["context"]["messages"] == big_messages
    finally:
        raw_connection.setlimit(
            sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER,
            previous_limit,
        )
        db.close()


def test_bulk_decode_can_skip_blob_hash_verification() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        messages = [{"role": "user", "content": "original"}]
        encoded = encode_checkpoint_data_for_storage(
            db,
            task_id=task_id,
            data=_checkpoint_data("exec-tampered", messages),
        )
        db.flush()

        message_ref = encoded["snapshot"]["context"]["messages"]["refs"][0]
        blob = (
            db.query(TraceMessageBlob)
            .filter(
                TraceMessageBlob.task_id == task_id,
                TraceMessageBlob.message_hash == message_ref,
            )
            .one()
        )
        blob.message_data = {"role": "user", "content": "tampered"}
        db.flush()

        decoded = decode_trace_events_data(
            db,
            task_id=task_id,
            data_items=[encoded],
            strict=False,
        )
        assert decoded[0]["snapshot"]["context"]["messages"] == [
            {"role": "user", "content": "tampered"}
        ]

        with pytest.raises(CheckpointMessageDecodeError):
            decode_trace_events_data(
                db,
                task_id=task_id,
                data_items=[encoded],
                strict=True,
            )
    finally:
        db.close()


def test_checkpoint_blob_refs_decoder_passes_old_inline_fields_through() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        data = _checkpoint_data_with_large_fields(
            "exec-old-fields",
            [{"role": "user", "content": "old"}],
        )

        decoded = decode_trace_event_data(
            db,
            task_id=int(task.id),
            data=data,
            strict=True,
        )

        assert decoded == data
    finally:
        db.close()


def test_checkpoint_blob_refs_decoder_rejects_missing_blob() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        data = _checkpoint_data(
            "exec-missing-field", [{"role": "user", "content": "x"}]
        )
        data["snapshot"]["pattern_state"]["tool_ledger"] = {
            "__encoding": CHECKPOINT_BLOB_REF_ENCODING,
            "kind": "pattern_state.tool_ledger",
            "hash": "sha256:missing",
        }

        with pytest.raises(CheckpointMessageDecodeError):
            decode_trace_event_data(db, task_id=int(task.id), data=data, strict=True)

        fallback = decode_trace_event_data(
            db,
            task_id=int(task.id),
            data=data,
            strict=False,
        )
        assert MESSAGE_REFS_DECODE_ERROR_KEY in fallback
    finally:
        db.close()


def test_database_trace_handler_stores_checkpoint_messages_as_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        handler = DatabaseTraceHandler(task_id)
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "working"},
        ]
        event = TraceEvent(
            CHECKPOINT_EVENT_TYPE,
            task_id=str(task_id),
            data=_checkpoint_data_with_large_fields("exec-handler", messages),
            require_persisted=True,
        )

        handler._save_trace_event(db, event)

        row = db.query(DatabaseTraceEvent).filter_by(task_id=task_id).one()
        stored_messages = row.data["snapshot"]["context"]["messages"]
        stored_tool_ledger = row.data["snapshot"]["pattern_state"]["tool_ledger"]
        assert stored_messages["__encoding"] == MESSAGE_REFS_ENCODING
        assert stored_tool_ledger["__encoding"] == LEDGER_REFS_ENCODING
        assert db.query(TraceMessageBlob).count() == 2
        assert db.query(TraceCheckpointBlob).count() == 2

        monkeypatch.setattr(
            "xagent.web.api.trace_handlers.get_db",
            lambda: _get_db_factory(SessionLocal),
        )
        loaded = handler._sync_load_latest_checkpoint("exec-handler")
        assert loaded is not None
        assert loaded["context"]["messages"] == messages
        assert loaded["pattern_state"]["tool_ledger"] == {
            "tool-1": {"result": "large tool output"}
        }
    finally:
        db.close()


def test_database_trace_handler_redacts_runtime_secrets_in_tool_events() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        handler = DatabaseTraceHandler(task_id)
        event = TraceEvent(
            TraceEventType(TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL),
            task_id=str(task_id),
            step_id="step-1",
            data={
                "tool_name": "shiftcare",
                "tool_args": {
                    "headers": {
                        "Authorization": "Bearer raw-runtime-token",
                        "X-Account": "6185",
                    },
                    "connector_runtime": {
                        "secrets": {"authorization": "Bearer raw-runtime-token"},
                        "auth_selector": {"resource_owner_key": "xagent:user:1"},
                    },
                },
            },
        )

        handler._save_trace_event(db, event)

        row = db.query(DatabaseTraceEvent).filter_by(task_id=task_id).one()
        assert "raw-runtime-token" not in str(row.data)
        assert "xagent:user:1" not in str(row.data)
        tool_args = row.data["tool_args"]
        assert tool_args["headers"]["Authorization"] == REDACTED_RUNTIME_SECRET
        assert tool_args["headers"]["X-Account"] == "6185"
        assert (
            tool_args["connector_runtime"]["secrets"]["authorization"]
            == REDACTED_RUNTIME_SECRET
        )
        assert (
            tool_args["connector_runtime"]["auth_selector"]["resource_owner_key"]
            == REDACTED_RUNTIME_SECRET
        )
    finally:
        db.close()


def test_database_trace_handler_shares_blobs_across_checkpoints() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        handler = DatabaseTraceHandler(task_id)
        first = [{"role": "user", "content": "task"}]
        second = [*first, {"role": "assistant", "content": "done"}]
        shared_tool_ledger = {"tool-1": {"result": "same"}}
        shared_metadata = {"retrieved_memories": {"react_memory": "same"}}

        handler._save_trace_event(
            db,
            TraceEvent(
                CHECKPOINT_EVENT_TYPE,
                task_id=str(task_id),
                data=_checkpoint_data_with_large_fields(
                    "exec-shared",
                    first,
                    tool_ledger=shared_tool_ledger,
                    metadata=shared_metadata,
                ),
            ),
        )
        handler._save_trace_event(
            db,
            TraceEvent(
                CHECKPOINT_EVENT_TYPE,
                task_id=str(task_id),
                data=_checkpoint_data_with_large_fields(
                    "exec-shared",
                    second,
                    tool_ledger=shared_tool_ledger,
                    metadata=shared_metadata,
                ),
            ),
        )

        assert db.query(DatabaseTraceEvent).count() == 2
        assert db.query(TraceMessageBlob).count() == 2
        assert db.query(TraceCheckpointBlob).count() == 2
    finally:
        db.close()


def test_database_trace_handler_reads_old_inline_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        messages = [{"role": "user", "content": "old"}]
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="old-inline",
                event_type="system_update_general",
                timestamp=datetime.now(timezone.utc),
                data=_checkpoint_data("exec-old-inline", messages),
            )
        )
        db.commit()

        monkeypatch.setattr(
            "xagent.web.api.trace_handlers.get_db",
            lambda: _get_db_factory(SessionLocal),
        )
        loaded = DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
            "exec-old-inline"
        )

        assert loaded is not None
        assert loaded["context"]["messages"] == messages
    finally:
        db.close()


def test_database_trace_handler_falls_back_when_latest_refs_are_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        now = datetime.now(timezone.utc)
        old_messages = [{"role": "user", "content": "old"}]
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="old-readable",
                event_type="system_update_general",
                timestamp=now,
                data=_checkpoint_data("exec-fallback", old_messages),
            )
        )
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="new-unreadable",
                event_type="system_update_general",
                timestamp=now + timedelta(seconds=1),
                data=_checkpoint_data(
                    "exec-fallback",
                    {
                        "__encoding": MESSAGE_REFS_ENCODING,
                        "count": 1,
                        "hash": canonical_json_hash(["sha256:missing"]),
                        "refs": ["sha256:missing"],
                    },
                ),
            )
        )
        db.commit()

        monkeypatch.setattr(
            "xagent.web.api.trace_handlers.get_db",
            lambda: _get_db_factory(SessionLocal),
        )
        loaded = DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
            "exec-fallback"
        )

        assert loaded is not None
        assert loaded["context"]["messages"] == old_messages
    finally:
        db.close()


def _context_payload(
    execution_id: str,
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> dict[str, Any]:
    return {
        "execution_id": execution_id,
        "messages": messages,
        "system_prompt": system_prompt,
        "metadata": {},
    }


def _dag_checkpoint_data(
    execution_id: str,
    messages: list[dict[str, Any]],
    system_prompt: str,
    tool_ledger: dict[str, Any],
) -> dict[str, Any]:
    root_context = _context_payload(execution_id, messages, system_prompt)
    child_context = _context_payload(f"{execution_id}_child", messages, system_prompt)
    return {
        "checkpoint_type": CHECKPOINT_TYPE,
        "root_execution_id": execution_id,
        "execution_id": execution_id,
        "label": "dag_step_completed",
        "snapshot": {
            "type": "checkpoint",
            "label": "dag_step_completed",
            "execution_id": execution_id,
            "context": root_context,
            "pattern": "DAGPlanExecutePattern",
            "pattern_state": {
                "status": "running",
                "active_step_contexts": {"step-1": dict(child_context)},
                "active_step_pattern_states": {"step-1": {"tool_ledger": tool_ledger}},
                "step_results": {},
            },
            "execution_snapshot": {
                "root_execution_id": execution_id,
                "frames": {
                    "root:dag": {
                        "frame_id": "root:dag",
                        "pattern_type": "dag",
                        "context": dict(root_context),
                        "pattern_state": {},
                    },
                    "child:react": {
                        "frame_id": "child:react",
                        "pattern_type": "react",
                        "context": dict(child_context),
                        "pattern_state": {"tool_ledger": tool_ledger},
                    },
                },
            },
        },
    }


def test_v2_ledger_records_dedupe_across_growing_checkpoints() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        record_one = {
            "tool_call_id": "call-1",
            "tool_name": "web_search",
            "args": {"query": "growth"},
            "status": "completed",
            "result": "large tool output " * 10,
        }
        record_two = {
            "tool_call_id": "call-2",
            "tool_name": "calculator",
            "args": {"expression": "1+1"},
            "status": "completed",
            "result": "2",
        }
        first = _checkpoint_data_with_large_fields(
            "exec-ledger",
            [{"role": "user", "content": "task"}],
            tool_ledger={"call-1": record_one},
        )
        second = _checkpoint_data_with_large_fields(
            "exec-ledger",
            [{"role": "user", "content": "task"}],
            tool_ledger={"call-1": record_one, "call-2": record_two},
        )

        encoded_first = encode_checkpoint_data_for_storage(
            db, task_id=task_id, data=first
        )
        encoded_second = encode_checkpoint_data_for_storage(
            db, task_id=task_id, data=second
        )
        db.flush()

        # The unchanged record is shared: two checkpoints with a growing
        # ledger store 2 record blobs, not 3.
        record_blobs = (
            db.query(TraceCheckpointBlob)
            .filter(TraceCheckpointBlob.blob_kind == TOOL_LEDGER_RECORD_KIND)
            .all()
        )
        assert len(record_blobs) == 2

        decoded_second = decode_trace_event_data(
            db, task_id=task_id, data=encoded_second, strict=True
        )
        assert decoded_second["snapshot"]["pattern_state"]["tool_ledger"] == {
            "call-1": record_one,
            "call-2": record_two,
        }
        assert list(decoded_second["snapshot"]["pattern_state"]["tool_ledger"]) == [
            "call-1",
            "call-2",
        ]

        decoded_first = decode_trace_event_data(
            db, task_id=task_id, data=encoded_first, strict=True
        )
        assert decoded_first["snapshot"]["pattern_state"]["tool_ledger"] == {
            "call-1": record_one
        }
    finally:
        db.close()


def test_v2_encodes_nested_contexts_system_prompt_and_frame_ledgers() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        messages = [
            {"role": "user", "content": "build a report"},
            {"role": "assistant", "content": "planning"},
        ]
        system_prompt = "You are a meticulous data analyst. " * 20
        tool_ledger = {"call-1": {"tool_call_id": "call-1", "result": "step output"}}
        data = _dag_checkpoint_data("exec-dag", messages, system_prompt, tool_ledger)
        original = copy.deepcopy(data)

        encoded = encode_checkpoint_data_for_storage(db, task_id=task_id, data=data)
        db.flush()

        snapshot = encoded["snapshot"]
        # Root context, both frames, and the active step context all encode
        # their messages as refs against the same shared blobs.
        assert snapshot["context"]["messages"]["__encoding"] == MESSAGE_REFS_ENCODING
        for frame in snapshot["execution_snapshot"]["frames"].values():
            assert frame["context"]["messages"]["__encoding"] == MESSAGE_REFS_ENCODING
        step_context = snapshot["pattern_state"]["active_step_contexts"]["step-1"]
        assert step_context["messages"]["__encoding"] == MESSAGE_REFS_ENCODING
        assert db.query(TraceMessageBlob).count() == len(messages)

        # The large system prompt is stored once and referenced everywhere.
        assert (
            snapshot["context"]["system_prompt"]["__encoding"]
            == CHECKPOINT_BLOB_REF_ENCODING
        )
        prompt_blobs = (
            db.query(TraceCheckpointBlob)
            .filter(TraceCheckpointBlob.blob_kind == "context.system_prompt")
            .all()
        )
        assert len(prompt_blobs) == 1
        assert prompt_blobs[0].blob_data == system_prompt

        # Nested ledgers (active step pattern state + react frame) encode
        # per record and share the same record blob.
        assert (
            snapshot["pattern_state"]["active_step_pattern_states"]["step-1"][
                "tool_ledger"
            ]["__encoding"]
            == LEDGER_REFS_ENCODING
        )
        assert (
            snapshot["execution_snapshot"]["frames"]["child:react"]["pattern_state"][
                "tool_ledger"
            ]["__encoding"]
            == LEDGER_REFS_ENCODING
        )
        record_blobs = (
            db.query(TraceCheckpointBlob)
            .filter(TraceCheckpointBlob.blob_kind == TOOL_LEDGER_RECORD_KIND)
            .all()
        )
        assert len(record_blobs) == 1

        decoded = decode_trace_event_data(
            db, task_id=task_id, data=encoded, strict=True
        )
        assert decoded == original
    finally:
        db.close()


def test_v2_small_system_prompt_and_empty_metadata_stay_inline() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        data = _checkpoint_data("exec-small", [{"role": "user", "content": "hi"}])
        data["snapshot"]["context"]["execution_id"] = "exec-small"
        data["snapshot"]["context"]["system_prompt"] = "short prompt"
        data["snapshot"]["context"]["metadata"] = {}

        encoded = encode_checkpoint_data_for_storage(db, task_id=task_id, data=data)
        db.flush()

        assert encoded["snapshot"]["context"]["system_prompt"] == "short prompt"
        assert encoded["snapshot"]["context"]["metadata"] == {}
        assert db.query(TraceCheckpointBlob).count() == 0
    finally:
        db.close()


def test_v1_flag_still_writes_whole_ledger_blob_and_v2_decodes_it() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        tool_ledger = {"call-1": {"result": "legacy output"}}
        data = _checkpoint_data_with_large_fields(
            "exec-v1",
            [{"role": "user", "content": "hello"}],
            tool_ledger=tool_ledger,
        )

        encoded = encode_checkpoint_data_for_storage(
            db, task_id=task_id, data=data, use_v2=False
        )
        db.flush()

        assert encoded["snapshot"]["pattern_state"]["tool_ledger"] == {
            "__encoding": CHECKPOINT_BLOB_REF_ENCODING,
            "kind": "pattern_state.tool_ledger",
            "hash": canonical_json_hash(tool_ledger),
        }

        decoded = decode_trace_event_data(
            db, task_id=task_id, data=encoded, strict=True
        )
        assert decoded["snapshot"]["pattern_state"]["tool_ledger"] == tool_ledger
    finally:
        db.close()


def test_database_trace_handler_prunes_checkpoint_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        handler = DatabaseTraceHandler(task_id)
        monkeypatch.setattr(
            "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
            lambda: 3,
        )
        # Force the batched-delete loop to run multiple chunks.
        monkeypatch.setattr(
            "xagent.web.api.trace_handlers.SQL_IN_CLAUSE_CHUNK_SIZE",
            1,
        )

        # A non-checkpoint system event and another execution's checkpoint
        # must both survive pruning.
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="other-system",
                event_type="system_update_general",
                timestamp=datetime.now(timezone.utc),
                data={"note": "not a checkpoint"},
            )
        )
        db.commit()
        handler._save_trace_event(
            db,
            TraceEvent(
                CHECKPOINT_EVENT_TYPE,
                task_id=str(task_id),
                data=_checkpoint_data(
                    "exec-other", [{"role": "user", "content": "other"}]
                ),
            ),
        )

        for index in range(5):
            handler._save_trace_event(
                db,
                TraceEvent(
                    CHECKPOINT_EVENT_TYPE,
                    task_id=str(task_id),
                    data=_checkpoint_data(
                        "exec-prune",
                        [{"role": "user", "content": f"turn-{index}"}],
                    ),
                ),
            )

        rows = (
            db.query(DatabaseTraceEvent)
            .filter(DatabaseTraceEvent.task_id == task_id)
            .order_by(DatabaseTraceEvent.id)
            .all()
        )
        prune_rows = [
            row for row in rows if row.data.get("execution_id") == "exec-prune"
        ]
        assert len(prune_rows) == 3
        # The most recent checkpoints are the ones kept.
        latest = decode_trace_event_data(
            db, task_id=task_id, data=prune_rows[-1].data, strict=True
        )
        assert latest["snapshot"]["context"]["messages"] == [
            {"role": "user", "content": "turn-4"}
        ]
        assert any(row.event_id == "other-system" for row in rows)
        assert any(row.data.get("execution_id") == "exec-other" for row in rows)
    finally:
        db.close()


def test_loader_falls_back_to_older_row_when_prefetch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient blob-prefetch failure must not abort checkpoint loading."""
    from unittest.mock import patch

    import xagent.web.services.trace_message_storage as tms

    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        old_messages = [{"role": "user", "content": "older inline"}]
        now = datetime.now(timezone.utc)
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="older-inline",
                event_type="system_update_general",
                timestamp=now,
                data=_checkpoint_data("exec-prefetch", old_messages),
            )
        )
        encoded = encode_checkpoint_data_for_storage(
            db,
            task_id=task_id,
            data=_checkpoint_data(
                "exec-prefetch", [{"role": "user", "content": "newest refs"}]
            ),
        )
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="newest-refs",
                event_type="system_update_general",
                timestamp=now + timedelta(seconds=1),
                data=encoded,
            )
        )
        db.commit()

        monkeypatch.setattr(
            "xagent.web.api.trace_handlers.get_db",
            lambda: _get_db_factory(SessionLocal),
        )
        # The newest row needs the prefetch (it has refs) and fails; the
        # older inline row has no markers, never touches the prefetch, and
        # must be returned as the fallback.
        with patch.object(
            tms,
            "_load_trace_blob_lookup",
            side_effect=RuntimeError("transient db error"),
        ):
            loaded = DatabaseTraceHandler(task_id)._sync_load_latest_checkpoint(
                "exec-prefetch"
            )

        assert loaded is not None
        assert loaded["context"]["messages"] == old_messages
    finally:
        db.close()


def test_prune_matches_legacy_execution_id_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows keyed only by root/nested execution ids must still be pruned."""
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        handler = DatabaseTraceHandler(task_id)
        monkeypatch.setattr(
            "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
            lambda: 2,
        )
        now = datetime.now(timezone.utc)

        # Legacy row: only root_execution_id set (no flat execution_id).
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="legacy-root-only",
                event_type="system_update_general",
                timestamp=now - timedelta(seconds=2),
                data={
                    "checkpoint_type": "agent_v2_execution_checkpoint",
                    "root_execution_id": "exec-legacy",
                    "snapshot": {"context": {"messages": []}},
                },
            )
        )
        # Legacy row: only the nested snapshot execution_id set.
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="legacy-nested-only",
                event_type="system_update_general",
                timestamp=now - timedelta(seconds=1),
                data={
                    "checkpoint_type": "agent_v2_execution_checkpoint",
                    "snapshot": {
                        "execution_id": "exec-legacy",
                        "context": {"messages": []},
                    },
                },
            )
        )
        db.commit()

        for index in range(2):
            handler._save_trace_event(
                db,
                TraceEvent(
                    CHECKPOINT_EVENT_TYPE,
                    task_id=str(task_id),
                    data=_checkpoint_data(
                        "exec-legacy",
                        [{"role": "user", "content": f"turn-{index}"}],
                    ),
                ),
            )

        remaining = [
            row.event_id
            for row in db.query(DatabaseTraceEvent)
            .filter(DatabaseTraceEvent.task_id == task_id)
            .order_by(DatabaseTraceEvent.id)
            .all()
        ]
        # Both legacy-shaped rows are older than the 2 new checkpoints and
        # must have been pruned by the coalesce predicate.
        assert "legacy-root-only" not in remaining
        assert "legacy-nested-only" not in remaining
        assert len(remaining) == 2
    finally:
        db.close()


def test_database_trace_handler_prune_disabled_keeps_all_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        handler = DatabaseTraceHandler(task_id)
        monkeypatch.setattr(
            "xagent.web.api.trace_handlers.get_checkpoint_history_limit",
            lambda: 0,
        )

        for index in range(4):
            handler._save_trace_event(
                db,
                TraceEvent(
                    CHECKPOINT_EVENT_TYPE,
                    task_id=str(task_id),
                    data=_checkpoint_data(
                        "exec-keep",
                        [{"role": "user", "content": f"turn-{index}"}],
                    ),
                ),
            )

        assert (
            db.query(DatabaseTraceEvent)
            .filter(DatabaseTraceEvent.task_id == task_id)
            .count()
            == 4
        )
    finally:
        db.close()


def test_convert_trace_checkpoint_messages_script_dry_run_and_execute() -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        inline_messages = [{"role": "user", "content": "convert me"}]
        already_encoded = encode_checkpoint_messages_for_storage(
            db,
            task_id=task_id,
            data=_checkpoint_data("exec-convert", inline_messages),
        )
        mixed_encoded = encode_checkpoint_messages_for_storage(
            db,
            task_id=task_id,
            data=_checkpoint_data_with_large_fields("exec-convert", inline_messages),
        )
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="inline",
                event_type="system_update_general",
                timestamp=datetime.now(timezone.utc),
                data=_checkpoint_data("exec-convert", inline_messages),
            )
        )
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="refs",
                event_type="system_update_general",
                timestamp=datetime.now(timezone.utc),
                data=already_encoded,
            )
        )
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="mixed",
                event_type="system_update_general",
                timestamp=datetime.now(timezone.utc),
                data=mixed_encoded,
            )
        )
        db.commit()

        dry_run_stats = convert_trace_checkpoint_messages(
            db,
            dry_run=True,
            batch_size=1,
        )
        assert dry_run_stats.converted_rows == 2
        assert dry_run_stats.already_refs_rows == 1
        inline_row = db.query(DatabaseTraceEvent).filter_by(event_id="inline").one()
        assert isinstance(inline_row.data["snapshot"]["context"]["messages"], list)
        mixed_row = db.query(DatabaseTraceEvent).filter_by(event_id="mixed").one()
        assert isinstance(
            mixed_row.data["snapshot"]["pattern_state"]["tool_ledger"], dict
        )
        assert (
            mixed_row.data["snapshot"]["pattern_state"]["tool_ledger"].get("__encoding")
            != CHECKPOINT_BLOB_REF_ENCODING
        )

        execute_stats = convert_trace_checkpoint_messages(
            db,
            dry_run=False,
            batch_size=1,
        )
        assert execute_stats.converted_rows == 2
        assert execute_stats.already_refs_rows == 1

        converted_row = db.query(DatabaseTraceEvent).filter_by(event_id="inline").one()
        stored_messages = converted_row.data["snapshot"]["context"]["messages"]
        assert stored_messages["__encoding"] == MESSAGE_REFS_ENCODING
        mixed_row = db.query(DatabaseTraceEvent).filter_by(event_id="mixed").one()
        assert (
            mixed_row.data["snapshot"]["pattern_state"]["tool_ledger"]["__encoding"]
            == LEDGER_REFS_ENCODING
        )
        assert db.query(TraceMessageBlob).count() == 1
        assert db.query(TraceCheckpointBlob).count() == 2

        repeat_stats = convert_trace_checkpoint_messages(db, dry_run=False)
        assert repeat_stats.converted_rows == 0
        assert repeat_stats.already_refs_rows == 3
    finally:
        db.close()


def test_convert_trace_checkpoint_messages_continues_after_row_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    SessionLocal = _session_factory()
    db = SessionLocal()
    try:
        task = _create_task(db)
        task_id = int(task.id)
        messages = [{"role": "user", "content": "convert me"}]
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="bad",
                event_type="system_update_general",
                timestamp=datetime.now(timezone.utc),
                data=_checkpoint_data("exec-bad", messages),
            )
        )
        db.add(
            DatabaseTraceEvent(
                task_id=task_id,
                event_id="good",
                event_type="system_update_general",
                timestamp=datetime.now(timezone.utc),
                data=_checkpoint_data("exec-good", messages),
            )
        )
        db.commit()

        from xagent.web.services import trace_message_storage

        original_encode = trace_message_storage.encode_checkpoint_data_for_storage

        def flaky_encode(db: Session, *, task_id: int, data: Any) -> Any:
            if data.get("execution_id") == "exec-bad":
                raise RuntimeError("synthetic conversion failure")
            return original_encode(db, task_id=task_id, data=data)

        monkeypatch.setattr(
            trace_message_storage,
            "encode_checkpoint_data_for_storage",
            flaky_encode,
        )

        stats = convert_trace_checkpoint_messages(
            db,
            dry_run=False,
            batch_size=10,
        )

        assert stats.error_rows == 1
        assert stats.converted_rows == 1
        bad_row = db.query(DatabaseTraceEvent).filter_by(event_id="bad").one()
        assert isinstance(bad_row.data["snapshot"]["context"]["messages"], list)
        good_row = db.query(DatabaseTraceEvent).filter_by(event_id="good").one()
        assert (
            good_row.data["snapshot"]["context"]["messages"]["__encoding"]
            == MESSAGE_REFS_ENCODING
        )
    finally:
        db.close()
