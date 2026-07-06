"""Storage codec for checkpoint message refs.

The refs shape is an internal database representation only. Runtime code and
public APIs should see decoded inline message lists.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy.orm import Session

from ...core.agent.checkpoint import READABLE_CHECKPOINT_TYPES
from ..models.task import TraceCheckpointBlob, TraceMessageBlob

logger = logging.getLogger(__name__)

MESSAGE_REFS_ENCODING = "xagent.message_refs_v1"
CHECKPOINT_BLOB_REF_ENCODING = "xagent.checkpoint_blob_ref_v1"
MESSAGE_HASH_PREFIX = "sha256:"
MESSAGE_REFS_DECODE_ERROR_KEY = "_decode_error"
SQL_IN_CLAUSE_CHUNK_SIZE = 900
CheckpointMessageStorageState = Literal["inline", "refs", "none"]
CheckpointStorageState = Literal["inline", "refs", "mixed", "none"]


@dataclass(frozen=True)
class CheckpointBlobField:
    kind: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class BlobCandidate:
    data: Any
    payload_bytes: int


@dataclass(frozen=True)
class TraceBlobLookup:
    message_data_by_hash: dict[str, Any]
    checkpoint_data_by_ref: dict[tuple[str, str], Any]


CHECKPOINT_BLOB_FIELDS: tuple[CheckpointBlobField, ...] = (
    CheckpointBlobField(
        "pattern_state.tool_ledger", ("snapshot", "pattern_state", "tool_ledger")
    ),
    CheckpointBlobField("context.metadata", ("snapshot", "context", "metadata")),
)


class CheckpointMessageDecodeError(ValueError):
    """Raised when checkpoint message refs cannot be restored."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_hash(value: Any) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"{MESSAGE_HASH_PREFIX}{digest}"


def canonical_json_hash_from_bytes(payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    return f"{MESSAGE_HASH_PREFIX}{digest}"


def get_checkpoint_messages_storage_state(
    data: Any,
) -> CheckpointMessageStorageState:
    """Return how checkpoint context messages are currently stored."""

    messages_payload = _get_checkpoint_messages_payload(data)
    if isinstance(messages_payload, list):
        return "inline"
    if _is_message_refs_payload(messages_payload):
        return "refs"
    return "none"


def get_checkpoint_storage_state(data: Any) -> CheckpointStorageState:
    """Return whether all checkpoint-optimized fields are inline or refs."""

    field_states = _checkpoint_storage_field_states(data)
    has_inline = "inline" in field_states
    has_refs = "refs" in field_states
    if has_inline and has_refs:
        return "mixed"
    if has_inline:
        return "inline"
    if has_refs:
        return "refs"
    return "none"


def checkpoint_storage_payload_bytes(data: Any) -> int:
    """Measure optimized checkpoint payload bytes inside one trace event."""

    if not _is_readable_checkpoint_data(data):
        return 0

    total = 0
    messages = _get_checkpoint_messages_payload(data)
    if isinstance(messages, list) or _is_message_refs_payload(messages):
        total += len(canonical_json_bytes(messages))

    for field in CHECKPOINT_BLOB_FIELDS:
        payload = _get_nested(data, field.path)
        if _should_store_checkpoint_blob_payload(
            payload
        ) or _is_checkpoint_blob_ref_payload(payload):
            total += len(canonical_json_bytes(payload))
    return total


def encode_checkpoint_data_for_storage(
    db: Session,
    *,
    task_id: int,
    data: Any,
) -> Any:
    """Encode all storage-optimized checkpoint fields."""

    should_encode_messages = get_checkpoint_messages_storage_state(data) == "inline"
    fields_to_encode = _checkpoint_blob_fields_to_encode(data)
    if not should_encode_messages and not fields_to_encode:
        return data

    encoded = copy.deepcopy(data)
    if should_encode_messages:
        _encode_checkpoint_messages_in_place(db, task_id=task_id, data=encoded)
    if fields_to_encode:
        _encode_checkpoint_blob_fields_in_place(
            db,
            task_id=task_id,
            data=encoded,
            fields_to_encode=fields_to_encode,
        )
    return encoded


def encode_checkpoint_messages_for_storage(
    db: Session,
    *,
    task_id: int,
    data: Any,
) -> Any:
    """Replace checkpoint context messages with refs and upsert message blobs."""

    if get_checkpoint_messages_storage_state(data) != "inline":
        return data

    messages = _get_checkpoint_messages_payload(data)
    if not isinstance(messages, list):
        return data

    encoded = copy.deepcopy(data)
    _encode_checkpoint_messages_in_place(db, task_id=task_id, data=encoded)
    return encoded


def _encode_checkpoint_messages_in_place(
    db: Session,
    *,
    task_id: int,
    data: dict[str, Any],
) -> None:
    messages = _get_checkpoint_messages_payload(data)
    if not isinstance(messages, list):
        return

    encoded_context = data["snapshot"]["context"]
    execution_id = _checkpoint_execution_id(data)
    refs: list[str] = []
    blobs_by_hash: dict[str, BlobCandidate] = {}
    for message in messages:
        message_payload = canonical_json_bytes(message)
        message_hash = canonical_json_hash_from_bytes(message_payload)
        message_bytes = len(message_payload)
        _remember_blob_candidate(
            blobs_by_hash,
            blob_hash=message_hash,
            data=message,
            payload_bytes=message_bytes,
            collision_message=(
                f"trace message blob hash collision for task {task_id}: {message_hash}"
            ),
        )
        refs.append(message_hash)

    _upsert_message_blobs(
        db,
        task_id=task_id,
        execution_id=execution_id,
        blobs_by_hash=blobs_by_hash,
    )

    encoded_context["messages"] = {
        "__encoding": MESSAGE_REFS_ENCODING,
        "count": len(refs),
        "hash": canonical_json_hash(refs),
        "refs": refs,
    }


def encode_checkpoint_blob_fields_for_storage(
    db: Session,
    *,
    task_id: int,
    data: Any,
) -> Any:
    """Replace selected checkpoint fields with blob refs."""

    fields_to_encode = _checkpoint_blob_fields_to_encode(data)
    if not fields_to_encode:
        return data

    encoded = copy.deepcopy(data)
    _encode_checkpoint_blob_fields_in_place(
        db,
        task_id=task_id,
        data=encoded,
        fields_to_encode=fields_to_encode,
    )
    return encoded


def _checkpoint_blob_fields_to_encode(
    data: Any,
) -> list[tuple[CheckpointBlobField, Any]]:
    if not _is_readable_checkpoint_data(data):
        return []

    fields_to_encode: list[tuple[CheckpointBlobField, Any]] = []
    for field in CHECKPOINT_BLOB_FIELDS:
        payload = _get_nested(data, field.path)
        if _should_store_checkpoint_blob_payload(payload):
            fields_to_encode.append((field, payload))
    return fields_to_encode


def _checkpoint_execution_id(data: dict[str, Any]) -> str:
    return str(
        data.get("root_execution_id")
        or data.get("execution_id")
        or _get_nested(data, ("snapshot", "execution_id"))
        or ""
    )


def _encode_checkpoint_blob_fields_in_place(
    db: Session,
    *,
    task_id: int,
    data: dict[str, Any],
    fields_to_encode: list[tuple[CheckpointBlobField, Any]],
) -> None:
    execution_id = _checkpoint_execution_id(data)

    blobs_by_ref: dict[tuple[str, str], BlobCandidate] = {}
    refs_by_field: dict[CheckpointBlobField, str] = {}
    for field, payload in fields_to_encode:
        blob_payload = canonical_json_bytes(payload)
        blob_hash = canonical_json_hash_from_bytes(blob_payload)
        _remember_blob_candidate(
            blobs_by_ref,
            blob_hash=(field.kind, blob_hash),
            data=payload,
            payload_bytes=len(blob_payload),
            collision_message=(
                f"trace checkpoint blob hash collision for task {task_id}: "
                f"{field.kind} {blob_hash}"
            ),
        )
        refs_by_field[field] = blob_hash

    _upsert_checkpoint_blobs(
        db,
        task_id=task_id,
        execution_id=execution_id,
        blobs_by_ref=blobs_by_ref,
    )

    for field, _payload in fields_to_encode:
        blob_hash = refs_by_field[field]
        _set_nested(
            data,
            field.path,
            {
                "__encoding": CHECKPOINT_BLOB_REF_ENCODING,
                "kind": field.kind,
                "hash": blob_hash,
            },
        )


def decode_trace_event_data(
    db: Session,
    *,
    task_id: int,
    data: Any,
    strict: bool = False,
    verify_blob_hashes: bool = True,
) -> Any:
    """Decode checkpoint message refs inside trace event data.

    Non-checkpoint data and old inline-list checkpoints pass through unchanged.
    With ``strict=False``, decode failures preserve the stored refs and attach a
    decode error marker for debug/raw API callers.
    """

    try:
        return _decode_trace_event_data(
            db,
            task_id=task_id,
            data=data,
            lookup=None,
            verify_blob_hashes=verify_blob_hashes,
        )
    except CheckpointMessageDecodeError as exc:
        if strict:
            raise
        if not isinstance(data, dict):
            return data
        fallback = copy.deepcopy(data)
        fallback[MESSAGE_REFS_DECODE_ERROR_KEY] = str(exc)
        return fallback


def decode_trace_events_data(
    db: Session,
    *,
    task_id: int,
    data_items: list[Any],
    strict: bool = False,
    verify_blob_hashes: bool | None = None,
) -> list[Any]:
    """Decode checkpoint refs for many trace event payloads with bulk blob loads."""

    if verify_blob_hashes is None:
        verify_blob_hashes = strict

    # A failed bulk blob lookup must not drop the whole task's events: under
    # strict mode surface it, otherwise degrade to no-lookup so each item still
    # decodes (blob-backed ones fall back per-item below).
    try:
        lookup = _load_trace_blob_lookup(db, task_id=task_id, data_items=data_items)
    except Exception:
        if strict:
            raise
        logger.warning(
            "Bulk trace-blob lookup failed for task %s; decoding without blobs",
            task_id,
            exc_info=True,
        )
        lookup = None
    decoded_items: list[Any] = []
    for data in data_items:
        try:
            decoded_items.append(
                _decode_trace_event_data(
                    db,
                    task_id=task_id,
                    data=data,
                    lookup=lookup,
                    verify_blob_hashes=verify_blob_hashes,
                )
            )
        except CheckpointMessageDecodeError as exc:
            if strict:
                raise
            if not isinstance(data, dict):
                decoded_items.append(data)
                continue
            fallback = copy.deepcopy(data)
            fallback[MESSAGE_REFS_DECODE_ERROR_KEY] = str(exc)
            decoded_items.append(fallback)
    return decoded_items


def _decode_trace_event_data(
    db: Session,
    *,
    task_id: int,
    data: Any,
    lookup: TraceBlobLookup | None,
    verify_blob_hashes: bool,
) -> Any:
    decoded = data

    messages_payload = _get_checkpoint_messages_payload(data)
    if isinstance(messages_payload, list):
        pass
    elif (
        isinstance(messages_payload, dict)
        and messages_payload.get("__encoding") == MESSAGE_REFS_ENCODING
        and not _is_message_refs_payload(messages_payload)
    ):
        raise CheckpointMessageDecodeError(
            "checkpoint message refs payload is malformed"
        )
    elif _is_message_refs_payload(messages_payload):
        refs = messages_payload["refs"]
        expected_count = messages_payload["count"]
        if expected_count != len(refs):
            raise CheckpointMessageDecodeError(
                f"checkpoint message refs count mismatch: expected {expected_count}, got {len(refs)}"
            )

        expected_sequence_hash = messages_payload["hash"]
        actual_sequence_hash = canonical_json_hash(refs)
        if expected_sequence_hash != actual_sequence_hash:
            raise CheckpointMessageDecodeError(
                "checkpoint message refs sequence hash mismatch"
            )

        messages = _load_messages_by_refs(
            db,
            task_id=task_id,
            refs=refs,
            lookup=lookup,
            verify_blob_hashes=verify_blob_hashes,
        )
        decoded = _ensure_decoded_copy(data, decoded)
        decoded["snapshot"]["context"]["messages"] = messages

    for field in CHECKPOINT_BLOB_FIELDS:
        payload = _get_nested(decoded, field.path)
        if (
            isinstance(payload, dict)
            and payload.get("__encoding") == CHECKPOINT_BLOB_REF_ENCODING
            and not _is_checkpoint_blob_ref_payload(payload)
        ):
            raise CheckpointMessageDecodeError(
                f"checkpoint blob refs payload is malformed for {field.kind}"
            )
        if not _is_checkpoint_blob_ref_payload(payload):
            continue
        if payload["kind"] != field.kind:
            raise CheckpointMessageDecodeError(
                f"checkpoint blob refs kind mismatch: expected {field.kind}, got {payload['kind']}"
            )

        blob_data = _load_checkpoint_blob_by_ref(
            db,
            task_id=task_id,
            blob_kind=field.kind,
            blob_hash=payload["hash"],
            lookup=lookup,
            verify_blob_hashes=verify_blob_hashes,
        )
        decoded = _ensure_decoded_copy(data, decoded)
        _set_nested(decoded, field.path, blob_data)
    return decoded


def _is_readable_checkpoint_data(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and data.get("checkpoint_type") in READABLE_CHECKPOINT_TYPES
    )


def _checkpoint_storage_field_states(data: Any) -> list[Literal["inline", "refs"]]:
    states: list[Literal["inline", "refs"]] = []
    messages_state = get_checkpoint_messages_storage_state(data)
    if messages_state == "inline":
        states.append("inline")
    elif messages_state == "refs":
        states.append("refs")

    if not _is_readable_checkpoint_data(data):
        return states

    for field in CHECKPOINT_BLOB_FIELDS:
        payload = _get_nested(data, field.path)
        if _is_checkpoint_blob_ref_payload(payload):
            states.append("refs")
        elif _should_store_checkpoint_blob_payload(payload):
            states.append("inline")
    return states


def _get_checkpoint_messages_payload(data: Any) -> Any:
    if not _is_readable_checkpoint_data(data):
        return None

    snapshot = data.get("snapshot")
    if not isinstance(snapshot, dict):
        return None
    context = snapshot.get("context")
    if not isinstance(context, dict):
        return None
    return context.get("messages")


def _get_nested(data: Any, path: tuple[str, ...]) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: dict[str, Any] = data
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def _is_message_refs_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("__encoding") != MESSAGE_REFS_ENCODING:
        return False
    return (
        isinstance(value.get("refs"), list)
        and isinstance(value.get("count"), int)
        and isinstance(value.get("hash"), str)
    )


def _is_checkpoint_blob_ref_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("__encoding") != CHECKPOINT_BLOB_REF_ENCODING:
        return False
    return isinstance(value.get("kind"), str) and isinstance(value.get("hash"), str)


def _should_store_checkpoint_blob_payload(value: Any) -> bool:
    if _is_checkpoint_blob_ref_payload(value):
        return False
    return isinstance(value, dict) and bool(value)


def _ensure_decoded_copy(original: Any, current: Any) -> Any:
    if current is original:
        return copy.deepcopy(original)
    return current


def _chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _sql_in_clause_chunk_size(
    db: Session,
    *,
    reserved_binds: int,
) -> int:
    bind = db.get_bind()
    if bind is None or bind.dialect.name != "sqlite":
        return SQL_IN_CLAUSE_CHUNK_SIZE

    try:
        import sqlite3

        raw_connection = db.connection().connection
        driver_connection = (
            getattr(raw_connection, "driver_connection", None)
            or getattr(raw_connection, "connection", None)
            or raw_connection
        )
        limit = int(driver_connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER))
    except Exception:
        return SQL_IN_CLAUSE_CHUNK_SIZE

    return max(1, min(SQL_IN_CLAUSE_CHUNK_SIZE, limit - reserved_binds))


def _load_trace_blob_lookup(
    db: Session,
    *,
    task_id: int,
    data_items: list[Any],
) -> TraceBlobLookup:
    message_refs: set[str] = set()
    checkpoint_refs: set[tuple[str, str]] = set()

    for data in data_items:
        item_message_refs, item_checkpoint_refs = _collect_trace_blob_refs(data)
        message_refs.update(item_message_refs)
        checkpoint_refs.update(item_checkpoint_refs)

    message_data_by_hash: dict[str, Any] = {}
    if message_refs:
        chunk_size = _sql_in_clause_chunk_size(db, reserved_binds=1)
        for refs_chunk in _chunks(sorted(message_refs), chunk_size):
            rows = (
                db.query(TraceMessageBlob)
                .filter(
                    TraceMessageBlob.task_id == task_id,
                    TraceMessageBlob.message_hash.in_(refs_chunk),
                )
                .all()
            )
            message_data_by_hash.update(
                {str(row.message_hash): row.message_data for row in rows}
            )

    checkpoint_data_by_ref: dict[tuple[str, str], Any] = {}
    if checkpoint_refs:
        refs_by_kind: dict[str, set[str]] = {}
        for kind, blob_hash in checkpoint_refs:
            refs_by_kind.setdefault(kind, set()).add(blob_hash)

        for kind, blob_hashes in refs_by_kind.items():
            chunk_size = _sql_in_clause_chunk_size(db, reserved_binds=2)
            for hash_chunk in _chunks(sorted(blob_hashes), chunk_size):
                rows = (
                    db.query(TraceCheckpointBlob)
                    .filter(
                        TraceCheckpointBlob.task_id == task_id,
                        TraceCheckpointBlob.blob_kind == kind,
                        TraceCheckpointBlob.blob_hash.in_(hash_chunk),
                    )
                    .all()
                )
                checkpoint_data_by_ref.update(
                    {
                        (str(row.blob_kind), str(row.blob_hash)): row.blob_data
                        for row in rows
                    }
                )
    return TraceBlobLookup(
        message_data_by_hash=message_data_by_hash,
        checkpoint_data_by_ref=checkpoint_data_by_ref,
    )


def _collect_trace_blob_refs(data: Any) -> tuple[set[str], set[tuple[str, str]]]:
    message_refs: set[str] = set()
    checkpoint_refs: set[tuple[str, str]] = set()

    messages_payload = _get_checkpoint_messages_payload(data)
    if _is_message_refs_payload(messages_payload):
        message_refs.update(
            ref for ref in messages_payload["refs"] if isinstance(ref, str)
        )

    for field in CHECKPOINT_BLOB_FIELDS:
        payload = _get_nested(data, field.path)
        if not _is_checkpoint_blob_ref_payload(payload):
            continue
        if payload["kind"] != field.kind:
            continue
        blob_hash = payload["hash"]
        if isinstance(blob_hash, str):
            checkpoint_refs.add((field.kind, blob_hash))

    return message_refs, checkpoint_refs


def _load_messages_by_refs(
    db: Session,
    *,
    task_id: int,
    refs: list[str],
    lookup: TraceBlobLookup | None,
    verify_blob_hashes: bool,
) -> list[Any]:
    if not refs:
        return []
    if any(not isinstance(ref, str) for ref in refs):
        raise CheckpointMessageDecodeError(
            "checkpoint message refs contain non-string hash"
        )

    unique_refs = sorted(set(refs))
    if lookup is None:
        by_hash: dict[str, Any] = {}
        chunk_size = _sql_in_clause_chunk_size(db, reserved_binds=1)
        for refs_chunk in _chunks(unique_refs, chunk_size):
            rows = (
                db.query(TraceMessageBlob)
                .filter(
                    TraceMessageBlob.task_id == task_id,
                    TraceMessageBlob.message_hash.in_(refs_chunk),
                )
                .all()
            )
            by_hash.update({str(row.message_hash): row.message_data for row in rows})
    else:
        by_hash = lookup.message_data_by_hash

    missing = [ref for ref in unique_refs if ref not in by_hash]
    if missing:
        raise CheckpointMessageDecodeError(
            f"checkpoint message blobs missing for {len(missing)} refs"
        )

    messages: list[Any] = []
    for ref in refs:
        message = by_hash[ref]
        if verify_blob_hashes and canonical_json_hash(message) != ref:
            raise CheckpointMessageDecodeError("checkpoint message blob hash mismatch")
        messages.append(copy.deepcopy(message))
    return messages


def _load_checkpoint_blob_by_ref(
    db: Session,
    *,
    task_id: int,
    blob_kind: str,
    blob_hash: str,
    lookup: TraceBlobLookup | None,
    verify_blob_hashes: bool,
) -> Any:
    if lookup is None:
        row = (
            db.query(TraceCheckpointBlob)
            .filter(
                TraceCheckpointBlob.task_id == task_id,
                TraceCheckpointBlob.blob_kind == blob_kind,
                TraceCheckpointBlob.blob_hash == blob_hash,
            )
            .first()
        )
        blob_data = row.blob_data if row is not None else None
    else:
        blob_data = lookup.checkpoint_data_by_ref.get((blob_kind, blob_hash))

    if blob_data is None:
        raise CheckpointMessageDecodeError(
            f"checkpoint blob missing for kind {blob_kind}: {blob_hash}"
        )
    if verify_blob_hashes and canonical_json_hash(blob_data) != blob_hash:
        raise CheckpointMessageDecodeError("checkpoint blob hash mismatch")
    return copy.deepcopy(blob_data)


def _remember_blob_candidate(
    candidates: dict[Any, BlobCandidate],
    *,
    blob_hash: Any,
    data: Any,
    payload_bytes: int,
    collision_message: str,
) -> None:
    existing = candidates.get(blob_hash)
    if existing is not None:
        if existing.payload_bytes != payload_bytes or existing.data != data:
            raise ValueError(collision_message)
        return
    candidates[blob_hash] = BlobCandidate(data=data, payload_bytes=payload_bytes)


def _pending_message_hashes(
    db: Session,
    *,
    task_id: int,
    blobs_by_hash: dict[str, BlobCandidate],
) -> set[str]:
    pending_hashes: set[str] = set()
    for pending in db.new:
        if not isinstance(pending, TraceMessageBlob):
            continue
        if pending.task_id != task_id:
            continue
        message_hash = str(pending.message_hash)
        candidate = blobs_by_hash.get(message_hash)
        if candidate is None:
            continue
        pending_hashes.add(message_hash)
        if (
            pending.message_bytes != candidate.payload_bytes
            or pending.message_data != candidate.data
        ):
            raise ValueError(
                f"trace message blob hash collision for task {task_id}: {message_hash}"
            )
    return pending_hashes


def _upsert_message_blobs(
    db: Session,
    *,
    task_id: int,
    execution_id: str,
    blobs_by_hash: dict[str, BlobCandidate],
) -> None:
    if not blobs_by_hash:
        return

    pending_hashes = _pending_message_hashes(
        db,
        task_id=task_id,
        blobs_by_hash=blobs_by_hash,
    )
    hashes_to_query = sorted(set(blobs_by_hash) - pending_hashes)
    existing_hashes: set[str] = set()
    chunk_size = _sql_in_clause_chunk_size(db, reserved_binds=1)
    for hashes_chunk in _chunks(hashes_to_query, chunk_size):
        rows = (
            db.query(TraceMessageBlob.message_hash, TraceMessageBlob.message_bytes)
            .filter(
                TraceMessageBlob.task_id == task_id,
                TraceMessageBlob.message_hash.in_(hashes_chunk),
            )
            .all()
        )
        for message_hash, message_bytes in rows:
            message_hash = str(message_hash)
            candidate = blobs_by_hash[message_hash]
            if message_bytes != candidate.payload_bytes:
                raise ValueError(
                    f"trace message blob hash collision for task {task_id}: "
                    f"{message_hash}"
                )
            existing_hashes.add(message_hash)

    for message_hash, candidate in blobs_by_hash.items():
        if message_hash in pending_hashes or message_hash in existing_hashes:
            continue
        db.add(
            TraceMessageBlob(
                task_id=task_id,
                execution_id=execution_id,
                message_hash=message_hash,
                message_data=copy.deepcopy(candidate.data),
                message_bytes=candidate.payload_bytes,
            )
        )


def _pending_checkpoint_blob_refs(
    db: Session,
    *,
    task_id: int,
    blobs_by_ref: dict[tuple[str, str], BlobCandidate],
) -> set[tuple[str, str]]:
    pending_refs: set[tuple[str, str]] = set()
    for pending in db.new:
        if not isinstance(pending, TraceCheckpointBlob):
            continue
        if pending.task_id != task_id:
            continue
        blob_ref = (str(pending.blob_kind), str(pending.blob_hash))
        candidate = blobs_by_ref.get(blob_ref)
        if candidate is None:
            continue
        pending_refs.add(blob_ref)
        if (
            pending.blob_bytes != candidate.payload_bytes
            or pending.blob_data != candidate.data
        ):
            raise ValueError(
                f"trace checkpoint blob hash collision for task {task_id}: "
                f"{blob_ref[0]} {blob_ref[1]}"
            )
    return pending_refs


def _upsert_checkpoint_blobs(
    db: Session,
    *,
    task_id: int,
    execution_id: str,
    blobs_by_ref: dict[tuple[str, str], BlobCandidate],
) -> None:
    if not blobs_by_ref:
        return

    pending_refs = _pending_checkpoint_blob_refs(
        db,
        task_id=task_id,
        blobs_by_ref=blobs_by_ref,
    )
    refs_to_query = sorted(set(blobs_by_ref) - pending_refs)
    existing_refs: set[tuple[str, str]] = set()
    if refs_to_query:
        refs_by_kind: dict[str, set[str]] = {}
        for kind, blob_hash in refs_to_query:
            refs_by_kind.setdefault(kind, set()).add(blob_hash)

        for kind, blob_hashes in refs_by_kind.items():
            chunk_size = _sql_in_clause_chunk_size(db, reserved_binds=2)
            for hash_chunk in _chunks(sorted(blob_hashes), chunk_size):
                rows = (
                    db.query(
                        TraceCheckpointBlob.blob_kind,
                        TraceCheckpointBlob.blob_hash,
                        TraceCheckpointBlob.blob_bytes,
                    )
                    .filter(
                        TraceCheckpointBlob.task_id == task_id,
                        TraceCheckpointBlob.blob_kind == kind,
                        TraceCheckpointBlob.blob_hash.in_(hash_chunk),
                    )
                    .all()
                )
                for blob_kind, blob_hash, blob_bytes in rows:
                    blob_ref = (str(blob_kind), str(blob_hash))
                    candidate = blobs_by_ref.get(blob_ref)
                    if candidate is None:
                        continue
                    if blob_bytes != candidate.payload_bytes:
                        raise ValueError(
                            f"trace checkpoint blob hash collision for task {task_id}: "
                            f"{blob_ref[0]} {blob_ref[1]}"
                        )
                    existing_refs.add(blob_ref)

    for blob_ref, candidate in blobs_by_ref.items():
        if blob_ref in pending_refs or blob_ref in existing_refs:
            continue
        blob_kind, blob_hash = blob_ref
        db.add(
            TraceCheckpointBlob(
                task_id=task_id,
                execution_id=execution_id,
                blob_kind=blob_kind,
                blob_hash=blob_hash,
                blob_data=copy.deepcopy(candidate.data),
                blob_bytes=candidate.payload_bytes,
            )
        )
