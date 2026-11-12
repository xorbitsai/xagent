"""Offline re-embedding migration for the LanceDB memory store.

Recomputes every note's vector under one target identity into a staging table,
verifies it against the live table, and only then replaces the live table in a
single atomic overwrite. A config change alone cannot do this: when the old and
new embeddings share a dimension, nothing else in the store notices.

Invariant: the live table is replaced only by a table whose every note with
text carries a vector produced by ``target_identity`` (notes with blank text
carry none, as on the write path), which declares that space, and whose
ids and row count match the source. Any shortfall leaves the live table alone;
the previous vectors stay in the backup table for rollback. Writers must be
quiesced (``docs/deployment.md``): the routine carries rows written before its
last version check and refuses to swap while they keep arriving, but a write
that commits during the swap's own table commit is not carried: no lock in the
write path closes that window, so it is detected and reported instead.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

import pyarrow as pa  # type: ignore
from filelock import FileLock, Timeout

from ..model.model import EmbeddingModelConfig
from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from ..utils.lancedb_sql import sql_string_in
from .lancedb_maintenance import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCK_TIMEOUT,
    MaintenanceLockTimeout,
    lancedb_lock_path,
    validate_lock_timeout,
)
from .scope_columns import SCOPE_DIMS_COLUMN, USER_ID_COLUMN, strict_scope_columns
from .vector_compatibility import (
    EmbeddingIdentity,
    VectorCompatibility,
    _is_fully_admitted,
    _lancedb_table_exists,
    _prepared_schema,
    canonical_embedding_identity,
    classify_vector_compatibility,
    embedding_identity_fingerprint,
)

logger = logging.getLogger(__name__)

STAGING_SUFFIX = "__migrate_"
BACKUP_SUFFIX = "__pre_migrate_"
_SOURCE_COLUMNS = ["id", "text", "metadata"]
_ID_CHUNK = 256


class MemoryReembedStatus(str, Enum):
    ABSENT = "absent"
    ALREADY_TARGET = "already_target"
    COMPLETE = "complete"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    INVALID_SOURCE_DATA = "invalid_source_data"
    EMBEDDING_FAILED = "embedding_failed"
    VERIFICATION_FAILED = "verification_failed"
    NOT_CONVERGED = "not_converged"
    SWAPPED_OVER_A_WRITE = "swapped_over_a_write"


@dataclass(frozen=True)
class MemoryReembedOutcome:
    status: MemoryReembedStatus
    source_rows: int = 0
    reembedded_rows: int = 0
    passes: int = 0
    staging_table: str | None = None
    backup_table: str | None = None
    detail: str | None = None


class _InvalidSource(ValueError):
    """The live or staging table holds a row this routine must not rewrite."""


class _EmbeddingFailed(RuntimeError):
    """The target model did not return one valid vector per text."""


def _checkpoint(_stage: str, _batch: int | None = None) -> None:
    """Test seam for deterministic interruption; intentionally does nothing."""


def staging_table_name(table_name: str, identity: EmbeddingIdentity) -> str:
    fingerprint = embedding_identity_fingerprint(identity)
    return f"{table_name}{STAGING_SUFFIX}{fingerprint[:16]}"


def backup_table_name(table_name: str, identity: EmbeddingIdentity) -> str:
    fingerprint = embedding_identity_fingerprint(identity)
    return f"{table_name}{BACKUP_SUFFIX}{fingerprint[:16]}"


def _source_schema_supported(schema: Any) -> bool:
    if any(
        name not in schema.names or schema.field(name).type != pa.string()
        for name in _SOURCE_COLUMNS
    ):
        return False
    expected = {USER_ID_COLUMN: pa.int64(), SCOPE_DIMS_COLUMN: pa.list_(pa.string())}
    if any(
        name in schema.names and schema.field(name).type != kind
        for name, kind in expected.items()
    ):
        return False
    if "vector" not in schema.names:
        return True
    kind = schema.field("vector").type
    return bool(pa.types.is_fixed_size_list(kind) and kind.value_type == pa.float32())


def _vectors(texts: list[str], encode: Callable, dimension: int) -> list[list[float]]:
    if not texts:
        return []
    try:
        result = encode(texts)
    except Exception as exc:
        raise _EmbeddingFailed(f"encoding {len(texts)} notes failed: {exc}") from exc
    if not isinstance(result, list) or len(result) != len(texts):
        raise _EmbeddingFailed(f"encoding {len(texts)} notes returned a bad shape")
    vectors = []
    for vector in result:
        if not isinstance(vector, (list, tuple)) or len(vector) != dimension:
            raise _EmbeddingFailed(f"a note did not produce {dimension} dimensions")
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise _EmbeddingFailed(f"a re-embedded vector is not numeric: {exc}")
        if not all(math.isfinite(value) for value in values):
            raise _EmbeddingFailed("a re-embedded vector holds a non-finite value")
        vectors.append(values)
    return vectors


def _blank_text(text: str | None) -> bool:
    """Whether a note carries no text, and so no vector, by design."""
    return not (text or "").strip()


def _fill_staging(
    source: Any,
    staging: Any,
    encode: Callable,
    *,
    batch_size: int,
    dimension: int,
    seen: sqlite3.Connection,
) -> tuple[int, int]:
    """Copy every source row the staging table is missing or holds stale."""
    rows = embedded = 0
    for index, batch in enumerate(
        source.search().select(_SOURCE_COLUMNS).to_batches(batch_size=batch_size), 1
    ):
        records = []
        for record in batch.to_pylist():
            note_id = record["id"]
            if not isinstance(note_id, str) or not note_id:
                raise _InvalidSource("note ids must be non-empty strings")
            try:
                seen.execute("INSERT INTO source_ids VALUES (?)", (note_id,))
            except sqlite3.IntegrityError:
                raise _InvalidSource("note ids must be unique")
            metadata = record["metadata"]
            if metadata is not None and not isinstance(metadata, str):
                raise _InvalidSource("note metadata must be a string or SQL NULL")
            try:
                owner, scope_dims = strict_scope_columns(metadata)
            except ValueError as exc:
                raise _InvalidSource(f"note {note_id!r}: {exc}")
            records.append((note_id, record["text"], metadata, owner, scope_dims))
        rows += len(records)
        ids = [record[0] for record in records]
        blank_ids = {record[0] for record in records if _blank_text(record[1])}
        present: dict[str, tuple] = {}
        staged: set[str] = set()
        for offset in range(0, len(ids), _ID_CHUNK):
            chunk = ids[offset : offset + _ID_CHUNK]
            found = (
                staging.search()
                .where(sql_string_in("id", chunk))
                .select(["id", "text", "metadata", USER_ID_COLUMN, SCOPE_DIMS_COLUMN])
                .to_arrow()
            )
            for row in found.to_pylist():
                staged.add(row["id"])
                present[row["id"]] = (
                    row["text"],
                    row["metadata"],
                    row[USER_ID_COLUMN],
                    row[SCOPE_DIMS_COLUMN],
                )
            # A staged row with no vector is not reusable however well it
            # matches: only a rewrite can give it one, so it counts as absent.
            # Blank text carries no vector by contract, so it is exempt.
            missing_vector = (
                staging.search()
                .where(f"({sql_string_in('id', chunk)}) AND vector IS NULL")
                .select(["id"])
                .to_arrow()
            )
            for row in missing_vector.to_pylist():
                if row["id"] not in blank_ids:
                    present.pop(row["id"], None)
        todo = [
            record
            for record in records
            if present.get(record[0])
            != (record[1] or "", record[2], record[3], record[4])
        ]
        if not todo:
            _checkpoint("batch_skipped", index)
            continue
        stale = [record[0] for record in todo if record[0] in staged]
        if stale:
            staging.delete(sql_string_in("id", stale))
        # Blank text is stored without a vector, exactly like the write path.
        pending = [record for record in todo if not _blank_text(record[1])]
        encoded = iter(
            _vectors([record[1] or "" for record in pending], encode, dimension)
        )
        vectors = [None if _blank_text(record[1]) else next(encoded) for record in todo]
        staging.add(
            pa.Table.from_pylist(
                [
                    {
                        "id": record[0],
                        "text": record[1],
                        "metadata": record[2],
                        "vector": vector,
                        USER_ID_COLUMN: record[3],
                        SCOPE_DIMS_COLUMN: record[4],
                    }
                    for record, vector in zip(todo, vectors)
                ],
                schema=staging.schema,
            )
        )
        embedded += len(pending)
        _checkpoint("batch_filled", index)
    return rows, embedded


def _verify(
    connection: Any,
    staging_name: str,
    *,
    identity: EmbeddingIdentity,
    source_rows: int,
    seen: sqlite3.Connection,
    batch_size: int,
) -> str | None:
    """Why this staging table must not replace the source, or ``None``."""
    staging = connection.open_table(staging_name)
    try:
        seen.execute("CREATE TABLE staging_ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
        try:
            for batch in (
                staging.search().select(["id"]).to_batches(batch_size=batch_size)
            ):
                for (note_id,) in zip(batch.column(0).to_pylist()):
                    try:
                        seen.execute("INSERT INTO staging_ids VALUES (?)", (note_id,))
                    except sqlite3.IntegrityError:
                        raise _InvalidSource("the staging table holds a duplicate id")
            missing = [
                row[0]
                for row in seen.execute(
                    "SELECT id FROM source_ids "
                    "WHERE id NOT IN (SELECT id FROM staging_ids)"
                )
            ]
            extra = [
                row[0]
                for row in seen.execute(
                    "SELECT id FROM staging_ids "
                    "WHERE id NOT IN (SELECT id FROM source_ids)"
                )
            ]
        finally:
            seen.execute("DROP TABLE staging_ids")
        if extra:
            staging.delete(sql_string_in("id", extra))
            _checkpoint("staging_pruned")
        if missing:
            return f"{len(missing)} source notes are missing from the staging table"
        if staging.count_rows() != source_rows:
            return (
                f"the staging table holds {staging.count_rows()} rows, "
                f"the source {source_rows}"
            )
        # SQL trim keeps newlines and tabs, so blankness is judged like the writer.
        offenders = sum(
            not _blank_text(text)
            for batch in staging.search()
            .where("vector IS NULL")
            .select(["text"])
            .to_batches(batch_size=batch_size)
            for text in batch.column(0).to_pylist()
        )
        if offenders:
            return f"{offenders} staged notes carry no vector"
        if (
            classify_vector_compatibility(staging.schema, identity)
            is not VectorCompatibility.MATCHING
        ):
            return "the staging table does not declare the target vector space"
        if not _is_fully_admitted(staging):
            return "the staging table is missing its admission certificate"
        return None
    finally:
        _safe_close_table(staging)


def _spill(connection: Any, source_name: str, scratch: str, batch_size: int) -> tuple:
    """Copy ``source_name`` to an Arrow IPC file; return its path and schema."""
    staged = os.path.join(scratch, f"spill-{source_name}.arrow")
    source = connection.open_table(source_name)
    try:
        schema = source.schema
        with pa.OSFile(staged, "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
            for batch in source.search().to_batches(batch_size=batch_size):
                writer.write_batch(batch)
    finally:
        _safe_close_table(source)
    return staged, schema


def _replace_from(
    connection: Any,
    target_name: str,
    spilled: str,
    schema: Any,
    *,
    on_commit: Callable[[], None] | None = None,
) -> int:
    """Replace ``target_name`` with the rows at ``spilled``; return its new version.

    ``on_commit`` runs once the overwrite has committed, before anything below
    it can fail.
    """
    with pa.memory_map(spilled, "r") as mapped:
        reader = pa.ipc.open_file(mapped)

        def batches() -> Any:
            for index in range(reader.num_record_batches):
                yield reader.get_batch(index)

        # Supported versions pull the first batch before they honour ``schema``,
        # so an empty generator raises instead of writing the zero-row table.
        data: Any = (
            batches()
            if reader.num_record_batches
            else pa.Table.from_batches([], schema)
        )
        written = connection.create_table(
            target_name, data=data, schema=schema, mode="overwrite"
        )
        if on_commit is not None:
            on_commit()
        try:
            version = int(written.version)
        finally:
            _safe_close_table(written)
    os.unlink(spilled)
    return version


def _copy_table(
    connection: Any,
    source_name: str,
    target_name: str,
    *,
    scratch: str,
    batch_size: int,
) -> None:
    """Atomically replace ``target_name`` with a copy of ``source_name``.

    Rows spill through an Arrow IPC file: writing from a live table's lazy scan
    deadlocks this LanceDB version, and neither table fits in memory.
    """
    _replace_from(
        connection, target_name, *_spill(connection, source_name, scratch, batch_size)
    )


def _backfill_vectors(
    connection: Any,
    table_name: str,
    encode: Callable,
    *,
    batch_size: int,
    dimension: int,
) -> int:
    """Embed in place the notes with text but no vector; return how many."""
    reader = connection.open_table(table_name)
    writer = None
    filled = 0
    try:
        writer = connection.open_table(table_name)
        for batch in (
            reader.search().where("vector IS NULL").to_batches(batch_size=batch_size)
        ):
            rows = [row for row in batch.to_pylist() if not _blank_text(row["text"])]
            if not rows:
                continue
            for row, vector in zip(
                rows, _vectors([row["text"] for row in rows], encode, dimension)
            ):
                row["vector"] = vector
            # Only rows a writer has not touched since they were read are filled.
            result = (
                writer.merge_insert("id")
                .when_matched_update_all(
                    where="target.vector IS NULL AND target.text = source.text AND "
                    "((target.metadata IS NULL AND source.metadata IS NULL) "
                    "OR target.metadata = source.metadata)"
                )
                .execute(pa.Table.from_pylist(rows, schema=reader.schema))
            )
            filled += int(result.num_updated_rows)
    finally:
        _safe_close_table(reader)
        _safe_close_table(writer)
    return filled


def _current_version(connection: Any, table_name: str) -> int:
    """The table's version on disk; an open handle stays pinned to its own."""
    table = connection.open_table(table_name)
    try:
        return int(table.version)
    finally:
        _safe_close_table(table)


def _staging_table(
    connection: Any, name: str, source_schema: Any, identity: EmbeddingIdentity
) -> Any:
    """The staging table for this target, ready to be filled."""
    if not _lancedb_table_exists(connection, name):
        # ``_prepared_schema`` keeps an existing vector column as-is, so the
        # source's is dropped: this table's vector takes the target width.
        return connection.create_table(
            name,
            schema=_prepared_schema(
                pa.schema(
                    [field for field in source_schema if field.name != "vector"],
                    metadata=source_schema.metadata,
                ),
                identity,
                1,
            ),
        )
    staging = connection.open_table(name)
    if classify_vector_compatibility(
        staging.schema, identity
    ) is VectorCompatibility.MATCHING and _is_fully_admitted(staging):
        return staging
    _safe_close_table(staging)
    raise _InvalidSource(
        f"staging table {name!r} does not hold the target vector space; "
        "remove it before retrying"
    )


def reembed_memory_table(
    connection: Any,
    table_name: str,
    target_identity: EmbeddingIdentity | EmbeddingModelConfig | Mapping[str, Any],
    *,
    encode: Callable[[list[str]], list[list[float]]],
    batch_size: int = DEFAULT_BATCH_SIZE,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
    max_passes: int = 4,
) -> MemoryReembedOutcome:
    """Recompute every note's vector under ``target_identity`` and swap it in.

    Resumable: the staging table persists across interruptions, and a rerun
    only embeds notes it is missing or whose text changed; on a table already in
    the target space it only embeds, in place, notes left without a vector.
    Notes with blank text carry no vector, as on the write path. Before the swap it
    must match the source's ids and row count and declare the target space; on
    any shortfall the live table keeps the vectors it had, and the previous
    table stays behind as a backup for rollback.

    Args:
        connection: A LanceDB connection to a writable local database.
        table_name: The live memory table, usually ``memories``.
        target_identity: The vector space every note is re-embedded into.
        encode: One batch of texts to one vector per text, in order, each with
            ``target_identity.dimension`` floats.
        max_passes: Catch-up passes allowed while writers keep committing.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(max_passes) is not int or max_passes <= 0:
        raise ValueError("max_passes must be a positive integer")
    validate_lock_timeout(lock_timeout)
    identity = canonical_embedding_identity(target_identity)
    staging_name = staging_table_name(table_name, identity)
    lock_path = lancedb_lock_path(connection, table_name, "maintenance")
    lock = FileLock(lock_path, timeout=lock_timeout)
    try:
        lock.acquire()
    except Timeout as exc:
        raise MaintenanceLockTimeout(
            f"Timed out after {lock_timeout}s waiting for the memory maintenance "
            f"lock on table {table_name!r}; stop the other maintenance process"
        ) from exc

    def stop(status: MemoryReembedStatus, **fields: Any) -> MemoryReembedOutcome:
        return MemoryReembedOutcome(status, staging_table=staging_name, **fields)

    swapped = False

    def mark_swapped() -> None:
        nonlocal swapped
        swapped = True

    source = staging = None
    scratch = ""
    # A backup only means "the previous table" once the swap committed; any
    # other exit drops the one this run wrote, and every abort leaves live intact.
    backup_written = None
    try:
        if not _lancedb_table_exists(connection, table_name):
            return stop(MemoryReembedStatus.ABSENT)
        source = connection.open_table(table_name)
        if (
            classify_vector_compatibility(source.schema, identity)
            is VectorCompatibility.MATCHING
        ):
            rows = source.count_rows()
            supported = _source_schema_supported(source.schema)
            _safe_close_table(source)
            source = None
            if _lancedb_table_exists(connection, staging_name):
                connection.drop_table(staging_name)
            if not supported:
                return stop(MemoryReembedStatus.ALREADY_TARGET, source_rows=rows)
            try:
                filled = _backfill_vectors(
                    connection,
                    table_name,
                    encode,
                    batch_size=batch_size,
                    dimension=identity.dimension,
                )
            except _EmbeddingFailed as exc:
                return stop(MemoryReembedStatus.EMBEDDING_FAILED, detail=str(exc))
            return stop(
                MemoryReembedStatus.ALREADY_TARGET,
                source_rows=rows,
                reembedded_rows=filled,
            )
        if not _source_schema_supported(source.schema):
            return stop(
                MemoryReembedStatus.INCOMPATIBLE_SCHEMA,
                detail="the live memory table does not have the supported schema",
            )
        scratch = tempfile.mkdtemp(dir=os.path.dirname(lock_path), prefix=".reembed-")
        try:
            staging = _staging_table(connection, staging_name, source.schema, identity)
        except _InvalidSource as exc:
            return stop(MemoryReembedStatus.INCOMPATIBLE_SCHEMA, detail=str(exc))

        source_rows = reembedded = 0
        for pass_no in range(1, max_passes + 1):
            # Handles stay pinned to the version they opened at, so a pass that
            # has to see a write, and verification, both start afresh.
            _safe_close_table(source)
            source = connection.open_table(table_name)
            _safe_close_table(staging)
            staging = connection.open_table(staging_name)
            snapshot = int(source.version)
            seen = sqlite3.connect(os.path.join(scratch, f"ids-{pass_no}.sqlite"))
            try:
                seen.execute(
                    "CREATE TABLE source_ids (id TEXT PRIMARY KEY) WITHOUT ROWID"
                )
                try:
                    source_rows, pass_embedded = _fill_staging(
                        source,
                        staging,
                        encode,
                        batch_size=batch_size,
                        dimension=identity.dimension,
                        seen=seen,
                    )
                    reembedded += pass_embedded
                    detail = _verify(
                        connection,
                        staging_name,
                        identity=identity,
                        source_rows=source_rows,
                        seen=seen,
                        batch_size=batch_size,
                    )
                except _InvalidSource as exc:
                    return stop(
                        MemoryReembedStatus.INVALID_SOURCE_DATA, detail=str(exc)
                    )
                except _EmbeddingFailed as exc:
                    return stop(MemoryReembedStatus.EMBEDDING_FAILED, detail=str(exc))
            finally:
                seen.close()
            if _current_version(connection, table_name) != snapshot:
                # A commit landed after this pass read the source; the next pass
                # sees it. Nothing was swapped, so nothing was lost.
                logger.warning(
                    "Memory table %r changed during re-embedding pass %d",
                    table_name,
                    pass_no,
                )
                continue
            if detail:
                return stop(
                    MemoryReembedStatus.VERIFICATION_FAILED,
                    source_rows=source_rows,
                    reembedded_rows=reembedded,
                    passes=pass_no,
                    detail=detail,
                )
            # Spill the verified staging table before the live table is touched,
            # so the swap itself is one commit and the version check below it is
            # the last thing standing between a writer and being overwritten.
            spilled, target_schema = _spill(
                connection, staging_name, scratch, batch_size
            )
            backup_name = backup_table_name(table_name, identity)
            _copy_table(
                connection,
                table_name,
                backup_name,
                scratch=scratch,
                batch_size=batch_size,
            )
            backup_written = backup_name
            _checkpoint("before_swap")
            if _current_version(connection, table_name) != snapshot:
                logger.warning(
                    "Memory table %r changed while its backup was written",
                    table_name,
                )
                continue
            _safe_close_table(source)
            source = None
            _safe_close_table(staging)
            staging = None
            written = _replace_from(
                connection, table_name, spilled, target_schema, on_commit=mark_swapped
            )
            _checkpoint("after_swap")
            swapped = connection.open_table(table_name)
            try:
                verified = (
                    classify_vector_compatibility(swapped.schema, identity)
                    is VectorCompatibility.MATCHING
                    and swapped.count_rows() == source_rows
                )
            finally:
                _safe_close_table(swapped)
            if not verified:
                raise RuntimeError(
                    "the swapped-in memory table failed its own verification"
                )
            connection.drop_table(staging_name)
            if written != snapshot + 1:
                # The overwrite landed on a later version: a write committed after
                # the last check and is now only in the live table's history.
                detail = (
                    f"a write committed during the swap was overwritten; versions "
                    f"{snapshot + 1}..{written - 1} of {table_name!r} still hold it"
                )
                logger.error("Memory table %r: %s", table_name, detail)
                return stop(
                    MemoryReembedStatus.SWAPPED_OVER_A_WRITE,
                    source_rows=source_rows,
                    reembedded_rows=reembedded,
                    passes=pass_no,
                    backup_table=backup_name,
                    detail=detail,
                )
            logger.info(
                "Re-embedded memory table %r into a new vector space (%d rows, "
                "%d passes); previous table retained as %r",
                table_name,
                source_rows,
                pass_no,
                backup_name,
            )
            return stop(
                MemoryReembedStatus.COMPLETE,
                source_rows=source_rows,
                reembedded_rows=reembedded,
                passes=pass_no,
                backup_table=backup_name,
            )
        return stop(
            MemoryReembedStatus.NOT_CONVERGED,
            source_rows=source_rows,
            reembedded_rows=reembedded,
            passes=max_passes,
            detail=(
                "writers kept committing to the memory table during migration; "
                "the live table still holds the previous vector space"
            ),
        )
    finally:
        _safe_close_table(source)
        _safe_close_table(staging)
        if backup_written and not swapped:
            try:
                connection.drop_table(backup_written)
            except Exception:
                logger.warning("Could not drop the stale backup %r", backup_written)
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        lock.release()
