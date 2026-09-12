"""Read-only vector-space compatibility capabilities for persistent memory."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

import pyarrow as pa  # type: ignore
from filelock import FileLock

from ..model.model import EmbeddingModelConfig
from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from ..tools.core.RAG_tools.utils.lancedb_query_utils import list_table_names
from . import lancedb_maintenance as maintenance
from .scope_columns import SCOPE_DIMS_COLUMN, USER_ID_COLUMN, derive_scope_columns

VECTOR_IDENTITY_METADATA_KEY = b"xagent.memory.vector_space"
DASHSCOPE_DEFAULT_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "text-embedding/text-embedding"
)

_DEFAULT_ENDPOINTS = {
    "dashscope": DASHSCOPE_DEFAULT_ENDPOINT,
    "openai": "https://api.openai.com/v1/embeddings",
    "xinference": "http://localhost:9997",
}
_PROVIDER_ALIASES = {"openai_embedding": "openai", "openai-compatible": "openai"}
_IDENTITY_FIELDS = {"provider", "model", "endpoint", "dimension", "instruct"}
_REQUIRED_SCHEMA = {"id": pa.string(), "text": pa.string(), "metadata": pa.string()}


def _checkpoint(_stage: str, _batch: int | None = None) -> None:
    """Test seam for atomic commit and bounded-batch checks."""


class _ArrowDataType(Protocol):
    """Structural subset used from a PyArrow data type."""

    @property
    def value_type(self) -> object: ...

    @property
    def list_size(self) -> int: ...


class _ArrowField(Protocol):
    """Structural subset used from a PyArrow field."""

    @property
    def type(self) -> _ArrowDataType: ...


class _ArrowSchema(Protocol):
    """Structural PyArrow schema boundary available without importing its type."""

    @property
    def metadata(self) -> Mapping[bytes, bytes] | None: ...

    def field(self, name: str) -> _ArrowField: ...


class VectorCompatibility(str, Enum):
    """Compatibility of persisted vectors with one configured vector space."""

    LEGACY_COMPATIBLE = "legacy_compatible"
    MATCHING = "matching"
    MISMATCHING = "mismatching"


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Canonical fields that uniquely identify an embedding vector space."""

    provider: str
    model: str
    endpoint: str
    dimension: int
    instruct: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


HISTORICAL_DASHSCOPE_IDENTITY = EmbeddingIdentity(
    "dashscope", "text-embedding-v4", DASHSCOPE_DEFAULT_ENDPOINT, 1024, None
)


def canonical_embedding_identity(
    identity: EmbeddingIdentity | EmbeddingModelConfig | Mapping[str, Any],
) -> EmbeddingIdentity:
    """Normalize the complete runtime embedding identity or raise ``ValueError``."""
    if isinstance(identity, EmbeddingIdentity):
        values = identity.as_dict()
    elif isinstance(identity, EmbeddingModelConfig):
        values = {
            "provider": identity.model_provider,
            "model": identity.model_name,
            "endpoint": identity.base_url,
            "dimension": identity.dimension,
            "instruct": identity.instruct,
        }
    else:
        values = dict(identity)
        if set(values) != _IDENTITY_FIELDS:
            raise ValueError("embedding identity must contain exactly five fields")

    provider = str(values.get("provider") or "").strip().lower()
    provider = _PROVIDER_ALIASES.get(provider, provider)
    model = str(values.get("model") or "").strip()
    endpoint_value = values.get("endpoint") or _DEFAULT_ENDPOINTS.get(provider)
    endpoint = str(endpoint_value or "").strip().rstrip("/")
    dimension_value = values.get("dimension")
    instruct = values.get("instruct") if provider == "dashscope" else None
    if provider == "openai" and endpoint.endswith("/v1"):
        endpoint += "/embeddings"
    if (
        not provider
        or not model
        or not endpoint
        or not isinstance(dimension_value, int)
        or isinstance(dimension_value, bool)
        or not isinstance(instruct, (str, type(None)))
    ):
        raise ValueError("embedding identity contains an invalid field")
    if dimension_value <= 0:
        raise ValueError("embedding dimension must be a positive integer")
    return EmbeddingIdentity(
        provider, model, endpoint, dimension_value, instruct or None
    )


def classify_vector_compatibility(
    schema: _ArrowSchema,
    expected_identity: EmbeddingIdentity | EmbeddingModelConfig | Mapping[str, Any],
) -> VectorCompatibility:
    """Purely classify an Arrow schema and its metadata; perform no I/O."""
    try:
        expected = canonical_embedding_identity(expected_identity)
        vector_type = schema.field("vector").type
        schema_matches = all(
            schema.field(name).type == field_type
            for name, field_type in _REQUIRED_SCHEMA.items()
        )
    except (KeyError, TypeError, ValueError):
        return VectorCompatibility.MISMATCHING
    if not (
        schema_matches
        and pa.types.is_fixed_size_list(vector_type)
        and vector_type.value_type == pa.float32()
        and vector_type.list_size == expected.dimension
    ):
        return VectorCompatibility.MISMATCHING

    encoded = (schema.metadata or {}).get(VECTOR_IDENTITY_METADATA_KEY)
    if encoded is None:
        return (
            VectorCompatibility.LEGACY_COMPATIBLE
            if expected == HISTORICAL_DASHSCOPE_IDENTITY
            else VectorCompatibility.MISMATCHING
        )
    try:
        stored = json.loads(encoded.decode("utf-8"))
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
        return VectorCompatibility.MISMATCHING
    return (
        VectorCompatibility.MATCHING
        if stored == expected.as_dict()
        else VectorCompatibility.MISMATCHING
    )


def inspect_lancedb_vector_compatibility(
    connection: Any,
    table_name: str,
    expected_identity: EmbeddingIdentity | EmbeddingModelConfig | Mapping[str, Any],
) -> VectorCompatibility:
    """Inspect one existing LanceDB table using only its Arrow schema metadata."""
    table = connection.open_table(table_name)
    try:
        return classify_vector_compatibility(table.schema, expected_identity)
    finally:
        _safe_close_table(table)


def _validated_rows(batch: Any, seen: set[str]) -> list[tuple[int | None, list[str]]]:
    derived = []
    for row in batch.select(["id", "metadata"]).to_pylist():
        identity, metadata = row["id"], row["metadata"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("legacy IDs must be non-empty strings")
        if identity in seen:
            raise ValueError("legacy IDs must be unique")
        if metadata is not None and not isinstance(metadata, str):
            raise ValueError("legacy metadata must be a string or SQL NULL")
        seen.add(identity)
        scope = derive_scope_columns(metadata)
        if scope[0] is not None and not -(2**63) <= scope[0] < 2**63:
            raise ValueError("legacy user_id must fit signed int64")
        derived.append(scope)
    return derived


def _prepared_schema(schema: Any, identity: EmbeddingIdentity, version: int) -> Any:
    fields = [
        field
        for field in schema
        if field.name not in {USER_ID_COLUMN, SCOPE_DIMS_COLUMN}
    ]
    if "vector" not in schema.names:
        fields.append(pa.field("vector", pa.list_(pa.float32(), identity.dimension)))
    marker = {
        maintenance.MAINTENANCE_METADATA_KEY: maintenance.MAINTENANCE_VERSION,
        maintenance.MAINTENANCE_TABLE_VERSION_KEY: str(version).encode(),
    }
    previous = (
        schema.field(USER_ID_COLUMN).metadata
        if USER_ID_COLUMN in schema.names
        else None
    )
    fields += [
        pa.field(USER_ID_COLUMN, pa.int64(), metadata=dict(previous or {}) | marker),
        pa.field(SCOPE_DIMS_COLUMN, pa.list_(pa.string())),
    ]
    metadata = dict(schema.metadata or {})
    if "vector" not in schema.names:
        metadata[VECTOR_IDENTITY_METADATA_KEY] = json.dumps(
            identity.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
    return pa.schema(fields, metadata=metadata)


def _stage_batches(
    table: Any, schema: Any, batch_size: int, path: str, stats: dict[str, int]
) -> None:
    seen: set[str] = set()
    with pa.OSFile(path, "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
        for batch in table.search().to_batches(batch_size=batch_size):
            _checkpoint("scan_batch", batch.num_rows)
            derived = _validated_rows(batch, seen)
            arrays = []
            for field in schema:
                if field.name == USER_ID_COLUMN:
                    arrays.append(pa.array([value[0] for value in derived], pa.int64()))
                elif field.name == SCOPE_DIMS_COLUMN:
                    arrays.append(
                        pa.array([value[1] for value in derived], pa.list_(pa.string()))
                    )
                elif field.name == "vector" and "vector" not in batch.schema.names:
                    arrays.append(pa.nulls(batch.num_rows, field.type))
                else:
                    arrays.append(
                        batch.column(batch.schema.get_field_index(field.name))
                    )
            stats["rows"] += batch.num_rows
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))


def prepare_lancedb_memory_table(
    connection: Any,
    table_name: str,
    expected_identity: EmbeddingIdentity | EmbeddingModelConfig | Mapping[str, Any],
    *,
    batch_size: int,
    lock_timeout: float,
) -> maintenance.MaintenanceOutcome:
    """Atomically add scope/vector columns using one bounded-memory scan."""
    identity = canonical_embedding_identity(expected_identity)
    lock_path = maintenance.lancedb_lock_path(connection, table_name, "maintenance")
    with FileLock(lock_path, timeout=lock_timeout):
        if table_name not in list_table_names(connection):
            schema = _prepared_schema(
                pa.schema(
                    [
                        ("id", pa.string()),
                        ("text", pa.string()),
                        ("metadata", pa.string()),
                    ]
                ),
                identity,
                1,
            )
            created = connection.create_table(table_name, schema=schema)
            try:
                status = (
                    maintenance.MaintenanceStatus.COMPLETE
                    if int(created.version) == 1
                    else maintenance.MaintenanceStatus.INCOMPLETE
                )
            finally:
                _safe_close_table(created)
            return maintenance.MaintenanceOutcome(status, batches_committed=1)

        table = connection.open_table(table_name)
        staged_path = ""
        try:
            if maintenance._is_complete(table) and "vector" in table.schema.names:
                return maintenance.MaintenanceOutcome(
                    maintenance.MaintenanceStatus.COMPLETE
                )
            required = {"id": pa.string(), "text": pa.string(), "metadata": pa.string()}
            if any(
                name not in table.schema.names or table.schema.field(name).type != kind
                for name, kind in required.items()
            ):
                return maintenance.MaintenanceOutcome(
                    maintenance.MaintenanceStatus.INCOMPATIBLE_SCHEMA
                )
            scope_types = {
                USER_ID_COLUMN: pa.int64(),
                SCOPE_DIMS_COLUMN: pa.list_(pa.string()),
            }
            if any(
                name in table.schema.names and table.schema.field(name).type != kind
                for name, kind in scope_types.items()
            ):
                return maintenance.MaintenanceOutcome(
                    maintenance.MaintenanceStatus.INCOMPATIBLE_SCHEMA
                )
            expected_version = int(table.version) + 1
            schema = _prepared_schema(table.schema, identity, expected_version)
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(lock_path),
                prefix=".memory-stage-",
                suffix=".arrow",
                delete=False,
            ) as staged:
                staged_path = staged.name
            stats = {"rows": 0}
            try:
                _stage_batches(table, schema, batch_size, staged_path, stats)
            except ValueError as exc:
                return maintenance.MaintenanceOutcome(
                    maintenance.MaintenanceStatus.INVALID_LEGACY_DATA,
                    scanned_rows=stats["rows"],
                    detail=str(exc),
                )
            _safe_close_table(table)
            table = None
            with pa.memory_map(staged_path, "r") as source:
                reader = pa.ipc.open_file(source)

                def batches() -> Any:
                    for index in range(reader.num_record_batches):
                        yield reader.get_batch(index)
                        _checkpoint("commit_batch", index + 1)

                rewritten = connection.create_table(
                    table_name,
                    data=batches(),
                    schema=schema,
                    mode="overwrite",
                    on_bad_vectors="null",
                )
                try:
                    actual_version = int(rewritten.version)
                finally:
                    _safe_close_table(rewritten)
            return maintenance.MaintenanceOutcome(
                maintenance.MaintenanceStatus.COMPLETE
                if actual_version == expected_version
                else maintenance.MaintenanceStatus.INCOMPLETE,
                scanned_rows=stats["rows"],
                updated_rows=stats["rows"],
                batches_committed=1,
            )
        finally:
            _safe_close_table(table)
            if staged_path:
                os.unlink(staged_path)


def create_or_recreate_vector_capable_table(
    connection: Any,
    table_name: str,
    expected_identity: EmbeddingIdentity | EmbeddingModelConfig | Mapping[str, Any],
) -> VectorCompatibility:
    """Create a missing table or overwrite a vectorless table with typed data.

    This explicit lifecycle primitive is intentionally not called by request-time
    store acquisition. Its future lifecycle caller must serialize it before writers
    start. Open/create failures propagate unchanged.
    """
    from .lancedb_maintenance import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_LOCK_TIMEOUT,
        MaintenanceStatus,
    )

    identity = canonical_embedding_identity(expected_identity)
    outcome = prepare_lancedb_memory_table(
        connection,
        table_name,
        identity,
        batch_size=DEFAULT_BATCH_SIZE,
        lock_timeout=DEFAULT_LOCK_TIMEOUT,
    )
    if outcome.status is not MaintenanceStatus.COMPLETE:
        raise ValueError(outcome.detail or outcome.status.value)
    compatibility = inspect_lancedb_vector_compatibility(
        connection, table_name, identity
    )
    if compatibility is not VectorCompatibility.MATCHING:
        raise RuntimeError("created memory table failed vector compatibility")
    return compatibility
