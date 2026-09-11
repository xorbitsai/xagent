"""Read-only vector-space compatibility capabilities for persistent memory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, cast

import pyarrow as pa  # type: ignore

from ..model.model import EmbeddingModelConfig
from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table
from ..tools.core.RAG_tools.utils.lancedb_query_utils import list_table_names
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


class _ArrowTable(Protocol):
    """Typed boundary for the Arrow table returned to LanceDB."""

    @property
    def schema(self) -> _ArrowSchema: ...


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


def open_lancedb_table_if_exists(connection: Any, table_name: str) -> Any | None:
    """Return a fresh table handle, or ``None`` when listing proves it absent.

    Listing and opening failures deliberately propagate. In particular, a
    ``ValueError`` or ``OSError`` from ``open_table`` is not reclassified by
    matching backend-specific error text.
    """
    if table_name not in list_table_names(connection):
        return None
    return connection.open_table(table_name)


def _vector_capable_data(
    identity: EmbeddingIdentity, existing: Any | None = None
) -> _ArrowTable:
    """Build typed memory data carrying one authoritative vector identity."""
    if existing is None:
        data = pa.table(
            {
                "id": pa.array(["__xagent_schema_seed__"], pa.string()),
                "text": pa.array([""], pa.string()),
                "metadata": pa.array(["{}"], pa.string()),
                "vector": pa.array(
                    [[0.0] * identity.dimension],
                    pa.list_(pa.float32(), identity.dimension),
                ),
                USER_ID_COLUMN: pa.array([None], pa.int64()),
                SCOPE_DIMS_COLUMN: pa.array([[]], pa.list_(pa.string())),
            }
        )
    else:
        names = set(existing.schema.names)
        missing = {"id", "text", "metadata"} - names
        if missing:
            raise ValueError(
                "vectorless memory table is missing required columns: "
                + ", ".join(sorted(missing))
            )
        columns = {name: existing[name] for name in existing.schema.names}
        for name in ("id", "text", "metadata"):
            columns[name] = columns[name].cast(pa.string())
        columns["vector"] = pa.nulls(
            existing.num_rows, pa.list_(pa.float32(), identity.dimension)
        )
        derived = [
            derive_scope_columns(value) for value in columns["metadata"].to_pylist()
        ]
        columns[USER_ID_COLUMN] = pa.array(
            [user_id for user_id, _dims in derived], pa.int64()
        )
        columns[SCOPE_DIMS_COLUMN] = pa.array(
            [dims for _user_id, dims in derived], pa.list_(pa.string())
        )
        data = pa.table(columns)
    metadata = dict(
        (existing.schema.metadata if existing is not None else data.schema.metadata)
        or {}
    )
    metadata[VECTOR_IDENTITY_METADATA_KEY] = json.dumps(
        identity.as_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return cast(_ArrowTable, data.replace_schema_metadata(metadata))


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
    identity = canonical_embedding_identity(expected_identity)
    existing = None
    table = None
    try:
        table = open_lancedb_table_if_exists(connection, table_name)
        if table is not None:
            if "vector" in table.schema.names:
                return classify_vector_compatibility(table.schema, identity)
            existing = table.to_arrow()
        data = _vector_capable_data(identity, existing)
    finally:
        _safe_close_table(table)

    created = connection.create_table(
        table_name,
        data=data,
        mode="overwrite" if existing is not None else "create",
    )
    try:
        if existing is None:
            created.delete("id = '__xagent_schema_seed__'")
        outcome = classify_vector_compatibility(created.schema, identity)
        if outcome is not VectorCompatibility.MATCHING:
            raise RuntimeError("created memory table failed vector compatibility")
        return outcome
    finally:
        _safe_close_table(created)
