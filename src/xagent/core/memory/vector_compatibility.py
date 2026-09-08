"""Read-only vector-space compatibility capabilities for persistent memory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

import pyarrow as pa  # type: ignore

from ..model.model import EmbeddingModelConfig
from ..tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table

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
