"""Real-LanceDB tests for the read-only vector compatibility capability."""

import json

import lancedb  # type: ignore
import pyarrow as pa  # type: ignore
import pytest

from xagent.core.memory.vector_compatibility import (
    DASHSCOPE_DEFAULT_ENDPOINT,
    HISTORICAL_DASHSCOPE_IDENTITY,
    VECTOR_IDENTITY_METADATA_KEY,
    VectorCompatibility,
    canonical_embedding_identity,
    inspect_lancedb_vector_compatibility,
)
from xagent.core.model.embedding import DashScopeEmbedding
from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import _safe_close_table


def _identity(**changes):
    values = HISTORICAL_DASHSCOPE_IDENTITY.as_dict()
    values.update(changes)
    return values


def _table_data(dimension=1024, metadata=None):
    table = pa.table(
        {
            "id": ["note-1"],
            "text": ["remember this"],
            "metadata": ["{}"],
            "vector": pa.array([[0.0] * dimension], pa.list_(pa.float32(), dimension)),
        }
    )
    return table.replace_schema_metadata(metadata) if metadata is not None else table


def _create_table(tmp_path, *, dimension=1024, identity=None, raw_metadata=None):
    connection = lancedb.connect(tmp_path)
    metadata = raw_metadata
    if identity is not None:
        metadata = {VECTOR_IDENTITY_METADATA_KEY: json.dumps(identity).encode("utf-8")}
    table = connection.create_table(
        "memories", data=_table_data(dimension=dimension, metadata=metadata)
    )
    _safe_close_table(table)
    return connection


def _inspect(connection, expected=None):
    return inspect_lancedb_vector_compatibility(
        connection, "memories", expected or HISTORICAL_DASHSCOPE_IDENTITY
    )


def test_known_historical_table_is_legacy_compatible(tmp_path):
    assert _inspect(_create_table(tmp_path)) is VectorCompatibility.LEGACY_COMPATIBLE


def test_canonical_identity_normalizes_runtime_fields():
    config = EmbeddingModelConfig(
        id="embedding",
        model_provider=" OpenAI-Compatible ",
        model_name=" text-embedding-3-small ",
        base_url="https://api.openai.com/v1/",
        dimension=1536,
        instruct="ignored by this provider",
    )
    assert canonical_embedding_identity(config).as_dict() == {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "endpoint": "https://api.openai.com/v1/embeddings",
        "dimension": 1536,
        "instruct": None,
    }


def test_exact_persisted_identity_is_matching(tmp_path):
    connection = _create_table(tmp_path, identity=_identity())
    assert _inspect(connection) is VectorCompatibility.MATCHING


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "openai"),
        ("model", "text-embedding-v3"),
        ("endpoint", "https://proxy.example/v1/embeddings"),
        ("dimension", 512),
        ("instruct", "Represent this sentence for retrieval"),
    ],
)
def test_each_identity_field_mismatch_is_rejected(tmp_path, field, value):
    expected = _identity(**{field: value})
    connection = _create_table(tmp_path, dimension=expected["dimension"])
    assert _inspect(connection, expected) is VectorCompatibility.MISMATCHING


@pytest.mark.parametrize("missing", sorted(_identity()))
def test_persisted_identity_requires_every_canonical_field(tmp_path, missing):
    stored = _identity()
    stored.pop(missing)
    connection = _create_table(tmp_path, identity=stored)
    assert _inspect(connection) is VectorCompatibility.MISMATCHING


def test_malformed_identity_metadata_is_mismatching(tmp_path):
    connection = _create_table(
        tmp_path, raw_metadata={VECTOR_IDENTITY_METADATA_KEY: b"not-json"}
    )
    assert _inspect(connection) is VectorCompatibility.MISMATCHING


def test_noncanonical_schema_is_mismatching(tmp_path):
    connection = lancedb.connect(tmp_path)
    table = connection.create_table(
        "memories",
        pa.table(
            {
                "id": pa.array([1], pa.int64()),
                "text": ["remember this"],
                "metadata": ["{}"],
                "vector": pa.array(
                    [
                        [0.0] * 1024,
                    ],
                    pa.list_(pa.float64(), 1024),
                ),
            }
        ),
    )
    _safe_close_table(table)
    assert _inspect(connection) is VectorCompatibility.MISMATCHING


def test_inspection_never_embeds_or_mutates_table(tmp_path, monkeypatch):
    connection = _create_table(tmp_path, identity=_identity())
    table = connection.open_table("memories")
    try:
        version_before = table.version
        schema_before = table.schema
        rows_before = table.to_arrow().to_pylist()
    finally:
        _safe_close_table(table)

    def forbidden_encode(*_args, **_kwargs):
        raise AssertionError("inspection must not call an embedding API")

    monkeypatch.setattr(DashScopeEmbedding, "encode", forbidden_encode)
    config = EmbeddingModelConfig(
        id="historical",
        model_provider="dashscope",
        model_name="text-embedding-v4",
        base_url=DASHSCOPE_DEFAULT_ENDPOINT,
        dimension=1024,
    )
    assert _inspect(connection, config) is VectorCompatibility.MATCHING

    table = connection.open_table("memories")
    try:
        assert table.version == version_before
        assert table.schema == schema_before
        assert table.to_arrow().to_pylist() == rows_before
    finally:
        _safe_close_table(table)
