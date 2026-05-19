"""Tests for list_candidates using VectorIndexStore contract."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    VersionManagementError,
)
from xagent.core.tools.core.RAG_tools.core.schemas import StepType
from xagent.core.tools.core.RAG_tools.version_management.list_candidates import (
    list_candidates,
)


def test_invalid_step_type() -> None:
    """Invalid step strings should raise VersionManagementError."""
    with pytest.raises(
        VersionManagementError,
        match="Invalid step_type string: 'invalid_step'",
    ):
        list_candidates("test_collection", "test_doc", "invalid_step")  # type: ignore[arg-type]


def test_parse_candidates_from_store_rows() -> None:
    """Parse candidates should be built from store contract rows."""
    now = datetime.now()
    rows = [
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "hash1",
            "parse_method": "unstructured",
            "parser": "local:UnstructuredParser@v1",
            "created_at": now + timedelta(milliseconds=1),
        },
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "hash2",
            "parse_method": "pypdf",
            "parser": "local:PyPDFParser@v1",
            "created_at": now,
        },
    ]
    store = Mock()
    store.list_version_candidates.return_value = rows

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates.get_vector_index_store",
        return_value=store,
    ):
        result = list_candidates("test_collection", "test_doc", StepType.PARSE)

    assert len(result["candidates"]) == 2
    assert result["total_count"] == 2
    assert result["returned_count"] == 2
    assert result["candidates"][0]["technical_id"] == "hash1"
    store.list_version_candidates.assert_called_once_with(
        collection="test_collection",
        doc_id="test_doc",
        step_type="parse",
        model_tag=None,
    )


def test_chunk_candidates_grouped_by_parse_hash() -> None:
    """Chunk candidates should be grouped by parse_hash."""
    base = datetime.now()
    rows = [
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "parse_hash1",
            "chunk_id": "chunk1",
            "text": "This is chunk 1",
            "created_at": base + timedelta(milliseconds=2),
        },
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "parse_hash1",
            "chunk_id": "chunk2",
            "text": "This is chunk 2",
            "created_at": base + timedelta(milliseconds=1),
        },
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "parse_hash2",
            "chunk_id": "chunk3",
            "text": "This is chunk 3",
            "created_at": base,
        },
    ]
    store = Mock()
    store.list_version_candidates.return_value = rows

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates.get_vector_index_store",
        return_value=store,
    ):
        result = list_candidates("test_collection", "test_doc", StepType.CHUNK)

    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["technical_id"] == "parse_hash1"


def test_embed_candidates_require_model_tag() -> None:
    """Embed listing should require model_tag."""
    store = Mock()
    store.list_version_candidates.return_value = []

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates.get_vector_index_store",
        return_value=store,
    ):
        with pytest.raises(VersionManagementError, match="model_tag is required"):
            list_candidates("test_collection", "test_doc", StepType.EMBED)


def test_embed_candidates_from_store_rows() -> None:
    """Embed candidates should be built from adapter-returned rows."""
    rows = [
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "model": "BAAI/bge-large-zh-v1.5",
            "parse_hash": "parse_hash1",
            "vector": [0.1, 0.2, 0.3],
            "created_at": datetime.now(),
        },
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "model": "BAAI/bge-large-zh-v1.5",
            "parse_hash": "parse_hash2",
            "vector": [0.4, 0.5, 0.6],
            "created_at": datetime.now(),
        },
    ]
    store = Mock()
    store.list_version_candidates.return_value = rows

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates.get_vector_index_store",
        return_value=store,
    ):
        result = list_candidates(
            "test_collection", "test_doc", StepType.EMBED, model_tag="bge_large"
        )

    assert len(result["candidates"]) == 2
    assert result["step_type"] == "embed"
    store.list_version_candidates.assert_called_once_with(
        collection="test_collection",
        doc_id="test_doc",
        step_type="embed",
        model_tag="bge_large",
    )


def test_state_filter() -> None:
    """State filter should keep only matching candidates."""
    now = datetime.now()
    rows = [
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "hash1",
            "parse_method": "unstructured",
            "parser": "local:UnstructuredParser@v1",
            "created_at": now,
        }
    ]
    store = Mock()
    store.list_version_candidates.return_value = rows

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates.get_vector_index_store",
        return_value=store,
    ):
        result = list_candidates(
            "test_collection",
            "test_doc",
            StepType.PARSE,
            state="candidate",
        )

    assert len(result["candidates"]) == 1
    assert result["total_count"] == 1
    assert result["returned_count"] == 1
    assert result["filters"]["state"] == "candidate"
    assert result["candidates"][0]["state"] == "candidate"


def test_limit_filter() -> None:
    """Limit should truncate results after sorting/filtering."""
    base = datetime.now()
    rows = [
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": f"hash{i}",
            "parse_method": "unstructured",
            "parser": "local:UnstructuredParser@v1",
            "created_at": base + timedelta(seconds=i),
        }
        for i in range(5)
    ]
    store = Mock()
    store.list_version_candidates.return_value = rows

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates.get_vector_index_store",
        return_value=store,
    ):
        result = list_candidates(
            "test_collection",
            "test_doc",
            StepType.PARSE,
            limit=3,
        )

    assert len(result["candidates"]) == 3
    assert result["total_count"] == 5
    assert result["returned_count"] == 3
    assert result["filters"]["limit"] == 3


def test_sort_before_limit() -> None:
    """Sorting must happen before limiting to keep newest N rows."""
    base = datetime(2024, 1, 1)
    rows = [
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "hash_oldest",
            "parse_method": "unstructured",
            "parser": "local:UnstructuredParser@v1",
            "created_at": base,
        },
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "hash_middle",
            "parse_method": "unstructured",
            "parser": "local:UnstructuredParser@v1",
            "created_at": base + timedelta(days=5),
        },
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "hash_newer",
            "parse_method": "unstructured",
            "parser": "local:UnstructuredParser@v1",
            "created_at": base + timedelta(days=7),
        },
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "hash_newest",
            "parse_method": "unstructured",
            "parser": "local:UnstructuredParser@v1",
            "created_at": base + timedelta(days=10),
        },
        {
            "collection": "test_collection",
            "doc_id": "test_doc",
            "parse_hash": "hash_second_newest",
            "parse_method": "unstructured",
            "parser": "local:UnstructuredParser@v1",
            "created_at": base + timedelta(days=8),
        },
    ]
    store = Mock()
    store.list_version_candidates.return_value = rows

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates.get_vector_index_store",
        return_value=store,
    ):
        result = list_candidates(
            "test_collection",
            "test_doc",
            StepType.PARSE,
            limit=3,
            order_by="created_at desc",
        )

    technical_ids = [c["technical_id"] for c in result["candidates"]]
    assert technical_ids == ["hash_newest", "hash_second_newest", "hash_newer"]
    assert result["total_count"] == 5
    assert result["returned_count"] == 3


def test_embed_model_tag_forwarded_to_store() -> None:
    """Embed model_tag should be forwarded to store query call."""
    store = Mock()
    store.list_version_candidates.return_value = []

    with patch(
        "xagent.core.tools.core.RAG_tools.version_management.list_candidates.get_vector_index_store",
        return_value=store,
    ):
        result = list_candidates(
            "test_collection",
            "test_doc",
            StepType.EMBED,
            model_tag="minilm",
        )

    assert result["step_type"] == "embed"
    assert result["model_tag"] == "minilm"
    store.list_version_candidates.assert_called_once_with(
        collection="test_collection",
        doc_id="test_doc",
        step_type="embed",
        model_tag="minilm",
    )
