"""Tests for RAG ingestion status utilities.

Phase 1A Part 2: Tests for both sync and async methods.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xagent.core.tools.core.RAG_tools.management.status import (
    clear_ingestion_status,
    clear_ingestion_status_async,
    load_ingestion_status,
    load_ingestion_status_async,
    write_ingestion_status,
    write_ingestion_status_async,
)


@pytest.fixture()
def temp_lancedb_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Isolate LANCE DB data directory per test."""

    original = os.environ.get("LANCEDB_DIR")
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path))
    yield str(tmp_path)
    if original is None:
        monkeypatch.delenv("LANCEDB_DIR", raising=False)
    else:
        monkeypatch.setenv("LANCEDB_DIR", original)


def test_write_ingestion_status(temp_lancedb_dir: str) -> None:
    """Test writing ingestion status for a document."""

    collection = "test_collection"
    doc_id = "test_doc"

    write_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        status="running",
        message="Processing document",
        parse_hash="hash-123",
    )

    records = load_ingestion_status(collection=collection, doc_id=doc_id, is_admin=True)
    assert len(records) == 1
    assert records[0]["collection"] == collection
    assert records[0]["doc_id"] == doc_id
    assert records[0]["status"] == "running"
    assert records[0]["message"] == "Processing document"
    assert records[0]["parse_hash"] == "hash-123"


def test_write_ingestion_status_overwrites_existing(temp_lancedb_dir: str) -> None:
    """Test that writing status overwrites existing status."""

    collection = "test_collection"
    doc_id = "test_doc"

    write_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        status="pending",
        message="Initial status",
    )

    write_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        status="success",
        message="Completed",
    )

    records = load_ingestion_status(collection=collection, doc_id=doc_id, is_admin=True)
    assert len(records) == 1
    assert records[0]["status"] == "success"
    assert records[0]["message"] == "Completed"


def test_failed_ingest_status_can_be_restored_then_cleared(
    temp_lancedb_dir: str,
) -> None:
    """Failed ingest rollback status can be restored and then cleared."""

    collection = "failed_status_collection"
    doc_id = "failed_doc"

    write_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        status="failed",
        message="Initial ingest failed",
        user_id=7,
    )
    failed_records = load_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        user_id=7,
        is_admin=False,
    )
    assert len(failed_records) == 1
    assert failed_records[0]["status"] == "failed"

    write_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        status="pending",
        message="Rollback restored status",
        user_id=7,
    )
    restored_records = load_ingestion_status(
        collection=collection,
        doc_id=doc_id,
        user_id=7,
        is_admin=False,
    )
    assert len(restored_records) == 1
    assert restored_records[0]["status"] == "pending"
    assert restored_records[0]["message"] == "Rollback restored status"

    clear_ingestion_status(collection, doc_id, user_id=7, is_admin=False)

    assert (
        load_ingestion_status(
            collection=collection,
            doc_id=doc_id,
            user_id=7,
            is_admin=False,
        )
        == []
    )


def test_load_ingestion_status_by_collection(temp_lancedb_dir: str) -> None:
    """Test loading status records filtered by collection."""

    collection1 = "collection1"
    collection2 = "collection2"

    write_ingestion_status(collection1, "doc1", status="running")
    write_ingestion_status(collection1, "doc2", status="success")
    write_ingestion_status(collection2, "doc1", status="pending")

    records = load_ingestion_status(collection=collection1, is_admin=True)
    assert len(records) == 2
    assert all(r["collection"] == collection1 for r in records)

    records = load_ingestion_status(collection=collection2, is_admin=True)
    assert len(records) == 1
    assert records[0]["collection"] == collection2


def test_load_ingestion_status_by_doc_id(temp_lancedb_dir: str) -> None:
    """Test loading status records filtered by doc_id."""

    collection = "test_collection"

    write_ingestion_status(collection, "doc1", status="running")
    write_ingestion_status(collection, "doc2", status="success")
    write_ingestion_status(collection, "doc3", status="pending")

    records = load_ingestion_status(collection=collection, doc_id="doc1", is_admin=True)
    assert len(records) == 1
    assert records[0]["doc_id"] == "doc1"
    assert records[0]["status"] == "running"


def test_load_ingestion_status_without_filters(temp_lancedb_dir: str) -> None:
    """Test loading all status records when no filters provided."""

    write_ingestion_status("collection1", "doc1", status="running")
    write_ingestion_status("collection2", "doc2", status="success")

    records = load_ingestion_status(is_admin=True)
    assert len(records) >= 2
    doc_ids = {r["doc_id"] for r in records}
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


def test_clear_ingestion_status(temp_lancedb_dir: str) -> None:
    """Test clearing ingestion status for a document."""

    collection = "test_collection"
    doc_id = "test_doc"

    write_ingestion_status(collection, doc_id, status="running", message="Processing")

    records = load_ingestion_status(collection=collection, doc_id=doc_id, is_admin=True)
    assert len(records) == 1

    clear_ingestion_status(collection, doc_id, is_admin=True)

    records = load_ingestion_status(collection=collection, doc_id=doc_id, is_admin=True)
    assert len(records) == 0


def test_clear_ingestion_status_nonexistent(temp_lancedb_dir: str) -> None:
    """Test clearing status for non-existent document (should not raise)."""

    clear_ingestion_status("nonexistent", "nonexistent_doc")

    records = load_ingestion_status(
        collection="nonexistent", doc_id="nonexistent_doc", is_admin=True
    )
    assert len(records) == 0


def test_write_ingestion_status_optional_fields(temp_lancedb_dir: str) -> None:
    """Test writing status with optional fields as None."""

    collection = "test_collection"
    doc_id = "test_doc"

    write_ingestion_status(collection=collection, doc_id=doc_id, status="pending")

    records = load_ingestion_status(collection=collection, doc_id=doc_id, is_admin=True)
    assert len(records) == 1
    assert records[0]["status"] == "pending"
    assert records[0]["message"] == ""
    assert records[0]["parse_hash"] == ""


# ============================================================================
# Async Method Tests (Phase 1A Part 2)
# ============================================================================


@pytest.mark.asyncio
async def test_write_ingestion_status_async(temp_lancedb_dir: str) -> None:
    """Test async version of write_ingestion_status."""

    collection = "test_collection"
    doc_id = "test_doc"

    await write_ingestion_status_async(
        collection=collection,
        doc_id=doc_id,
        status="running",
        message="Processing document",
        parse_hash="hash-123",
    )

    records = await load_ingestion_status_async(
        collection=collection, doc_id=doc_id, is_admin=True
    )
    assert len(records) == 1
    assert records[0]["collection"] == collection
    assert records[0]["doc_id"] == doc_id
    assert records[0]["status"] == "running"
    assert records[0]["message"] == "Processing document"
    assert records[0]["parse_hash"] == "hash-123"


@pytest.mark.asyncio
async def test_write_ingestion_status_overwrites_existing_async(
    temp_lancedb_dir: str,
) -> None:
    """Test async version of write overwrites existing status."""

    collection = "test_collection"
    doc_id = "test_doc"

    await write_ingestion_status_async(
        collection=collection,
        doc_id=doc_id,
        status="pending",
        message="Initial status",
    )

    await write_ingestion_status_async(
        collection=collection,
        doc_id=doc_id,
        status="success",
        message="Completed",
    )

    records = await load_ingestion_status_async(
        collection=collection, doc_id=doc_id, is_admin=True
    )
    assert len(records) == 1
    assert records[0]["status"] == "success"
    assert records[0]["message"] == "Completed"


@pytest.mark.asyncio
async def test_load_ingestion_status_by_collection_async(temp_lancedb_dir: str) -> None:
    """Test async version of load status by collection."""

    collection1 = "collection1"
    collection2 = "collection2"

    await write_ingestion_status_async(collection1, "doc1", status="running")
    await write_ingestion_status_async(collection1, "doc2", status="success")
    await write_ingestion_status_async(collection2, "doc1", status="pending")

    records = await load_ingestion_status_async(collection=collection1, is_admin=True)
    assert len(records) == 2
    assert all(r["collection"] == collection1 for r in records)

    records = await load_ingestion_status_async(collection=collection2, is_admin=True)
    assert len(records) == 1
    assert records[0]["collection"] == collection2


@pytest.mark.asyncio
async def test_clear_ingestion_status_async(temp_lancedb_dir: str) -> None:
    """Test async version of clear ingestion status."""

    collection = "test_collection"
    doc_id = "test_doc"

    await write_ingestion_status_async(
        collection, doc_id, status="running", message="Processing"
    )

    records = await load_ingestion_status_async(
        collection=collection, doc_id=doc_id, is_admin=True
    )
    assert len(records) == 1

    await clear_ingestion_status_async(collection, doc_id, is_admin=True)

    records = await load_ingestion_status_async(
        collection=collection, doc_id=doc_id, is_admin=True
    )
    assert len(records) == 0
