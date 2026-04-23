"""Integration tests for automatic LanceDB migration on application startup.

This module tests the automatic migration logic that runs during application
startup to add user_id fields to existing LanceDB tables.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from types import ModuleType

import pyarrow as pa
import pytest

from xagent.core.tools.core.RAG_tools.core.config import MIN_INT64
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    check_table_needs_migration,
)
from xagent.migrations.lancedb.backfill_user_id import (
    ORPHANED_PERMANENT,
    ORPHANED_TEMPORARY,
    backfill_all,
    backfill_chunks_table,
    backfill_orphaned_chunks,
    backfill_orphaned_embeddings,
)
from xagent.providers.vector_store.lancedb import get_connection_from_env


def _patch_channel_modules_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid Telegram/Feishu startup errors when optional deps are missing.

    ``startup_event`` optionally starts the Telegram and Feishu channel managers.
    These managers pull optional dependencies (aiogram, feishu) that CI may omit.

    We inject lightweight stub modules into ``sys.modules`` instead of
    ``monkeypatch.setattr("...telegram.bot...", ...)``, because importing the real
    ``telegram.bot`` or ``feishu.bot`` pulls optional dependencies.
    """

    class _FakeTelegramChannel:
        enabled = False

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class _FakeFeishuChannel:
        enabled = False  # Disabled to prevent task creation in tests

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    # Create fake telegram.bot module
    fake_telegram_bot = ModuleType("xagent.web.channels.telegram.bot")
    fake_telegram_bot.get_telegram_channel = lambda: _FakeTelegramChannel()
    monkeypatch.setitem(
        sys.modules, "xagent.web.channels.telegram.bot", fake_telegram_bot
    )

    # Create fake feishu.bot module
    fake_feishu_bot = ModuleType("xagent.web.channels.feishu.bot")
    fake_feishu_bot.get_feishu_channel = lambda: _FakeFeishuChannel()
    monkeypatch.setitem(sys.modules, "xagent.web.channels.feishu.bot", fake_feishu_bot)


@pytest.fixture
def temp_lancedb_dir():
    """Create a temporary directory for LanceDB."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = temp_dir
        yield temp_dir
        if original_env is not None:
            os.environ["LANCEDB_DIR"] = original_env
        else:
            os.environ.pop("LANCEDB_DIR", None)


def test_auto_migration_detects_old_schema(temp_lancedb_dir):
    """Test that auto-migration detects tables with old schema (missing user_id)."""
    conn = get_connection_from_env()

    # Create tables with old schema (without user_id)
    old_chunks_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.large_string()),
            pa.field("metadata", pa.string()),
        ]
    )
    conn.create_table("chunks", schema=old_chunks_schema)

    old_docs_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("source_path", pa.string()),
        ]
    )
    conn.create_table("documents", schema=old_docs_schema)

    # Verify that migration is needed
    assert check_table_needs_migration(conn, "chunks") is True
    assert check_table_needs_migration(conn, "documents") is True


def test_auto_migration_skips_new_schema(temp_lancedb_dir):
    """Test that auto-migration skips tables with new schema (has user_id)."""
    conn = get_connection_from_env()

    # Create tables with new schema (with user_id)
    new_chunks_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("text", pa.large_string()),
            pa.field("metadata", pa.string()),
            pa.field("user_id", pa.int32(), nullable=True),
        ]
    )
    conn.create_table("chunks", schema=new_chunks_schema)

    # Verify that no migration is needed
    assert check_table_needs_migration(conn, "chunks") is False


def test_auto_migration_handles_missing_tables(temp_lancedb_dir):
    """Test that auto-migration handles non-existent tables gracefully."""
    conn = get_connection_from_env()

    # Check non-existent tables
    assert check_table_needs_migration(conn, "nonexistent_table") is False
    assert check_table_needs_migration(conn, "chunks") is False
    assert check_table_needs_migration(conn, "documents") is False


def test_migration_detection_for_multiple_tables(temp_lancedb_dir):
    """Test that migration detection works correctly for multiple tables."""
    conn = get_connection_from_env()

    # Create multiple tables with old schema
    old_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("text", pa.string()),
        ]
    )

    conn.create_table("chunks", schema=old_schema)
    conn.create_table("documents", schema=old_schema)
    conn.create_table("parses", schema=old_schema)

    # All should need migration
    assert check_table_needs_migration(conn, "chunks") is True
    assert check_table_needs_migration(conn, "documents") is True
    assert check_table_needs_migration(conn, "parses") is True

    # Non-existent table should not need migration
    assert check_table_needs_migration(conn, "nonexistent") is False


def test_auto_migration_handles_embeddings_tables(temp_lancedb_dir):
    """Test that auto-migration detects embeddings tables correctly."""
    conn = get_connection_from_env()

    # Create embeddings table with old schema
    old_embeddings_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
            pa.field("text", pa.large_string()),
        ]
    )
    conn.create_table("embeddings_test_model", schema=old_embeddings_schema)

    # Verify that migration is needed
    assert check_table_needs_migration(conn, "embeddings_test_model") is True

    # Create embeddings table with new schema
    new_embeddings_schema = pa.schema(
        [
            pa.field("collection", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
            pa.field("text", pa.large_string()),
            pa.field("user_id", pa.int32(), nullable=True),
        ]
    )
    conn.create_table("embeddings_test_model_new", schema=new_embeddings_schema)

    # Verify that no migration is needed
    assert check_table_needs_migration(conn, "embeddings_test_model_new") is False


def test_two_phase_migration_recovers_orphaned_records(temp_lancedb_dir):
    """Test that two-phase migration recovers orphaned records when documents are created after chunks."""
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_chunks_table,
        ensure_documents_table,
    )

    conn = get_connection_from_env()

    # Create chunks table using ensure_chunks_table (creates proper schema)
    ensure_chunks_table(conn)
    ensure_documents_table(conn)

    # Add some chunks without user_id (simulating old data)
    chunks_table = conn.open_table("chunks")
    chunks_table.add(
        [
            {
                "collection": "test",
                "doc_id": "doc_1",
                "parse_hash": "hash1",
                "chunk_id": "chunk_1",
                "index": 0,
                "text": "Test chunk 1",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "ch1",
                "config_hash": "cfg1",
                "created_at": 1700000000000,
                "metadata": "{}",
                "user_id": None,
            },
            {
                "collection": "test",
                "doc_id": "doc_2",
                "parse_hash": "hash2",
                "chunk_id": "chunk_2",
                "index": 0,
                "text": "Test chunk 2",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "ch2",
                "config_hash": "cfg2",
                "created_at": 1700000000000,
                "metadata": "{}",
                "user_id": None,
            },
        ]
    )

    # Phase 1: Run migration - chunks should be marked as orphaned (user_id = ORPHANED_TEMPORARY)
    result_phase1 = backfill_chunks_table(dry_run=False, conn=conn)
    assert result_phase1["skipped"] == 2  # Both chunks orphaned in phase 1
    assert result_phase1["backfilled"] == 0

    # Verify that chunks are marked with ORPHANED_TEMPORARY after Phase 1
    phase1_chunks = conn.open_table("chunks").search().to_arrow()
    phase1_chunk_data = phase1_chunks.to_pylist()
    for chunk in phase1_chunk_data:
        assert (
            chunk["user_id"] == ORPHANED_TEMPORARY
        )  # Should be marked as temporary orphan

    # Now add documents (simulating concurrent document creation)
    docs_table = conn.open_table("documents")
    docs_table.add(
        [
            {
                "collection": "test",
                "doc_id": "doc_1",
                "source_path": "/path/doc1.txt",
                "file_type": "txt",
                "content_hash": "h1",
                "uploaded_at": 1700000000000,
                "title": "Doc 1",
                "language": "en",
                "user_id": 100,
            },
            {
                "collection": "test",
                "doc_id": "doc_2",
                "source_path": "/path/doc2.txt",
                "file_type": "txt",
                "content_hash": "h2",
                "uploaded_at": 1700000000000,
                "title": "Doc 2",
                "language": "en",
                "user_id": 200,
            },
        ]
    )

    # Phase 2: Retry orphaned chunks
    result_phase2 = backfill_orphaned_chunks(dry_run=False, conn=conn)

    # Verify that orphaned chunks were recovered
    assert result_phase2["backfilled"] == 2  # Both chunks recovered
    assert result_phase2["skipped"] == 0  # No longer orphaned

    # Verify final state
    final_chunks = conn.open_table("chunks").search().to_arrow()
    assert len(final_chunks) == 2
    # Check that user_id values are correct
    chunk_user_ids = final_chunks.to_pylist()
    doc_1_chunk = next(c for c in chunk_user_ids if c["doc_id"] == "doc_1")
    doc_2_chunk = next(c for c in chunk_user_ids if c["doc_id"] == "doc_2")
    assert doc_1_chunk["user_id"] == 100
    assert doc_2_chunk["user_id"] == 200


def test_two_phase_migration_handles_permanently_orphaned_records(temp_lancedb_dir):
    """Test that permanently orphaned records (no matching document ever exists) are marked correctly."""
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_chunks_table,
        ensure_documents_table,
    )

    conn = get_connection_from_env()

    # Create chunks and documents tables
    ensure_chunks_table(conn)
    ensure_documents_table(conn)

    # Add a chunk with a doc_id that will never have a matching document
    chunks_table = conn.open_table("chunks")
    chunks_table.add(
        [
            {
                "collection": "test",
                "doc_id": "deleted_doc",
                "parse_hash": "hash_deleted",
                "chunk_id": "orphaned_chunk",
                "index": 0,
                "text": "This chunk has no matching document",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "ch_deleted",
                "config_hash": "cfg_deleted",
                "created_at": 1700000000000,
                "metadata": "{}",
                "user_id": None,
            }
        ]
    )

    # Run full migration (both phases)
    result = backfill_all(dry_run=False, conn=conn)

    # Verify results
    assert result["chunks"]["backfilled"] == 0  # No chunks backfilled
    assert result["chunks"]["skipped"] == 1  # One chunk permanently orphaned

    # Verify the chunk is marked with user_id = ORPHANED_PERMANENT
    final_chunks = conn.open_table("chunks").search().to_arrow()
    chunk_data = final_chunks.to_pylist()[0]
    assert chunk_data["user_id"] == ORPHANED_PERMANENT  # Marked as permanently orphaned


def test_two_phase_migration_with_mixed_scenarios(temp_lancedb_dir):
    """Test two-phase migration with a mix of normal, temporarily orphaned, and permanently orphaned records."""
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_chunks_table,
        ensure_documents_table,
    )

    conn = get_connection_from_env()

    # Create chunks and documents tables
    ensure_chunks_table(conn)
    ensure_documents_table(conn)

    # Add documents for doc_1 and doc_3 (doc_2 will be added later, doc_4 never exists)
    docs_table = conn.open_table("documents")
    docs_table.add(
        [
            {
                "collection": "test",
                "doc_id": "doc_1",
                "source_path": "/path/doc1.txt",
                "file_type": "txt",
                "content_hash": "h1",
                "uploaded_at": 1700000000000,
                "title": "Doc 1",
                "language": "en",
                "user_id": 100,
            },
            {
                "collection": "test",
                "doc_id": "doc_3",
                "source_path": "/path/doc3.txt",
                "file_type": "txt",
                "content_hash": "h3",
                "uploaded_at": 1700000000000,
                "title": "Doc 3",
                "language": "en",
                "user_id": 300,
            },
        ]
    )

    # Add chunks with proper schema
    chunks_table = conn.open_table("chunks")
    chunks_table.add(
        [
            {
                "collection": "test",
                "doc_id": "doc_1",
                "parse_hash": "hash1",
                "chunk_id": "chunk_1",
                "index": 0,
                "text": "Chunk with doc",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "ch1",
                "config_hash": "cfg1",
                "created_at": 1700000000000,
                "metadata": "{}",
                "user_id": None,
            },
            {
                "collection": "test",
                "doc_id": "doc_2",
                "parse_hash": "hash2",
                "chunk_id": "chunk_2",
                "index": 0,
                "text": "Chunk without doc yet",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "ch2",
                "config_hash": "cfg2",
                "created_at": 1700000000000,
                "metadata": "{}",
                "user_id": None,
            },
            {
                "collection": "test",
                "doc_id": "doc_3",
                "parse_hash": "hash3",
                "chunk_id": "chunk_3",
                "index": 0,
                "text": "Another chunk with doc",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "ch3",
                "config_hash": "cfg3",
                "created_at": 1700000000000,
                "metadata": "{}",
                "user_id": None,
            },
            {
                "collection": "test",
                "doc_id": "doc_4",
                "parse_hash": "hash4",
                "chunk_id": "chunk_4",
                "index": 0,
                "text": "Chunk with deleted doc",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "ch4",
                "config_hash": "cfg4",
                "created_at": 1700000000000,
                "metadata": "{}",
                "user_id": None,
            },
        ]
    )

    # Phase 1: Initial migration
    result_phase1 = backfill_chunks_table(dry_run=False, conn=conn)

    # chunk_1 and chunk_3 should be backfilled
    # chunk_2 and chunk_4 should be marked orphaned
    assert result_phase1["backfilled"] == 2
    assert result_phase1["skipped"] == 2

    # Verify that chunk_2 and chunk_4 are marked with ORPHANED_TEMPORARY after Phase 1
    phase1_chunks = conn.open_table("chunks").search().to_arrow()
    phase1_chunk_data = phase1_chunks.to_pylist()
    chunk_2_phase1 = next(c for c in phase1_chunk_data if c["chunk_id"] == "chunk_2")
    chunk_4_phase1 = next(c for c in phase1_chunk_data if c["chunk_id"] == "chunk_4")
    assert chunk_2_phase1["user_id"] == ORPHANED_TEMPORARY
    assert chunk_4_phase1["user_id"] == ORPHANED_TEMPORARY

    # Now add document for doc_2 (simulating late-arriving document)
    docs_table.add(
        [
            {
                "collection": "test",
                "doc_id": "doc_2",
                "source_path": "/path/doc2.txt",
                "file_type": "txt",
                "content_hash": "h2",
                "uploaded_at": 1700000000000,
                "title": "Doc 2",
                "language": "en",
                "user_id": 200,
            }
        ]
    )

    # Phase 2: Retry orphaned chunks
    result_phase2 = backfill_orphaned_chunks(dry_run=False, conn=conn)

    # chunk_2 should be recovered
    # chunk_4 remains orphaned
    assert result_phase2["backfilled"] == 1
    assert result_phase2["skipped"] == 1

    # Verify final state
    final_chunks = conn.open_table("chunks").search().to_arrow()
    chunk_data = final_chunks.to_pylist()

    # Find each chunk and verify user_id
    chunk_1 = next(c for c in chunk_data if c["chunk_id"] == "chunk_1")
    chunk_2 = next(c for c in chunk_data if c["chunk_id"] == "chunk_2")
    chunk_3 = next(c for c in chunk_data if c["chunk_id"] == "chunk_3")
    chunk_4 = next(c for c in chunk_data if c["chunk_id"] == "chunk_4")

    assert chunk_1["user_id"] == 100  # Backfilled in phase 1
    assert chunk_2["user_id"] == 200  # Recovered in phase 2
    assert chunk_3["user_id"] == 300  # Backfilled in phase 1
    assert chunk_4["user_id"] == ORPHANED_PERMANENT  # Permanently orphaned


def test_two_phase_migration_with_no_orphaned_records(temp_lancedb_dir):
    """Test that phase 2 is skipped when there are no orphaned records."""
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_chunks_table,
        ensure_documents_table,
    )

    conn = get_connection_from_env()

    # Create chunks and documents tables
    ensure_chunks_table(conn)
    ensure_documents_table(conn)

    # Add document and chunk
    docs_table = conn.open_table("documents")
    docs_table.add(
        [
            {
                "collection": "test",
                "doc_id": "doc_1",
                "source_path": "/path/doc1.txt",
                "file_type": "txt",
                "content_hash": "h1",
                "uploaded_at": 1700000000000,
                "title": "Doc 1",
                "language": "en",
                "user_id": 100,
            }
        ]
    )

    chunks_table = conn.open_table("chunks")
    chunks_table.add(
        [
            {
                "collection": "test",
                "doc_id": "doc_1",
                "parse_hash": "hash1",
                "chunk_id": "chunk_1",
                "index": 0,
                "text": "Test chunk",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "ch1",
                "config_hash": "cfg1",
                "created_at": 1700000000000,
                "metadata": "{}",
                "user_id": None,
            }
        ]
    )

    # Run full migration
    result = backfill_all(dry_run=False, conn=conn)

    # All records should be backfilled in phase 1, phase 2 should be skipped
    assert result["chunks"]["backfilled"] == 1
    assert result["chunks"]["skipped"] == 0  # No orphaned records
    assert result["chunks"].get("failed", 0) == 0

    # Verify final state
    final_chunks = conn.open_table("chunks").search().to_arrow()
    chunk_data = final_chunks.to_pylist()[0]
    assert chunk_data["user_id"] == 100


def test_backfill_all_remaps_legacy_minus_one_orphans(temp_lancedb_dir):
    """Legacy -1 orphan markers are remapped and then retried in phase 2."""
    from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
        ensure_chunks_table,
        ensure_documents_table,
    )

    conn = get_connection_from_env()
    ensure_chunks_table(conn)
    ensure_documents_table(conn)

    docs_table = conn.open_table("documents")
    docs_table.add(
        [
            {
                "collection": "test",
                "doc_id": "doc_legacy",
                "source_path": "/path/doc_legacy.txt",
                "file_type": "txt",
                "content_hash": "hash_legacy",
                "uploaded_at": 1700000000000,
                "title": "Legacy Doc",
                "language": "en",
                "user_id": 777,
            }
        ]
    )

    chunks_table = conn.open_table("chunks")
    chunks_table.add(
        [
            {
                "collection": "test",
                "doc_id": "doc_legacy",
                "parse_hash": "hash_legacy",
                "chunk_id": "chunk_legacy",
                "index": 0,
                "text": "legacy orphaned chunk",
                "page_number": 1,
                "section": "",
                "anchor": "",
                "json_path": "",
                "chunk_hash": "ch_legacy",
                "config_hash": "cfg_legacy",
                "created_at": 1700000000000,
                "metadata": "{}",
                "user_id": -1,  # historical transitional value
            }
        ]
    )

    result = backfill_all(dry_run=False, conn=conn)

    # Legacy -1 rows should be remapped to MIN_INT64 then recovered in phase 2.
    assert result["chunks"]["backfilled"] == 1
    assert result["chunks"]["skipped"] == 0

    final_chunks = conn.open_table("chunks").search().to_arrow().to_pylist()
    assert len(final_chunks) == 1
    assert final_chunks[0]["user_id"] == 777
    assert final_chunks[0]["user_id"] != -1
    assert final_chunks[0]["user_id"] != MIN_INT64


def test_phase2_ensures_tables_exist(temp_lancedb_dir):
    """Phase 2 helpers should be callable without prior Phase 1 table creation."""
    conn = get_connection_from_env()

    # Should not raise even if tables don't exist yet.
    res_chunks = backfill_orphaned_chunks(dry_run=True, conn=conn)
    assert res_chunks["table"] == "chunks"

    res_embeddings = backfill_orphaned_embeddings(dry_run=True, conn=conn)
    assert res_embeddings["table"] == "embeddings"


@pytest.mark.asyncio
async def test_startup_event_skips_when_auto_migrate_disabled(
    monkeypatch: pytest.MonkeyPatch, temp_lancedb_dir
):
    """Startup should not create migration task when auto migration is disabled."""
    import importlib

    _patch_channel_modules_disabled(monkeypatch)
    web_app_module = importlib.import_module("xagent.web.app")

    class _FakeManager:
        async def initialize(self) -> None:
            return None

        async def list_skills(self) -> list[str]:
            return []

        async def list_templates(self) -> list[str]:
            return []

    class _FakeMemoryStoreManager:
        def get_store_info(self) -> dict[str, object]:
            return {
                "is_lancedb": True,
                "embedding_model_id": "test-model",
                "similarity_threshold": 0.5,
            }

    class _FakeSandboxManager:
        async def cleanup(self) -> None:
            return None

        async def warmup(self) -> None:
            return None

    class _FakeConn:
        pass

    migration_called = {"value": False}
    created_tasks: list[asyncio.Task] = []
    original_create_task = asyncio.create_task

    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "false")
    monkeypatch.setattr(web_app_module, "init_db", lambda: None)
    monkeypatch.setattr(web_app_module, "_migration_task", None)
    monkeypatch.setattr(
        "xagent.skills.utils.create_skill_manager",
        lambda: _FakeManager(),
    )
    monkeypatch.setattr(
        "xagent.templates.utils.create_template_manager",
        lambda: _FakeManager(),
    )
    monkeypatch.setattr(
        "xagent.web.dynamic_memory_store.get_memory_store_manager",
        lambda: _FakeMemoryStoreManager(),
    )
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager",
        lambda: _FakeSandboxManager(),
    )
    monkeypatch.setattr(
        "xagent.providers.vector_store.lancedb.get_connection_from_env",
        lambda: _FakeConn(),
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.check_table_needs_migration",
        lambda _conn, table_name: table_name == "chunks",
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils.list_embeddings_table_names",
        lambda _conn: [],
    )

    def _fake_backfill_all(*, dry_run: bool = False, conn=None) -> dict:
        migration_called["value"] = True
        return {
            "chunks": {"backfilled": 1, "skipped": 0, "failed": 0},
            "embeddings": {"backfilled": 0, "skipped": 0, "failed": 0},
            "locked": True,
        }

    async def _fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def _track_create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(
        "xagent.migrations.lancedb.backfill_user_id.backfill_all",
        _fake_backfill_all,
    )
    monkeypatch.setattr(web_app_module.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(web_app_module.asyncio, "create_task", _track_create_task)

    await web_app_module.startup_event()
    if created_tasks:
        await asyncio.gather(*created_tasks)

    assert created_tasks == []
    assert migration_called["value"] is False


@pytest.mark.asyncio
async def test_startup_event_triggers_background_auto_migration(
    monkeypatch: pytest.MonkeyPatch, temp_lancedb_dir
):
    """Startup should create task and execute backfill when enabled and needed."""
    import importlib

    _patch_channel_modules_disabled(monkeypatch)
    web_app_module = importlib.import_module("xagent.web.app")

    class _FakeManager:
        async def initialize(self) -> None:
            return None

        async def list_skills(self) -> list[str]:
            return []

        async def list_templates(self) -> list[str]:
            return []

    class _FakeMemoryStoreManager:
        def get_store_info(self) -> dict[str, object]:
            return {
                "is_lancedb": True,
                "embedding_model_id": "test-model",
                "similarity_threshold": 0.5,
            }

    class _FakeSandboxManager:
        async def cleanup(self) -> None:
            return None

        async def warmup(self) -> None:
            return None

    class _FakeConn:
        pass

    migration_called = {"value": False}
    created_tasks: list[asyncio.Task] = []
    original_create_task = asyncio.create_task

    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "true")
    monkeypatch.setattr(web_app_module, "init_db", lambda: None)
    monkeypatch.setattr(web_app_module, "_migration_task", None)
    monkeypatch.setattr(
        "xagent.skills.utils.create_skill_manager",
        lambda: _FakeManager(),
    )
    monkeypatch.setattr(
        "xagent.templates.utils.create_template_manager",
        lambda: _FakeManager(),
    )
    monkeypatch.setattr(
        "xagent.web.dynamic_memory_store.get_memory_store_manager",
        lambda: _FakeMemoryStoreManager(),
    )
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager",
        lambda: _FakeSandboxManager(),
    )
    monkeypatch.setattr(
        "xagent.providers.vector_store.lancedb.get_connection_from_env",
        lambda: _FakeConn(),
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.check_table_needs_migration",
        lambda _conn, table_name: table_name == "chunks",
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils.list_embeddings_table_names",
        lambda _conn: [],
    )

    def _fake_backfill_all(*, dry_run: bool = False, conn=None) -> dict:
        migration_called["value"] = True
        return {
            "chunks": {"backfilled": 1, "skipped": 0, "failed": 0},
            "embeddings": {"backfilled": 0, "skipped": 0, "failed": 0},
            "locked": True,
        }

    async def _fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def _track_create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(
        "xagent.migrations.lancedb.backfill_user_id.backfill_all",
        _fake_backfill_all,
    )
    monkeypatch.setattr(web_app_module.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(web_app_module.asyncio, "create_task", _track_create_task)

    await web_app_module.startup_event()
    if created_tasks:
        await asyncio.gather(*created_tasks)

    # We expect 2 tasks: backfill migration + uploaded files reconcile
    assert len(created_tasks) == 2
    assert migration_called["value"] is True


@pytest.mark.asyncio
async def test_startup_event_no_task_when_no_table_needs_migration(
    monkeypatch: pytest.MonkeyPatch, temp_lancedb_dir
):
    """Startup should not create migration task when no table needs migration."""
    import importlib

    _patch_channel_modules_disabled(monkeypatch)
    web_app_module = importlib.import_module("xagent.web.app")

    class _FakeManager:
        async def initialize(self) -> None:
            return None

        async def list_skills(self) -> list[str]:
            return []

        async def list_templates(self) -> list[str]:
            return []

    class _FakeMemoryStoreManager:
        def get_store_info(self) -> dict[str, object]:
            return {
                "is_lancedb": True,
                "embedding_model_id": "test-model",
                "similarity_threshold": 0.5,
            }

    class _FakeSandboxManager:
        async def cleanup(self) -> None:
            return None

        async def warmup(self) -> None:
            return None

    class _FakeConn:
        pass

    migration_called = {"value": False}
    created_tasks: list[asyncio.Task] = []
    original_create_task = asyncio.create_task

    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "true")
    monkeypatch.setattr(web_app_module, "init_db", lambda: None)
    monkeypatch.setattr(web_app_module, "_migration_task", None)
    monkeypatch.setattr(
        "xagent.skills.utils.create_skill_manager",
        lambda: _FakeManager(),
    )
    monkeypatch.setattr(
        "xagent.templates.utils.create_template_manager",
        lambda: _FakeManager(),
    )
    monkeypatch.setattr(
        "xagent.web.dynamic_memory_store.get_memory_store_manager",
        lambda: _FakeMemoryStoreManager(),
    )
    monkeypatch.setattr(
        "xagent.web.sandbox_manager.get_sandbox_manager",
        lambda: _FakeSandboxManager(),
    )
    monkeypatch.setattr(
        "xagent.providers.vector_store.lancedb.get_connection_from_env",
        lambda: _FakeConn(),
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.LanceDB.schema_manager.check_table_needs_migration",
        lambda _conn, _table_name: False,
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils.list_embeddings_table_names",
        lambda _conn: [],
    )

    def _fake_backfill_all(*, dry_run: bool = False, conn=None) -> dict:
        migration_called["value"] = True
        return {
            "chunks": {"backfilled": 0, "skipped": 0, "failed": 0},
            "embeddings": {"backfilled": 0, "skipped": 0, "failed": 0},
            "locked": True,
        }

    async def _fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def _track_create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(
        "xagent.migrations.lancedb.backfill_user_id.backfill_all",
        _fake_backfill_all,
    )
    monkeypatch.setattr(web_app_module.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(web_app_module.asyncio, "create_task", _track_create_task)

    await web_app_module.startup_event()
    if created_tasks:
        await asyncio.gather(*created_tasks)

    # We expect 1 task: uploaded files reconcile (even without migration needs)
    assert len(created_tasks) == 1
    assert migration_called["value"] is False
