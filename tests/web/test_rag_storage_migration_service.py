from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from xagent.web.services import rag_storage_migration_service
from xagent.web.services.rag_storage_migration_service import RAGStorageMigrationService


@pytest.mark.asyncio
async def test_start_background_migrations_creates_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RAGStorageMigrationService()
    called = {"run": 0}

    async def _fake_run() -> None:
        called["run"] += 1

    monkeypatch.setattr(service, "_run_migrations", _fake_run)

    task = await service.start_background_migrations()
    assert task is not None
    await task

    assert isinstance(task, asyncio.Task)
    assert called["run"] == 1


@pytest.mark.asyncio
async def test_run_migrations_skips_documents_backfill_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RAGStorageMigrationService()
    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "false")
    called = {"user_id": 0, "docs": 0}

    async def _fake_user_id(*, auto_migrate: bool) -> None:
        assert auto_migrate is False
        called["user_id"] += 1

    async def _fake_docs() -> None:
        called["docs"] += 1

    monkeypatch.setattr(service, "_check_and_migrate_user_id", _fake_user_id)
    monkeypatch.setattr(service, "_check_and_backfill_documents_table", _fake_docs)

    await service._run_migrations()

    assert called["user_id"] == 1
    assert called["docs"] == 0


@pytest.mark.asyncio
async def test_run_migrations_runs_documents_backfill_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RAGStorageMigrationService()
    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "true")
    called = {"user_id": 0, "docs": 0}

    async def _fake_user_id(*, auto_migrate: bool) -> None:
        assert auto_migrate is True
        called["user_id"] += 1

    async def _fake_docs() -> None:
        called["docs"] += 1

    monkeypatch.setattr(service, "_check_and_migrate_user_id", _fake_user_id)
    monkeypatch.setattr(service, "_check_and_backfill_documents_table", _fake_docs)

    await service._run_migrations()

    assert called["user_id"] == 1
    assert called["docs"] == 1


@pytest.mark.asyncio
async def test_start_background_migrations_always_schedules_task_when_auto_migrate_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatibility checks must run even when LANCEDB_AUTO_MIGRATE=false."""
    service = RAGStorageMigrationService()
    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "false")
    called = {"run": 0}

    async def _fake_run() -> None:
        called["run"] += 1

    monkeypatch.setattr(service, "_run_migrations", _fake_run)

    task = await service.start_background_migrations()
    assert task is not None
    await task
    assert called["run"] == 1


@pytest.mark.asyncio
async def test_documents_backfill_missing_table_logs_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RAGStorageMigrationService()
    mock_conn = Mock()
    mock_conn.open_table.side_effect = RuntimeError("table does not exist")

    mock_logger = Mock()
    monkeypatch.setattr(rag_storage_migration_service, "logger", mock_logger)
    monkeypatch.setattr(
        "xagent.providers.vector_store.lancedb.get_connection_from_env",
        lambda: mock_conn,
    )
    monkeypatch.setattr(
        "xagent.migrations.lancedb.fix_file_id_nullable.fix_file_id_nullable",
        lambda dry_run, conn: {},
    )

    await service._check_and_backfill_documents_table()
    assert mock_logger.debug.called
    assert not mock_logger.error.called


@pytest.mark.asyncio
async def test_documents_backfill_runtime_error_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = RAGStorageMigrationService()
    mock_conn = Mock()
    mock_table = Mock()
    mock_query = Mock()
    mock_where = Mock()
    mock_limit = Mock()
    mock_limit.side_effect = RuntimeError("query crashed")
    mock_where.limit = mock_limit
    mock_query.where.return_value = mock_where
    mock_table.search.return_value = mock_query
    mock_conn.open_table.return_value = mock_table

    mock_logger = Mock()
    monkeypatch.setattr(rag_storage_migration_service, "logger", mock_logger)
    monkeypatch.setattr(
        "xagent.providers.vector_store.lancedb.get_connection_from_env",
        lambda: mock_conn,
    )
    monkeypatch.setattr(
        "xagent.migrations.lancedb.fix_file_id_nullable.fix_file_id_nullable",
        lambda dry_run, conn: {},
    )
    monkeypatch.setattr(
        "xagent.core.tools.core.RAG_tools.utils.lancedb_query_utils.query_to_list",
        lambda *_args, **_kwargs: [],
    )

    await service._check_and_backfill_documents_table()
    assert mock_logger.error.called
    assert mock_logger.warning.called
