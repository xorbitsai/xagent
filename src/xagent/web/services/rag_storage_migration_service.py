"""RAG storage migration startup service.

This module isolates backend-specific LanceDB migration checks from `app.py`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RAGStorageMigrationService:
    """Run storage compatibility checks and background migrations."""

    async def start_background_migrations(self) -> asyncio.Task[None]:
        """Schedule compatibility checks; backfill execution is env-gated."""
        return asyncio.create_task(self._run_migrations())

    async def _run_migrations(self) -> None:
        """Run migration checks and backfills in background."""
        auto_migrate = os.getenv("LANCEDB_AUTO_MIGRATE", "true").lower() == "true"

        try:
            await self._check_and_migrate_user_id(auto_migrate=auto_migrate)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not check LanceDB migration status: %s. "
                "Application will continue, but some features may not work correctly.",
                e,
            )

        if auto_migrate:
            try:
                await self._check_and_backfill_documents_table()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Could not check documents table backfill status: %s. "
                    "Application will continue.",
                    e,
                )

    async def _check_and_migrate_user_id(self, *, auto_migrate: bool) -> None:
        """Check user_id migrations and run backfill when needed."""
        from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
            check_table_needs_migration,
        )
        from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import (
            list_embeddings_table_names,
        )
        from ...migrations.lancedb.backfill_user_id import backfill_all
        from ...providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        tables_to_check = [
            "chunks",
            "documents",
            "parses",
            "ingestion_runs",
            "prompt_templates",
        ]

        tables_need_migration_list: list[str] = []
        for table_name in tables_to_check:
            if check_table_needs_migration(conn, table_name):
                logger.warning(
                    "Table '%s' needs migration (missing user_id field)",
                    table_name,
                )
                tables_need_migration_list.append(table_name)

        try:
            for table_name in list_embeddings_table_names(conn):
                if check_table_needs_migration(conn, table_name):
                    logger.warning(
                        "Table '%s' needs migration (missing user_id field)",
                        table_name,
                    )
                    tables_need_migration_list.append(table_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not check embeddings tables: %s", e)

        if not tables_need_migration_list:
            logger.info("LanceDB tables are up to date, no migration needed")
            return

        logger.warning(
            "Tables requiring migration: %s",
            ", ".join(tables_need_migration_list),
        )

        if not auto_migrate:
            logger.warning(
                "LANCEDB_AUTO_MIGRATE is disabled. "
                "Migration will NOT run automatically. "
                "To enable automatic migration, set LANCEDB_AUTO_MIGRATE=true. "
                "To run migration manually: python -m xagent.migrations.lancedb.backfill_user_id"
            )
            return

        logger.info("=" * 60)
        logger.info("STARTING BACKGROUND LANCEDB MIGRATION")
        logger.info("=" * 60)
        result = await asyncio.to_thread(backfill_all, dry_run=False)
        logger.info("=" * 60)
        logger.info("BACKGROUND LANCEDB MIGRATION COMPLETED")
        logger.info("=" * 60)
        logger.info(
            "Migration results: chunks=%s backfilled, embeddings=%s backfilled",
            result.get("chunks", {}).get("backfilled", 0),
            result.get("embeddings", {}).get("backfilled", 0),
        )

        chunks_skipped = result.get("chunks", {}).get("skipped", 0)
        embeddings_skipped = result.get("embeddings", {}).get("skipped", 0)
        if chunks_skipped > 0 or embeddings_skipped > 0:
            logger.warning(
                "Some records were skipped (no matching document): chunks=%s, embeddings=%s",
                chunks_skipped,
                embeddings_skipped,
            )

    async def _check_and_backfill_documents_table(self) -> None:
        """Fix nullability and backfill empty file_id / NULL user_id when needed."""
        from ...core.tools.core.RAG_tools.LanceDB.schema_manager import (
            _safe_close_table,
        )
        from ...core.tools.core.RAG_tools.utils.lancedb_query_utils import query_to_list
        from ...migrations.lancedb.backfill_documents_file_id import backfill_all
        from ...migrations.lancedb.fix_file_id_nullable import fix_file_id_nullable
        from ...providers.vector_store.lancedb import get_connection_from_env

        conn = get_connection_from_env()
        try:
            fix_result = fix_file_id_nullable(dry_run=False, conn=conn)
            if fix_result.get("fixed"):
                logger.info("Auto-fixed file_id column to nullable in documents table")
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not fix file_id nullability: %s", e)

        documents_table: Optional[Any] = None
        try:
            documents_table = conn.open_table("documents")
            empty_file_id_count = len(
                query_to_list(documents_table.search().where("file_id = ''").limit(1))
            )
            null_user_id_count = len(
                query_to_list(
                    documents_table.search().where("user_id IS NULL").limit(1)
                )
            )

            if empty_file_id_count == 0 and null_user_id_count == 0:
                logger.info("Documents table backfill not needed")
                return

            logger.info("=" * 60)
            logger.info("STARTING BACKGROUND DOCUMENTS TABLE BACKFILL")
            logger.info("=" * 60)

            result = await asyncio.to_thread(backfill_all, dry_run=False, conn=conn)
            logger.info("=" * 60)
            logger.info("DOCUMENTS TABLE BACKFILL COMPLETED")
            logger.info("=" * 60)

            file_id_result = result.get("file_id", {})
            user_id_result = result.get("user_id", {})
            if file_id_result.get("updated", 0) > 0:
                logger.info(
                    "file_id backfill: %d rows updated",
                    file_id_result.get("updated", 0),
                )
            if user_id_result.get("updated", 0) > 0:
                logger.info(
                    "user_id backfill: %d rows updated",
                    user_id_result.get("updated", 0),
                )
            if file_id_result.get("error"):
                logger.warning(
                    "file_id backfill error: %s", file_id_result.get("error")
                )
            if user_id_result.get("error"):
                logger.warning(
                    "user_id backfill error: %s", user_id_result.get("error")
                )
        except Exception as e:  # noqa: BLE001
            error_message = str(e).lower()
            # Table may legitimately not exist in early bootstrap or fresh DB states.
            if "not found" in error_message or "does not exist" in error_message:
                logger.debug("Documents table does not exist yet: %s", e)
            else:
                logger.error("=" * 60)
                logger.error("DOCUMENTS TABLE BACKFILL FAILED")
                logger.error("=" * 60)
                logger.error("Error: %s", e, exc_info=True)
                logger.warning(
                    "Some features may not work correctly. "
                    "Please run backfill manually: python -m xagent.migrations.lancedb.backfill_documents_file_id"
                )
        finally:
            _safe_close_table(documents_table)
