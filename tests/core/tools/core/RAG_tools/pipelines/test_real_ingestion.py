"""
E2E test for the real document ingestion pipeline.

This test is not meant for CI/CD but serves as a utility to run the full
document ingestion pipeline on a real PDF. Its primary purpose is to generate
a valid LanceDB database artifact that can be used for subsequent, separate
testing of the retrieval pipeline.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import pytest

from xagent.core.storage.manager import initialize_storage_manager
from xagent.core.tools.core.RAG_tools.core.schemas import IngestionConfig, ParseMethod
from xagent.core.tools.core.RAG_tools.pipelines import document_ingestion
from xagent.core.tools.core.RAG_tools.utils import model_resolver

logger = logging.getLogger(__name__)


def _resolve_embedding_model_id_or_skip() -> str:
    """Resolve a real embedding model id, or skip with actionable guidance."""
    # NOTE:
    # This utility E2E test defaults to DashScope-style embedding setup.
    # We intentionally pin `text-embedding-v4` as the default model id here.
    # If you are using a different API provider, update this model id to a
    # provider-compatible embedding model before running the test.
    preferred_embedding_model_id = "text-embedding-v4"
    try:
        embedding_cfg, _ = model_resolver.resolve_embedding_adapter(
            preferred_embedding_model_id
        )
        logger.info(
            "[REAL_INGESTION_TEST] Resolved embedding model: id=%s, provider=%s, model_name=%s",
            embedding_cfg.id,
            embedding_cfg.model_provider,
            embedding_cfg.model_name,
        )
        return embedding_cfg.id
    except Exception as exc:
        logger.warning(
            "[REAL_INGESTION_TEST] Cannot resolve embedding model from ModelHub/env: %s",
            exc,
        )
        logger.warning(
            "[REAL_INGESTION_TEST] Test default model is '%s'.",
            preferred_embedding_model_id,
        )
        logger.warning(
            "[REAL_INGESTION_TEST] To run this test, ensure API key is valid and model id matches your provider."
        )
        logger.warning(
            "[REAL_INGESTION_TEST] If needed, update preferred_embedding_model_id in this test file."
        )
        pytest.skip(
            f"No resolvable real embedding model for '{preferred_embedding_model_id}'."
        )


def test_run_real_ingestion_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Runs the real document ingestion pipeline to generate a DB artifact.
    This test's main goal is to produce a side-effect (the DB) for manual
    cross-pipeline testing.
    """
    # 1. --- Environment Setup ---
    # Use a predictable, persistent path for the database so we can access it later.
    # This path will be relative to the xagent project root.
    # NOTE:
    # Use a subdirectory under tmp_path to avoid different tests/workers (xdist)
    # sharing the same LanceDB directory, which can cause schema and file conflicts.
    # Previously we used a fixed path under the project root, which led to state
    # pollution and race conditions when tests ran in parallel.
    db_output_dir = tmp_path / "generated_db_for_test"
    db_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        monkeypatch.setenv("LANCEDB_DIR", str(db_output_dir.resolve()))

        storage_root = tmp_path / "storage"
        uploads_dir = storage_root / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        initialize_storage_manager(str(storage_root), str(uploads_dir))

        embedding_model_id = _resolve_embedding_model_id_or_skip()

        logger.info("--- E2E Ingestion Pipeline Runner ---")
        logger.info(f"[*] Using output LanceDB directory: {db_output_dir.resolve()}")
        logger.info(
            "[*] Using real embedding model id: %s (no stub adapter)",
            embedding_model_id,
        )

        # 2. --- Pipeline Execution ---
        # Robustly locate the project root relative to this test file
        project_root = Path(__file__).resolve().parents[6]
        test_pdf = project_root / "tests" / "resources" / "test_files" / "test.pdf"
        collection = f"test_collection_{uuid.uuid4().hex[:8]}"

        logger.info(f"[*] Ingesting document: {test_pdf}")
        logger.info(f"[*] Target collection: {collection}")

        # Call the real pipeline function
        result = document_ingestion.process_document(
            collection=collection,
            source_path=str(test_pdf),
            config=IngestionConfig(
                embedding_model_id=embedding_model_id,
                parse_method=ParseMethod.PYPDF,
            ),
            user_id=1,
            is_admin=True,
        )

        # 3. --- Log Results ---
        logger.info("--- Ingestion Result ---")
        logger.info(f"[*] Status: {result.status}")
        logger.info(f"[*] Message: {result.message}")
        logger.info(f"[*] Doc ID: {result.doc_id}")
        completed_steps = [step.name for step in result.completed_steps]
        logger.info(f"[*] Steps completed: {completed_steps}")

        if result.failed_step:
            logger.error(f"[!] FAILED at step: {result.failed_step}")

        # Final check to ensure the test framework knows if it succeeded.
        assert result.status == "success", "Document ingestion pipeline failed."

        logger.info("\n\n[SUCCESS] Pipeline finished.")
        logger.info("The generated database is available for the next step at:")
        logger.info(f"==> {db_output_dir.resolve()}\n")
    finally:
        # Cleanup generated database directory
        if db_output_dir.exists():
            logger.info(
                f"[*] Cleaning up generated database: {db_output_dir.resolve()}"
            )
            shutil.rmtree(db_output_dir)
