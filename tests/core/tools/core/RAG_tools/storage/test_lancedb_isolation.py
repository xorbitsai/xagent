"""Tests to ensure pytest does not pollute the default LanceDB directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import (
    ensure_documents_table,
)
from xagent.core.tools.core.RAG_tools.storage import (
    get_vector_index_store,
    reset_rag_storage_for_tests,
)
from xagent.providers.vector_store.lancedb import LanceDBConnectionManager


def test_tests_do_not_pollute_default_lancedb_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Creating tables in tests should not touch the default LanceDB directory.

    This test explicitly forces `LANCEDB_DIR` to a temporary directory to
    avoid relying on any developer machine environment settings.
    """
    expected_dir = tmp_path / "lancedb"
    expected_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LANCEDB_DIR", str(expected_dir))
    reset_rag_storage_for_tests()

    default_dir = Path(LanceDBConnectionManager.get_default_lancedb_dir())
    default_exists_before = default_dir.exists()
    default_listing_before = (
        {p.name for p in default_dir.iterdir()} if default_exists_before else set()
    )

    # Trigger a write path that creates tables in the isolated test directory.
    conn = get_vector_index_store().get_raw_connection()
    ensure_documents_table(conn)

    default_exists_after = default_dir.exists()
    default_listing_after = (
        {p.name for p in default_dir.iterdir()} if default_exists_after else set()
    )

    assert default_exists_after == default_exists_before
    assert default_listing_after == default_listing_before
