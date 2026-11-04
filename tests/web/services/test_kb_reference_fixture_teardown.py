"""Temporary module databases must not affect later KB reference writers."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.web.services import test_read_surface_shape_contract as surface
from xagent.core.tools.core.RAG_tools.file.register_document import register_document
from xagent.core.tools.core.RAG_tools.storage.lancedb_stores import (
    LanceDBVectorIndexStore,
)
from xagent.web.models import database


@pytest.mark.parametrize("setup_failure", [False, True])
def test_module_database_teardown_preserves_standalone_registration(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    setup_failure: bool,
) -> None:
    monkeypatch.setattr(database, "_SessionLocal", None)
    monkeypatch.setattr(database, "_engine", None)
    initialize = surface.init_db
    if setup_failure:

        def fail_after_binding(**kwargs):
            initialize(**kwargs)
            raise RuntimeError("module setup failed")

        monkeypatch.setattr(surface, "init_db", fail_after_binding)
    fixture_request = Mock(spec=pytest.FixtureRequest)
    lifecycle = surface._environment.__wrapped__(tmp_path_factory, fixture_request)
    try:
        if setup_failure:
            with pytest.raises(RuntimeError, match="module setup failed"):
                next(lifecycle)
        else:
            next(lifecycle)
            with pytest.raises(StopIteration):
                next(lifecycle)
    finally:
        lifecycle.close()
        for call in reversed(fixture_request.addfinalizer.call_args_list):
            call.args[0]()

    assert database.get_optional_session_local() is None
    source = tmp_path / "standalone.txt"
    source.write_text("A document registered after the module database closes.")
    result = register_document(
        collection="standalone", source_path=str(source), file_id="standalone-file"
    )
    assert result["created"] is True
    records = LanceDBVectorIndexStore().list_document_records_by_file_ids(
        ["standalone-file"]
    )
    assert [row.doc_id for row in records] == [result["doc_id"]]


def test_pool_monkeypatch_teardown_preserves_later_kb_ingest(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "tests/web/api/test_agents_management.py::test_create_from_template_releases_request_session_before_async_runtime",
            "tests/web/api/test_kb_raised_ingest_identity.py::test_ingest_raise_after_registration_removes_the_new_document",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "XAGENT_STORAGE_ROOT": str(tmp_path / "storage")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    failures = [
        line for line in result.stdout.splitlines() if line.startswith("FAILED ")
    ]
    exit_code = result.returncode
    assert exit_code == 0, "\n".join(failures)
