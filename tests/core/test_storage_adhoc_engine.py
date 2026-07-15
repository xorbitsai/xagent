"""Tests for the shared ad-hoc engine behind create_db_session (issue #889)."""

import pytest

from xagent.core.storage import manager as storage_manager


@pytest.fixture(autouse=True)
def _reset_adhoc_engine():
    """Isolate the module-level engine cache between tests."""

    def reset():
        engine = storage_manager._adhoc_engine
        storage_manager._adhoc_engine = None
        storage_manager._adhoc_engine_url = None
        if engine is not None:
            engine.dispose()

    reset()
    yield
    reset()


def test_create_db_session_reuses_one_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/a.db")

    db1 = storage_manager.create_db_session()
    db2 = storage_manager.create_db_session()
    try:
        assert db1.get_bind() is db2.get_bind()
    finally:
        db1.close()
        db2.close()


def test_engine_swaps_when_database_url_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/a.db")
    db1 = storage_manager.create_db_session()
    bind1 = db1.get_bind()
    db1.close()

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/b.db")
    db2 = storage_manager.create_db_session()
    try:
        assert db2.get_bind() is not bind1
    finally:
        db2.close()
