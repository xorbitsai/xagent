"""ensure_sqlite_parent_directory — fresh-install SQLite bootstrap.

sqlite3 creates a missing database file on connect but not missing parent
directories, so on a fresh install the default ``~/.xagent`` storage root made
the very first connection fail with "unable to open database file".
"""

from __future__ import annotations

from sqlalchemy import create_engine

from xagent.db.sqlite import ensure_sqlite_parent_directory


def test_creates_missing_parent_directories(tmp_path) -> None:
    db_path = tmp_path / "storage-root" / "nested" / "xagent.db"
    url = f"sqlite:///{db_path}"

    ensure_sqlite_parent_directory(url)

    assert db_path.parent.is_dir()
    engine = create_engine(url)
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT 1").scalar() == 1
    engine.dispose()


def test_existing_parent_directory_is_left_alone(tmp_path) -> None:
    db_path = tmp_path / "xagent.db"
    db_path.write_bytes(b"")

    ensure_sqlite_parent_directory(f"sqlite:///{db_path}")

    assert db_path.exists()


def test_ignores_in_memory_and_non_sqlite_urls(tmp_path) -> None:
    ensure_sqlite_parent_directory("sqlite:///:memory:")
    ensure_sqlite_parent_directory("sqlite://")
    ensure_sqlite_parent_directory("sqlite:///file:shared?mode=memory&uri=true")
    ensure_sqlite_parent_directory("postgresql://user:pw@localhost/xagent")
