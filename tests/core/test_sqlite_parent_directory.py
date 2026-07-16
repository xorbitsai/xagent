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


def test_relative_path_without_parent_segment_is_a_noop(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    ensure_sqlite_parent_directory("sqlite:///relative.db")

    # Parent of "relative.db" is "." — nothing gets created, nothing raises.
    assert not (tmp_path / "relative.db").exists()


def test_tilde_path_expands_to_home_and_connects(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    url = ensure_sqlite_parent_directory("sqlite:///~/storage-root/xagent.db")

    # The returned URL must point at the expanded path (sqlite3 does not
    # expand ~), so connecting with it works end-to-end.
    assert (tmp_path / "storage-root").is_dir()
    engine = create_engine(url)
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT 1").scalar() == 1
    engine.dispose()
    assert (tmp_path / "storage-root" / "xagent.db").exists()


def test_plain_urls_are_returned_unchanged(tmp_path) -> None:
    url = f"sqlite:///{tmp_path / 'xagent.db'}"

    assert ensure_sqlite_parent_directory(url) == url
    assert ensure_sqlite_parent_directory("sqlite:///:memory:") == "sqlite:///:memory:"
