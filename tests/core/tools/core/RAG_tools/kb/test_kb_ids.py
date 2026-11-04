"""One stable kb_id per (collection name, owner), read in the caller's scope."""

from __future__ import annotations

import errno
import multiprocessing
import threading
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
import pytest
from filelock import FileLock

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    ConfigurationError,
    DatabaseOperationError,
)
from xagent.core.tools.core.RAG_tools.kb import kb_ids
from xagent.core.tools.core.RAG_tools.kb.kb_ids import (
    delete_kb_ids,
    get_or_create_kb_id,
    read_kb_ids,
    rename_kb_ids,
    set_kb_ids_collection,
)
from xagent.core.tools.core.RAG_tools.LanceDB.schema_manager import KB_IDS_TABLE
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store
from xagent.core.tools.core.RAG_tools.storage.vector_backend import (
    lock_deployment_kb_engine,
)


@pytest.fixture
def conn() -> Any:
    return get_vector_index_store().get_raw_connection()


def test_each_owner_of_a_name_gets_its_own_stable_kb_id(conn: Any) -> None:
    mine = get_or_create_kb_id(conn, "kb", 1)
    theirs = get_or_create_kb_id(conn, "kb", 2)
    ownerless = get_or_create_kb_id(conn, "kb", None)
    other = get_or_create_kb_id(conn, "other", 1)

    assert len({mine, theirs, ownerless, other}) == 4
    assert get_or_create_kb_id(conn, "kb", 1) == mine
    assert get_or_create_kb_id(conn, "kb", None) == ownerless
    assert read_kb_ids(conn, user_id=1, is_admin=False) == {
        "kb": [mine],
        "other": [other],
    }
    admin = read_kb_ids(conn, user_id=None, is_admin=True, collection="kb")
    assert admin.keys() == {"kb"}
    assert sorted(admin["kb"]) == sorted([mine, theirs, ownerless])
    assert read_kb_ids(conn, user_id=3, is_admin=False) == {}
    assert read_kb_ids(conn, user_id=None, is_admin=False) == {}


def _first_write(db_dir: str, start: Any) -> str:
    start.wait()
    return get_or_create_kb_id(lancedb.connect(db_dir), "new", 1)


@pytest.mark.parametrize(
    "workers", ["threads", pytest.param("processes", marks=pytest.mark.slow)]
)
def test_concurrent_first_writes_agree_on_one_kb_id(conn: Any, workers: str) -> None:
    writers = 6
    spawn = multiprocessing.get_context("spawn")
    with ExitStack() as stack:
        if workers == "threads":
            start: Any = threading.Barrier(writers)
            pool: Executor = ThreadPoolExecutor(writers)
        else:
            start = stack.enter_context(spawn.Manager()).Barrier(writers)
            pool = ProcessPoolExecutor(writers, mp_context=spawn)
        with pool:
            kb_ids = set(
                pool.map(_first_write, [conn.uri] * writers, [start] * writers)
            )

    assert len(kb_ids) == 1
    assert conn.open_table(KB_IDS_TABLE).count_rows() == 1


def test_a_kb_id_written_through_another_connection_is_reused(conn: Any) -> None:
    assert read_kb_ids(conn, user_id=1, is_admin=False) == {}
    elsewhere = get_or_create_kb_id(lancedb.connect(conn.uri), "kb", 1)

    assert get_or_create_kb_id(conn, "kb", 1) == elsewhere


def _add_row(conn: Any, collection: str, user_id: int, kb_id: str) -> None:
    conn.open_table(KB_IDS_TABLE).add(
        [
            {
                "collection": collection,
                "user_id": user_id,
                "kb_id": kb_id,
                "created_at": datetime.now(timezone.utc),
            }
        ]
    )


def test_several_kb_ids_for_one_owner_are_refused(conn: Any) -> None:
    get_or_create_kb_id(conn, "kb", 1)
    _add_row(conn, "kb", 1, "duplicate")

    with pytest.raises(DatabaseOperationError, match="several kb_ids"):
        get_or_create_kb_id(conn, "kb", 1)
    with pytest.raises(DatabaseOperationError, match="several kb_ids"):
        read_kb_ids(conn, user_id=None, is_admin=True)


def test_the_duplicate_error_lists_only_the_duplicated_rows(conn: Any) -> None:
    original = get_or_create_kb_id(conn, "kb", 1)
    unrelated = get_or_create_kb_id(conn, "unrelated", 1)
    _add_row(conn, "kb", 1, "duplicate")

    with pytest.raises(DatabaseOperationError) as caught:
        read_kb_ids(conn, user_id=None, is_admin=True)

    message = str(caught.value)
    assert original in message and "duplicate" in message
    assert unrelated not in message and "unrelated" not in message


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("denied"),
        NotImplementedError(),
        OSError(errno.EROFS, "read-only file system"),
        OSError(errno.ENOSPC, "no space left on device"),
        FileNotFoundError(errno.ENOENT, "no such file or directory"),
    ],
    ids=["EACCES", "no-locking", "EROFS", "ENOSPC", "ENOENT"],
)
def test_a_lock_that_cannot_be_taken_fails_naming_the_lock_file(
    conn: Any, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    class BrokenLock:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def acquire(self) -> None:
            raise error

    monkeypatch.setattr(kb_ids, "FileLock", BrokenLock)

    with pytest.raises(DatabaseOperationError, match=r"kb_ids\.lock") as caught:
        get_or_create_kb_id(conn, "kb", 1)
    assert str(Path(conn.uri) / f"{KB_IDS_TABLE}.lock") in str(caught.value)
    assert caught.value.__cause__ is error
    monkeypatch.undo()
    assert read_kb_ids(conn, user_id=None, is_admin=True) == {}


def test_a_lock_held_too_long_fails_naming_the_lock_file(
    conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kb_ids, "_LOCK_TIMEOUT_SECONDS", 0.1)
    path = str(Path(conn.uri) / f"{KB_IDS_TABLE}.lock")

    with FileLock(path):
        with pytest.raises(DatabaseOperationError, match=r"kb_ids\.lock") as caught:
            get_or_create_kb_id(conn, "kb", 1)
    assert path in str(caught.value)
    assert read_kb_ids(conn, user_id=None, is_admin=True) == {}


def test_a_kb_id_marks_the_deployment_as_milvus(conn: Any) -> None:
    get_or_create_kb_id(conn, "kb", 1)

    with pytest.raises(ConfigurationError, match=r"is milvus \(kb_ids hold data\)"):
        lock_deployment_kb_engine()


def _owners(conn: Any, collection: str) -> dict[int | None, str]:
    rows = conn.open_table(KB_IDS_TABLE).to_arrow().to_pylist()
    return {
        row["user_id"]: row["kb_id"] for row in rows if row["collection"] == collection
    }


def test_delete_removes_the_callers_row_or_every_owners_for_an_admin(conn: Any) -> None:
    for owner in (1, 2, None):
        get_or_create_kb_id(conn, "kb", owner)
    keep = get_or_create_kb_id(conn, "other", 1)

    assert delete_kb_ids(conn, "kb", user_id=1, is_admin=False) == 1
    assert delete_kb_ids(conn, "kb", user_id=1, is_admin=False) == 0
    assert set(_owners(conn, "kb")) == {2, None}
    assert delete_kb_ids(conn, "kb", user_id=None, is_admin=True) == 2
    assert _owners(conn, "kb") == {}
    assert _owners(conn, "other") == {1: keep}


def test_delete_on_a_deployment_without_the_table_deletes_nothing(conn: Any) -> None:
    assert delete_kb_ids(conn, "kb", user_id=1, is_admin=False) == 0


def test_a_caller_without_a_user_changes_no_row_but_an_admin_reaches_ownerless_rows(
    conn: Any,
) -> None:
    ownerless = get_or_create_kb_id(conn, "kb", None)
    get_or_create_kb_id(conn, "new", None)
    mine = get_or_create_kb_id(conn, "kb", 1)

    assert delete_kb_ids(conn, "kb", user_id=None, is_admin=False) == 0
    assert rename_kb_ids(conn, "kb", "new", user_id=None, is_admin=False) == []
    assert _owners(conn, "kb") == {None: ownerless, 1: mine}

    renamed = rename_kb_ids(conn, "kb", "new", user_id=None, is_admin=True)

    assert sorted(renamed) == sorted([ownerless, mine])
    assert _owners(conn, "new") == {None: ownerless, 1: mine}
    assert conn.open_table(KB_IDS_TABLE).count_rows() == 2


def test_a_tenant_rename_moves_only_the_callers_row_and_keeps_its_kb_id(
    conn: Any,
) -> None:
    mine = get_or_create_kb_id(conn, "kb", 1)
    theirs = get_or_create_kb_id(conn, "kb", 2)

    assert rename_kb_ids(conn, "kb", "new", user_id=1, is_admin=False) == [mine]

    assert _owners(conn, "new") == {1: mine}
    assert _owners(conn, "kb") == {2: theirs}
    assert get_or_create_kb_id(conn, "new", 1) == mine


def test_an_admin_rename_moves_every_owners_row(conn: Any) -> None:
    ids = {owner: get_or_create_kb_id(conn, "kb", owner) for owner in (1, 2, None)}

    renamed = rename_kb_ids(conn, "kb", "new", user_id=None, is_admin=True)

    assert sorted(renamed) == sorted(ids.values())
    assert _owners(conn, "new") == ids and _owners(conn, "kb") == {}


def test_a_rename_removes_the_owners_leftover_row_under_the_target_name(
    conn: Any,
) -> None:
    mine = get_or_create_kb_id(conn, "kb", 1)
    get_or_create_kb_id(conn, "new", 1)
    other = get_or_create_kb_id(conn, "new", 2)

    rename_kb_ids(conn, "kb", "new", user_id=1, is_admin=False)

    assert _owners(conn, "new") == {1: mine, 2: other}
    assert read_kb_ids(conn, user_id=1, is_admin=False) == {"new": [mine]}


def test_a_rename_of_a_name_without_rows_changes_nothing(conn: Any) -> None:
    keep = get_or_create_kb_id(conn, "new", 1)

    assert rename_kb_ids(conn, "kb", "new", user_id=1, is_admin=False) == []

    assert _owners(conn, "new") == {1: keep}


def test_the_rows_can_be_moved_back_by_their_kb_ids(conn: Any) -> None:
    mine = get_or_create_kb_id(conn, "kb", 1)
    theirs = get_or_create_kb_id(conn, "kb", 2)
    renamed = rename_kb_ids(conn, "kb", "new", user_id=1, is_admin=False)

    set_kb_ids_collection(conn, renamed, "kb")
    set_kb_ids_collection(conn, [], "ignored")

    assert _owners(conn, "kb") == {1: mine, 2: theirs} and _owners(conn, "new") == {}


@pytest.mark.parametrize(
    "change",
    [
        lambda conn: delete_kb_ids(conn, "kb", user_id=1, is_admin=False),
        lambda conn: rename_kb_ids(conn, "kb", "new", user_id=1, is_admin=False),
        lambda conn: set_kb_ids_collection(conn, ["x"], "new"),
    ],
    ids=["delete", "rename", "set"],
)
def test_deletes_and_renames_wait_for_the_lock_of_the_first_write(
    conn: Any, monkeypatch: pytest.MonkeyPatch, change: Any
) -> None:
    monkeypatch.setattr(kb_ids, "_LOCK_TIMEOUT_SECONDS", 0.1)
    get_or_create_kb_id(conn, "kb", 1)

    with FileLock(str(Path(conn.uri) / f"{KB_IDS_TABLE}.lock")):
        with pytest.raises(DatabaseOperationError, match=r"kb_ids\.lock"):
            change(conn)

    assert _owners(conn, "kb").keys() == {1}
