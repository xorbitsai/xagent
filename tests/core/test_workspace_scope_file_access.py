"""Scope confinement for file-id resolution in TaskWorkspace.

`_file_record_allowed_for_workspace` gates `resolve_file_id`. Before this
change it keyed only on the owner `user_id` (+ optional task_id), so a scoped
task could resolve, by file_id, a record belonging to the same owner but a
different scope subtree (e.g. another end user's upload, `task_id=None`).
These tests pin the scope-subtree confinement, and confirm unscoped workspaces
keep the owner-only behavior byte-for-byte.
"""

from types import SimpleNamespace

from xagent.core.workspace import TaskWorkspace

_SEGMENTS = ("clients", "3", "end_users", "7")


def _scoped_workspace(tmp_path, owner=5):
    base_dir = tmp_path / "user_5" / "clients" / "3" / "end_users" / "7"
    ws = TaskWorkspace("web_task_10", str(base_dir), scope_segments=_SEGMENTS)
    ws.owner_user_id = owner
    return ws


def _record(**kwargs):
    kwargs.setdefault("storage_path", None)
    kwargs.setdefault("storage_key", None)
    return SimpleNamespace(**kwargs)


def test_scoped_allows_record_in_own_scope_subtree(tmp_path):
    ws = _scoped_workspace(tmp_path)
    record = _record(
        user_id=5,
        task_id=None,
        storage_key="users/5/clients/3/end_users/7/uploads/f1/report.txt",
    )
    assert ws._file_record_allowed_for_workspace(record) is True


def test_scoped_rejects_sibling_end_user_record(tmp_path):
    ws = _scoped_workspace(tmp_path)
    # Same owner, different end user (8) — the pre-fix leak.
    record = _record(
        user_id=5,
        task_id=None,
        storage_key="users/5/clients/3/end_users/8/uploads/f1/secret.txt",
    )
    assert ws._file_record_allowed_for_workspace(record) is False


def test_scoped_rejects_owner_unscoped_upload(tmp_path):
    ws = _scoped_workspace(tmp_path)
    # Creator's own unscoped upload (task_id=None) — allowed pre-fix, now denied.
    record = _record(
        user_id=5,
        task_id=None,
        storage_key="users/5/uploads/f1/creator.txt",
    )
    assert ws._file_record_allowed_for_workspace(record) is False


def test_scoped_rejects_different_owner(tmp_path):
    ws = _scoped_workspace(tmp_path)
    record = _record(
        user_id=6,
        task_id=None,
        storage_key="users/6/clients/3/end_users/7/uploads/f1/x.txt",
    )
    assert ws._file_record_allowed_for_workspace(record) is False


def test_scoped_path_based_containment(tmp_path):
    ws = _scoped_workspace(tmp_path)
    in_scope = (
        tmp_path / "user_5" / "clients" / "3" / "end_users" / "7" / "uploads" / "a.txt"
    )
    sibling = (
        tmp_path / "user_5" / "clients" / "3" / "end_users" / "8" / "uploads" / "a.txt"
    )

    in_record = _record(user_id=5, task_id=None, storage_path=str(in_scope))
    sibling_record = _record(user_id=5, task_id=None, storage_path=str(sibling))

    assert ws._file_record_allowed_for_workspace(in_record) is True
    assert ws._file_record_allowed_for_workspace(sibling_record) is False


def test_scoped_path_under_workspace_dir_allowed(tmp_path):
    ws = _scoped_workspace(tmp_path)
    inside = ws.workspace_dir / "output" / "result.txt"
    record = _record(user_id=5, task_id=None)
    # The explicit path lands inside the task workspace → always allowed.
    assert ws._file_record_allowed_for_workspace(record, inside) is True


def test_scoped_rejects_in_scope_key_with_out_of_scope_local_path(tmp_path):
    ws = _scoped_workspace(tmp_path)
    # A record whose durable key is in scope but whose local storage_path
    # points at a sibling scope. resolve_file_id's local-path branch hands
    # back the local path, so an in-scope key must NOT admit it.
    out_of_scope = (
        tmp_path
        / "user_5"
        / "clients"
        / "3"
        / "end_users"
        / "8"
        / "uploads"
        / "leak.txt"
    )
    record = _record(
        user_id=5,
        task_id=None,
        storage_key="users/5/clients/3/end_users/7/uploads/f1/report.txt",
        storage_path=str(out_of_scope),
    )
    # Called as resolve_file_id's local-path branch does: explicit path == the
    # local storage_path that would be returned.
    assert ws._file_record_allowed_for_workspace(record, out_of_scope) is False


def test_scoped_durable_key_not_rejected_by_out_of_scope_local_path(tmp_path):
    ws = _scoped_workspace(tmp_path)
    # Same divergent record, but reached via the durable-materialize branch
    # (no explicit path): resolution builds a fresh path from the in-scope
    # key, so the stale/out-of-scope local storage_path must not cause a
    # false rejection.
    out_of_scope = (
        tmp_path
        / "user_5"
        / "clients"
        / "3"
        / "end_users"
        / "8"
        / "uploads"
        / "leak.txt"
    )
    record = _record(
        user_id=5,
        task_id=None,
        storage_key="users/5/clients/3/end_users/7/uploads/f1/report.txt",
        storage_path=str(out_of_scope),
    )
    assert ws._file_record_allowed_for_workspace(record) is True


def test_unscoped_workspace_keeps_owner_only_behavior(tmp_path):
    # No scope_segments → the new confinement is skipped; owner + task_id only.
    ws = TaskWorkspace("web_task_10", str(tmp_path))
    ws.owner_user_id = 5
    record = _record(
        user_id=5,
        task_id=None,
        storage_key="users/5/uploads/f1/creator.txt",
    )
    assert ws._file_record_allowed_for_workspace(record) is True

    foreign = _record(user_id=6, task_id=None, storage_key="users/6/uploads/f/x.txt")
    assert ws._file_record_allowed_for_workspace(foreign) is False
