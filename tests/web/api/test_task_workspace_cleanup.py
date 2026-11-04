"""Deleting a task's workspace finds it whichever spelling wrote it.

``_cleanup_workspace_directory`` runs when no agent is in memory, so it has
to locate the workspace from configuration alone, and ``TaskWorkspace``'s
constructor creates the tree it is pointed at -- so the candidate probe has to
be side-effect free, or the first candidate always "exists" and the real
workspace is never touched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.shared.execution_scope import register_scope_resolver
from xagent.core import workspace as workspace_module
from xagent.core.execution_scope import (
    ExecutionScope,
    ExecutionScopeContext,
    set_execution_scope_snapshot_loader,
)
from xagent.web.services.agent_service_manager import AgentServiceManager
from xagent.web.services.task_workspace_cleanup import (
    capture_workspace_cleanup_target,
    remove_task_workspace,
)

OWNER_ID = 7
TASK_ID = 42
WORKSPACE_ID = f"web_task_{TASK_ID}"


@pytest.fixture(autouse=True)
def _no_external_dirs(monkeypatch):
    monkeypatch.setenv("XAGENT_EXTERNAL_UPLOAD_DIRS", "")


def _make_workspace(base: Path) -> Path:
    workspace = base / WORKSPACE_ID
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "result.txt").write_text("payload")
    return workspace


def test_cleans_the_workspace_at_the_current_spelling(tmp_path, monkeypatch):
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    workspace = _make_workspace(tmp_path / "uploads" / f"user_{OWNER_ID}")

    AgentServiceManager()._cleanup_workspace_directory(TASK_ID, OWNER_ID)

    assert not workspace.exists()


def test_probing_candidates_creates_nothing(tmp_path, monkeypatch):
    """The probe cannot be the constructor.

    With nothing on disk, cleanup must leave nothing on disk: constructing a
    ``TaskWorkspace`` per candidate would create the first one's tree, report
    it as found, and delete that instead of searching on.
    """
    uploads = tmp_path / "uploads"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads))

    AgentServiceManager()._cleanup_workspace_directory(TASK_ID, OWNER_ID)

    assert not (uploads / f"user_{OWNER_ID}").exists()


def test_an_authority_mismatch_still_deletes_the_workspace(tmp_path, monkeypatch):
    """Cleanup runs off-turn, so a mismatch must not abandon the directory.

    There is no turn left to fail here -- the agent is already gone -- and the
    resolver has given an authoritative answer to delete against. Resolving
    fail-closed instead would leave the tree on disk for good, with nothing
    left to retry it.
    """
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    register_scope_resolver(
        lambda task_id: ExecutionScope(
            sandbox_key_suffix="from-resolver", workspace_segments=("from-resolver",)
        )
    )
    set_execution_scope_snapshot_loader(
        lambda task_id: ExecutionScope(
            sandbox_key_suffix="from-snapshot", workspace_segments=("from-snapshot",)
        )
    )
    workspace = _make_workspace(
        tmp_path / "uploads" / f"user_{OWNER_ID}" / "from-resolver"
    )

    AgentServiceManager()._cleanup_workspace_directory(TASK_ID, OWNER_ID)

    assert not workspace.exists()


def test_removal_is_idempotent(tmp_path, monkeypatch):
    """A second pass over an already-clean target is a success, not an error.

    Deletion removes the directory either through the cached agent or through
    a captured target, and a retry of an interrupted deletion runs over
    whatever the first attempt already finished.
    """
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    register_scope_resolver(None)
    set_execution_scope_snapshot_loader(None)
    workspace = _make_workspace(tmp_path / "uploads" / f"user_{OWNER_ID}")

    target = capture_workspace_cleanup_target(TASK_ID, OWNER_ID)

    remove_task_workspace(target)
    assert not workspace.exists()

    # The second pass finds nothing and must still be a success, not a raise:
    # the agent-owned path may already have removed the tree, and a retry of an
    # interrupted deletion runs over whatever the first attempt finished.
    remove_task_workspace(target)


def test_capturing_for_another_task_ignores_the_activated_scope(tmp_path, monkeypatch):
    """One activated scope cannot be the answer for a batch of tasks.

    A caller capturing for many tasks at once -- account deletion walks every
    task the user owns -- would otherwise look for all of them under whichever
    scope happens to be active, a base directory that was never theirs.
    """
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    register_scope_resolver(None)
    set_execution_scope_snapshot_loader(lambda task_id: None)

    activated = ExecutionScope(
        sandbox_key_suffix="someone-elses", workspace_segments=("someone-elses",)
    )
    with ExecutionScopeContext(activated):
        borrowed = capture_workspace_cleanup_target(TASK_ID, OWNER_ID)
        own = capture_workspace_cleanup_target(
            TASK_ID, OWNER_ID, prefer_active_scope=False
        )

    assert any("someone-elses" in base for base in borrowed.base_dirs)
    assert not any("someone-elses" in base for base in own.base_dirs)


def test_removal_tolerates_a_concurrent_remover(tmp_path, monkeypatch):
    """Two removers can reach the same tree, and the loser must not cry leak.

    Deleting a task cancels its background turn, and that turn's own unwind
    removes the runtime too -- on a separate worker thread, against the same
    directory. Whoever loses the race sees the tree vanish mid-walk. The tree
    being gone is the outcome either way, so reporting it as a failure would
    tell deletion a directory leaked when it had just been removed.
    """
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    register_scope_resolver(None)
    set_execution_scope_snapshot_loader(None)
    workspace = _make_workspace(tmp_path / "uploads" / f"user_{OWNER_ID}")

    real_rmtree = shutil.rmtree

    def _lost_the_race(path, *args, **kwargs):
        # Stand in for the other thread: the tree really goes, and this caller
        # meets the gap it left behind.
        real_rmtree(path)
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(workspace_module.shutil, "rmtree", _lost_the_race)

    target = capture_workspace_cleanup_target(TASK_ID, OWNER_ID)
    remove_task_workspace(target)

    assert not workspace.exists()
