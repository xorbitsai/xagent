"""Deleting a task's workspace finds it whichever spelling wrote it.

``_cleanup_workspace_directory`` runs when no agent is in memory, so it has
to locate the workspace from configuration alone, and ``TaskWorkspace``'s
constructor creates the tree it is pointed at -- so the candidate probe has to
be side-effect free, or the first candidate always "exists" and the real
workspace is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xagent.core.execution_scope import set_execution_scope_resolver
from xagent.web.api.chat import AgentServiceManager

OWNER_ID = 7
TASK_ID = 42
WORKSPACE_ID = f"web_task_{TASK_ID}"


@pytest.fixture(autouse=True)
def _no_resolver():
    set_execution_scope_resolver(None)
    yield
    set_execution_scope_resolver(None)


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
