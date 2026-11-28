"""Task workspace directory initialization on an ordinary filesystem."""

from pathlib import Path

import pytest

from xagent.core.workspace import TaskWorkspace


@pytest.mark.parametrize("managed", [True, False], ids=["managed", "external"])
def test_task_workspace_initializes_regular_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, managed: bool
) -> None:
    uploads = tmp_path / "managed"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads))
    base_dir = uploads / "actor" if managed else tmp_path / "external"

    workspace = TaskWorkspace(id="task-123", base_dir=str(base_dir))

    assert workspace.workspace_dir == base_dir / "task-123"
    assert workspace.input_dir.is_dir()
    assert workspace.output_dir.is_dir()
    assert workspace.temp_dir.is_dir()
