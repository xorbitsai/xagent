"""LinkedIn create_post accepts a registered file reference for image_path.

A post approved in one run can be published by a later run of the same task,
after the proposing run's workspace no longer holds the image
(xorbitsai/xagent-saas#3379). ``image_path`` therefore accepts ``file:<id>``,
and the host stages that registered file into the current task workspace
before the connector runs.
"""

import json
from pathlib import Path

import pytest

from xagent.core.tools.adapters.vibe.mcp_adapter import (
    _build_mcp_tool_adapter,
    _durable_upload_fields,
)
from xagent.core.workspace import TaskWorkspace
from xagent.web.tools.config import WebToolConfig
from xagent.web.tools.mcp import linkedin

_ALLOWED_DIRS_ENV_VAR = "XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS"
_FILE_ID = "31218a9f-1497-4216-9f75-1fc8d51368c4"


class _StagingWorkspace:
    def __init__(self, staged_path: Path) -> None:
        self.staged_path = staged_path
        self.staged_ids: list[str] = []

    def stage_file_for_external_upload(self, file_id: str) -> Path:
        self.staged_ids.append(file_id)
        return self.staged_path


async def _linkedin_create_post_adapter(workspace: _StagingWorkspace):
    tools = await linkedin.list_tools()
    create_post = next(tool for tool in tools if tool.name == "create_post")
    # Staging is keyed by argument name only; a renamed connector argument
    # would leave the mapping matching nothing.
    staged_fields = _durable_upload_fields("LinkedIn", "create_post")
    assert set(staged_fields) <= set(create_post.inputSchema["properties"])
    return _build_mcp_tool_adapter(
        "LinkedIn",
        {"transport": "stdio", "command": "python", "args": []},
        create_post,
        workspace=workspace,
    )


@pytest.mark.asyncio
async def test_the_linkedin_adapter_stages_a_file_ref_into_image_path():
    staged_path = Path("/task/temp/.xagent-internal/mcp-upload/abc/poster.png")
    workspace = _StagingWorkspace(staged_path)
    adapter = await _linkedin_create_post_adapter(workspace)

    assert "file_id" in adapter.description
    prepared, staged, sources = await adapter._stage_external_upload_args(
        {"text": "Launch day", "image_path": f"file:{_FILE_ID}"}
    )

    assert workspace.staged_ids == [_FILE_ID]
    assert prepared == {"text": "Launch day", "image_path": str(staged_path)}
    assert staged == [staged_path]
    assert sources == {str(staged_path): f"file:{_FILE_ID}"}


@pytest.mark.asyncio
async def test_the_linkedin_adapter_passes_a_local_image_path_through():
    """Guard: a same-run post still names the image by its workspace path."""
    workspace = _StagingWorkspace(Path("/unused"))
    adapter = await _linkedin_create_post_adapter(workspace)
    arguments = {"text": "Launch day", "image_path": "/task/output/poster.png"}

    prepared, staged, sources = await adapter._stage_external_upload_args(arguments)

    assert prepared == arguments
    assert staged == []
    assert sources == {}
    assert workspace.staged_ids == []


def test_a_staged_file_passes_the_linkedin_image_allowlist(tmp_path, monkeypatch):
    """Premise check: a staged path sits under the task allowlist that
    WebToolConfig builds for local MCP file tools, so the connector accepts
    the file it is handed even after the original is gone."""
    workspace = TaskWorkspace(id="web_task_9999", base_dir=str(tmp_path))
    source = workspace.output_dir / "poster.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\n approved poster")
    monkeypatch.setattr(workspace, "resolve_file_id_detached", lambda file_id: source)
    staged = workspace.stage_file_for_external_upload(_FILE_ID)
    source.unlink()

    config = WebToolConfig(
        db=None,
        request=None,
        task_id="web_task_9999",
        workspace_base_dir=str(tmp_path),
    )
    monkeypatch.setenv(
        _ALLOWED_DIRS_ENV_VAR, json.dumps(config._mcp_file_allowed_dir_paths())
    )

    resolved = linkedin._resolve_allowed_image_path(str(staged))

    assert resolved == Path(staged).resolve()
    assert resolved.read_bytes() == b"\x89PNG\r\n\x1a\n approved poster"
