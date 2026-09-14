"""Tests for injecting the file-upload allowlist directory into OAuth-transport
MCP subprocess environments (LinkedIn's image upload, Slack's file upload,
Gmail's message attachments, OneDrive's file upload, Google Drive's file
upload) and Google Drive's dedicated write-target output directory."""

import json
from pathlib import Path
from types import SimpleNamespace

from xagent.web.tools.config import WebToolConfig

_READ_ALLOWLIST_ENV_VARS = (
    "XAGENT_SLACK_FILE_ALLOWED_DIRS",
    "XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS",
    "XAGENT_GMAIL_FILE_ALLOWED_DIRS",
    "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS",
    "XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS",
)


def _app_info(module: str, access_token_env: str) -> dict:
    return {
        "launch_config": {
            "command": "python",
            "args": ["-m", f"xagent.web.tools.mcp.{module}"],
            "env_mapping": {access_token_env: "access_token"},
        }
    }


def test_transport_config_sets_all_allowlist_vars_when_workspace_has_a_task(
    tmp_path: Path,
) -> None:
    cfg = WebToolConfig(
        db=None,
        request=None,
        task_id="task-123",
        workspace_base_dir=str(tmp_path),
    )

    transport_config = cfg._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Slack"),
        app_info=_app_info("slack", "SLACK_ACCESS_TOKEN"),
        access_token="user-access-token",
    )

    expected_dir = str((tmp_path / "task-123").resolve())
    for env_var in _READ_ALLOWLIST_ENV_VARS:
        assert json.loads(transport_config["env"][env_var]) == [expected_dir]
    assert transport_config["env"]["XAGENT_GOOGLE_DRIVE_OUTPUT_DIR"] == expected_dir


def test_transport_config_omits_allowlist_vars_without_a_task_id() -> None:
    """Regression guard for the branch that actually runs in production
    unpatched: with no task_id, _build_mcp_file_allowed_dirs() returns an
    empty string and neither allowlist var should be set at all — this is
    the fallback-to-cwd path the allowlist is meant to close off."""
    cfg = WebToolConfig(db=None, request=None)

    transport_config = cfg._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Slack"),
        app_info=_app_info("slack", "SLACK_ACCESS_TOKEN"),
        access_token="user-access-token",
    )

    assert "XAGENT_SLACK_FILE_ALLOWED_DIRS" not in transport_config["env"]
    assert "XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS" not in transport_config["env"]
    assert "XAGENT_GMAIL_FILE_ALLOWED_DIRS" not in transport_config["env"]
    assert "XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS" not in transport_config["env"]
    assert "XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS" not in transport_config["env"]
    assert "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR" not in transport_config["env"]


def test_drive_output_dir_excludes_external_dirs_unlike_the_read_allowlists(
    tmp_path: Path,
) -> None:
    """XAGENT_GOOGLE_DRIVE_OUTPUT_DIR must never pick up
    allowed_external_dirs the way the read allowlists do — those can be a
    read-only KB folder, and picking one as a *write* target would be
    wrong, not just a different (but still safe) choice."""
    external_dir = tmp_path / "kb"
    external_dir.mkdir()
    cfg = WebToolConfig(
        db=None,
        request=None,
        task_id="task-123",
        workspace_base_dir=str(tmp_path),
    )
    cfg._workspace_config["allowed_external_dirs"] = [str(external_dir)]

    transport_config = cfg._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Slack"),
        app_info=_app_info("slack", "SLACK_ACCESS_TOKEN"),
        access_token="user-access-token",
    )

    task_dir = str((tmp_path / "task-123").resolve())
    assert transport_config["env"]["XAGENT_GOOGLE_DRIVE_OUTPUT_DIR"] == task_dir
    # The read allowlists, by contrast, legitimately include the external
    # dir alongside the task dir.
    for env_var in _READ_ALLOWLIST_ENV_VARS:
        assert str(external_dir.resolve()) in json.loads(
            transport_config["env"][env_var]
        )


def test_drive_output_dir_omitted_when_only_external_dirs_are_configured(
    tmp_path: Path,
) -> None:
    """No task_id at all (only allowed_external_dirs) must leave
    XAGENT_GOOGLE_DRIVE_OUTPUT_DIR unset entirely rather than falling back
    to one of those external dirs as a write target."""
    external_dir = tmp_path / "kb"
    external_dir.mkdir()
    cfg = WebToolConfig(db=None, request=None)
    cfg._workspace_config["allowed_external_dirs"] = [str(external_dir)]

    transport_config = cfg._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="Slack"),
        app_info=_app_info("slack", "SLACK_ACCESS_TOKEN"),
        access_token="user-access-token",
    )

    assert "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR" not in transport_config["env"]
    for env_var in _READ_ALLOWLIST_ENV_VARS:
        assert str(external_dir.resolve()) in json.loads(
            transport_config["env"][env_var]
        )


def test_read_allowlists_preserve_comma_in_directory_name(tmp_path: Path) -> None:
    workspace_base = tmp_path / "workspaces,active"
    external_dir = tmp_path / "knowledge,base"
    cfg = WebToolConfig(
        db=None,
        request=None,
        task_id="task-123",
        workspace_base_dir=str(workspace_base),
    )
    cfg._workspace_config["allowed_external_dirs"] = [str(external_dir)]

    transport_config = cfg._build_oauth_mcp_stdio_transport_config(
        server=SimpleNamespace(name="OneDrive"),
        app_info=_app_info("onedrive", "AUTH_TOKEN"),
        access_token="user-access-token",
    )

    expected_dirs = [
        str((workspace_base / "task-123").resolve()),
        str(external_dir.resolve()),
    ]
    for env_var in _READ_ALLOWLIST_ENV_VARS:
        assert json.loads(transport_config["env"][env_var]) == expected_dirs
