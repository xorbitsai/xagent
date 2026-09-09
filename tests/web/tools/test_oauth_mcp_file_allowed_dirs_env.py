"""Tests for injecting the file-upload allowlist directory into OAuth-transport
MCP subprocess environments (LinkedIn's image upload, Slack's file upload,
Gmail's message attachments) and Google Drive's dedicated write-target
output directory."""

from types import SimpleNamespace

from xagent.web.tools.config import WebToolConfig


def _app_info(module: str, access_token_env: str) -> dict:
    return {
        "launch_config": {
            "command": "python",
            "args": ["-m", f"xagent.web.tools.mcp.{module}"],
            "env_mapping": {access_token_env: "access_token"},
        }
    }


def test_transport_config_sets_both_allowlist_vars_when_workspace_has_a_task(
    tmp_path,
):
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
    assert transport_config["env"]["XAGENT_SLACK_FILE_ALLOWED_DIRS"] == expected_dir
    assert transport_config["env"]["XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS"] == expected_dir
    assert transport_config["env"]["XAGENT_GMAIL_FILE_ALLOWED_DIRS"] == expected_dir
    assert transport_config["env"]["XAGENT_GOOGLE_DRIVE_OUTPUT_DIR"] == expected_dir


def test_transport_config_omits_allowlist_vars_without_a_task_id():
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
    assert "XAGENT_GOOGLE_DRIVE_OUTPUT_DIR" not in transport_config["env"]


def test_drive_output_dir_excludes_external_dirs_unlike_the_read_allowlists(
    tmp_path,
):
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
    assert str(external_dir.resolve()) in transport_config["env"][
        "XAGENT_SLACK_FILE_ALLOWED_DIRS"
    ].split(",")


def test_drive_output_dir_omitted_when_only_external_dirs_are_configured(
    tmp_path,
):
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
    assert str(external_dir.resolve()) in transport_config["env"][
        "XAGENT_SLACK_FILE_ALLOWED_DIRS"
    ].split(",")
