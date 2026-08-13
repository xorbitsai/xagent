"""Tests for injecting the file-upload allowlist directory into OAuth-transport
MCP subprocess environments (LinkedIn's image upload, Slack's file upload)."""

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
