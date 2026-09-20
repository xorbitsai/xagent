import json
from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module_name, helper_name, env_var",
    [
        ("gmail", "_allowed_file_dirs", "XAGENT_GMAIL_FILE_ALLOWED_DIRS"),
        ("slack", "_allowed_file_dirs", "XAGENT_SLACK_FILE_ALLOWED_DIRS"),
        ("linkedin", "_allowed_image_dirs", "XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS"),
    ],
)
def test_connectors_delegate_configured_upload_roots_to_shared_parser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    helper_name: str,
    env_var: str,
) -> None:
    module = import_module(f"xagent.web.tools.mcp.{module_name}")
    configured_root = tmp_path / "reports,final"
    monkeypatch.setenv(env_var, json.dumps([str(configured_root)]))

    assert getattr(module, helper_name)() == [configured_root.resolve()]


@pytest.mark.parametrize(
    "module_name, helper_name, env_var",
    [
        ("gmail", "_allowed_file_dirs", "XAGENT_GMAIL_FILE_ALLOWED_DIRS"),
        ("slack", "_allowed_file_dirs", "XAGENT_SLACK_FILE_ALLOWED_DIRS"),
        ("linkedin", "_allowed_image_dirs", "XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS"),
    ],
)
def test_connectors_scrub_invalid_upload_root_configuration(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    module_name: str,
    helper_name: str,
    env_var: str,
) -> None:
    module = import_module(f"xagent.web.tools.mcp.{module_name}")
    monkeypatch.setenv(env_var, "[")

    with pytest.raises(ValueError, match="^Upload directory configuration is invalid$"):
        getattr(module, helper_name)()

    assert env_var in caplog.text
