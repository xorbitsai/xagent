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
        (
            "google_drive",
            "_upload_allowed_dirs",
            "XAGENT_GOOGLE_DRIVE_FILE_ALLOWED_DIRS",
        ),
    ],
)
@pytest.mark.parametrize(
    "raw_value",
    [None, "", " , ", "legacy_roots", "json_roots", "json_empty", "invalid_json"],
)
def test_connectors_parse_configured_upload_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    helper_name: str,
    env_var: str,
    raw_value: str | None,
) -> None:
    module = import_module(f"xagent.web.tools.mcp.{module_name}")
    monkeypatch.chdir(tmp_path)
    if raw_value is None:
        monkeypatch.delenv(env_var, raising=False)
    elif raw_value == "legacy_roots":
        monkeypatch.setenv(env_var, f" {tmp_path / 'first'} , {tmp_path / 'second'} ")
    elif raw_value == "json_roots":
        monkeypatch.setenv(
            env_var,
            json.dumps([str(tmp_path / "first,part"), str(tmp_path / "second")]),
        )
    elif raw_value == "json_empty":
        monkeypatch.setenv(env_var, "[]")
    elif raw_value == "invalid_json":
        monkeypatch.setenv(env_var, "[")
    else:
        monkeypatch.setenv(env_var, raw_value)

    if raw_value == "invalid_json":
        with pytest.raises(ValueError, match="JSON array"):
            getattr(module, helper_name)()
        return

    if raw_value == "legacy_roots":
        expected = [(tmp_path / "first").resolve(), (tmp_path / "second").resolve()]
    elif raw_value == "json_roots":
        expected = [
            (tmp_path / "first,part").resolve(),
            (tmp_path / "second").resolve(),
        ]
    elif raw_value == "json_empty":
        expected = []
    else:
        expected = [Path.cwd().resolve()]
    assert getattr(module, helper_name)() == expected
