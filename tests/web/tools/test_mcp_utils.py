import json
import os
import re
from pathlib import Path

import pytest
import requests

from xagent.web.tools.mcp import utils


def test_require_clean_identifier_rejects_empty_and_whitespace():
    with pytest.raises(ValueError, match="record_id"):
        utils.require_clean_identifier("", "record_id")
    with pytest.raises(ValueError, match="record_id"):
        utils.require_clean_identifier(" 001xx ", "record_id")
    assert utils.require_clean_identifier("001xx", "record_id") == "001xx"


def test_require_clean_identifier_rejects_non_string():
    """A truthy non-str (e.g. an int) previously slipped past `not value`
    and crashed on `.strip()` with a raw AttributeError instead of a clean
    ValueError."""
    with pytest.raises(ValueError, match="record_id"):
        utils.require_clean_identifier(12345, "record_id")


def test_url_path_id_percent_encodes_reserved_characters():
    # A literal ".." blocklist misses "/" and "?", which redirect the
    # request to a different endpoint or inject query params without ever
    # containing "..". Percent-encoding closes off all of them at once.
    assert utils.url_path_id("Account/001abc", "sobject_type") == ("Account%2F001abc")
    assert utils.url_path_id("001x?fields=Id", "record_id") == ("001x%3Ffields%3DId")
    with pytest.raises(ValueError):
        utils.url_path_id("", "record_id")


def test_url_path_id_rejects_exact_dot_segments():
    # "." and ".." are always-unreserved characters -- quote() never
    # touches them, and requests/urllib3 collapse dot-segments out of the
    # final URL before sending it, so percent-encoding alone can't close
    # this off the way it does for "/" and "?".
    with pytest.raises(ValueError, match="record_id"):
        utils.url_path_id("..", "record_id")
    with pytest.raises(ValueError, match="record_id"):
        utils.url_path_id(".", "record_id")


@pytest.mark.parametrize(
    "limit,expected",
    [
        (50, 50),  # within range, passed through unchanged
        (1, 1),  # lower boundary, passed through unchanged
        (200, 200),  # exactly max_limit, passed through unchanged
        (201, 200),  # just above max_limit, clamped down
        (10**9, 200),  # extreme, clamped down the same as a mild overage
        (0, 1),  # zero would slice to an empty page forever -- clamped up
        (-1, 1),  # mild negative, clamped up
        (-(10**9), 1),  # extreme negative, clamped up the same as mild
    ],
)
def test_clamp_limit_boundaries(limit, expected):
    assert utils.clamp_limit(limit, max_limit=200) == expected


@pytest.mark.parametrize(
    "offset,expected",
    [
        (0, 0),
        (5, 5),
        (-1, 0),  # mild negative -- would slice from the end unclamped
        (-(10**9), 0),  # extreme negative, clamped the same as mild
    ],
)
def test_clamp_offset_boundaries(offset, expected):
    assert utils.clamp_offset(offset) == expected


_TEST_URL_ID_PATTERN = re.compile(r"/d/([a-zA-Z0-9_-]+)")


def test_resolve_id_from_url_extracts_id_from_matching_url():
    assert (
        utils.resolve_id_from_url(
            "https://docs.google.com/document/d/abc123/edit",
            _TEST_URL_ID_PATTERN,
            "document_id",
        )
        == "abc123"
    )


def test_resolve_id_from_url_passes_through_bare_id():
    assert (
        utils.resolve_id_from_url(" abc123 ", _TEST_URL_ID_PATTERN, "document_id")
        == "abc123"
    )


def test_resolve_id_from_url_rejects_non_string():
    """Mirrors require_clean_identifier's fix: pattern.search() would
    otherwise raise a raw TypeError on a non-string value instead of a
    clean ValueError."""
    with pytest.raises(ValueError, match="document_id"):
        utils.resolve_id_from_url(12345, _TEST_URL_ID_PATTERN, "document_id")


def test_url_path_id_output_survives_requests_url_normalization():
    """Confirms the actual exploit this guards against: a naively
    interpolated ".." collapses the path via requests' own URL
    normalization to a completely different (still valid) endpoint."""
    prepared = requests.Request(
        "GET", "https://acme.my.salesforce.com/services/data/v59.0/sobjects/Account/.."
    ).prepare()
    assert (
        prepared.url == "https://acme.my.salesforce.com/services/data/v59.0/sobjects/"
    )

    with pytest.raises(ValueError):
        utils.url_path_id("..", "record_id")


_TEST_ALLOWED_DIRS_ENV_VAR = "XAGENT_TEST_FILE_ALLOWED_DIRS"


def test_allowed_dirs_from_env_falls_back_to_cwd_when_unset(monkeypatch):
    monkeypatch.delenv(_TEST_ALLOWED_DIRS_ENV_VAR, raising=False)
    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        Path.cwd().resolve()
    ]


def test_allowed_dirs_from_env_falls_back_to_cwd_when_blank(monkeypatch):
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, "   ")
    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        Path.cwd().resolve()
    ]


@pytest.mark.parametrize("raw_value", [",", " , ", ",,,"])
def test_allowed_dirs_from_env_falls_back_to_cwd_when_malformed(monkeypatch, raw_value):
    """Regression guard for the real bug this consolidation was written to
    fix: a lone comma (or several) parses to zero real entries, and
    without this fallback that becomes an empty allowlist that silently
    rejects every upload with nothing pointing at the env var as the
    cause -- previously true for onedrive.py/gmail.py/slack.py/linkedin.py,
    but not google_drive.py, whose copy already guarded against it."""
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, raw_value)
    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        Path.cwd().resolve()
    ]


def test_allowed_dirs_from_env_parses_multiple_dirs_with_whitespace(
    monkeypatch, tmp_path
):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, f" {dir_a} , {dir_b} ")

    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        dir_a.resolve(),
        dir_b.resolve(),
    ]


def test_allowed_dirs_from_env_parses_json_paths_containing_commas(
    monkeypatch, tmp_path
):
    directory = tmp_path / "reports,final"
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, json.dumps([str(directory)]))

    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        directory.resolve()
    ]


@pytest.mark.parametrize("raw_value", ["[]", '[""]', '["   "]'])
def test_allowed_dirs_from_env_treats_empty_json_paths_as_deny_all(
    monkeypatch, raw_value
):
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, raw_value)

    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == []


@pytest.mark.parametrize("raw_value", ["[", '["ok", 42]'])
def test_allowed_dirs_from_env_rejects_invalid_json(monkeypatch, raw_value):
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, raw_value)

    with pytest.raises(ValueError, match="JSON array"):
        utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges")
def test_allowed_dirs_from_env_rejects_unresolvable_path(monkeypatch, tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, json.dumps([str(loop)]))

    with pytest.raises(ValueError, match="invalid path") as exc_info:
        utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR)
    assert str(loop) not in str(exc_info.value)


def test_allowed_dirs_from_env_expands_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, "~/workspace")

    assert utils.allowed_dirs_from_env(_TEST_ALLOWED_DIRS_ENV_VAR) == [
        (tmp_path / "workspace").resolve()
    ]
