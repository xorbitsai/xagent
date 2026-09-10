import logging
import re
from pathlib import Path

import pytest
import requests

from xagent.web.tools.mcp import utils

_TEST_LOGGER = logging.getLogger("test-mcp-utils")
_TEST_ALLOWED_DIRS_ENV_VAR = "TEST_MCP_UTILS_ALLOWED_DIRS"


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


# ---------------------------------------------------------------------------
# allowed_file_dirs / resolve_allowed_file_path
# ---------------------------------------------------------------------------


def test_allowed_file_dirs_strips_whitespace_around_entries(tmp_path, monkeypatch):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, f"  {dir_a} ,{dir_b}  ")

    result = utils.allowed_file_dirs(_TEST_ALLOWED_DIRS_ENV_VAR)

    assert result == [dir_a.resolve(), dir_b.resolve()]


def test_allowed_file_dirs_falls_back_to_cwd_when_unset(monkeypatch):
    monkeypatch.delenv(_TEST_ALLOWED_DIRS_ENV_VAR, raising=False)

    assert utils.allowed_file_dirs(_TEST_ALLOWED_DIRS_ENV_VAR) == [Path.cwd().resolve()]


def test_resolve_allowed_file_path_accepts_file_inside_allowed_dir(
    tmp_path, monkeypatch
):
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    target = allowed / "report.pdf"
    target.write_bytes(b"content")
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, str(allowed))

    result = utils.resolve_allowed_file_path(
        str(target), _TEST_ALLOWED_DIRS_ENV_VAR, _TEST_LOGGER
    )

    assert result == target.resolve()


def test_resolve_allowed_file_path_rejects_path_outside_allowed_dirs(
    tmp_path, monkeypatch
):
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"content")
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, str(allowed))

    with pytest.raises(PermissionError, match="allowed directories"):
        utils.resolve_allowed_file_path(
            str(outside), _TEST_ALLOWED_DIRS_ENV_VAR, _TEST_LOGGER
        )


def test_resolve_allowed_file_path_does_not_leak_host_path_or_existence(
    tmp_path, monkeypatch
):
    """Regression guard: a path outside the allowlist must report the same
    "outside allowed directories" message whether or not it actually
    exists on disk — the message must never embed the absolute host path,
    which would otherwise let the error itself be used as an oracle for
    probing the host filesystem's layout."""
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, str(allowed))
    missing_outside = tmp_path / "does_not_exist.pdf"

    with pytest.raises(PermissionError) as exc_info:
        utils.resolve_allowed_file_path(
            str(missing_outside), _TEST_ALLOWED_DIRS_ENV_VAR, _TEST_LOGGER
        )

    assert "allowed directories" in str(exc_info.value)
    assert str(tmp_path) not in str(exc_info.value)


def test_resolve_allowed_file_path_rejects_dot_dot_traversal(tmp_path, monkeypatch):
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    (tmp_path / "secret.txt").write_text("secret")
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, str(allowed))

    with pytest.raises(PermissionError, match="allowed directories"):
        utils.resolve_allowed_file_path(
            str(allowed / ".." / "secret.txt"),
            _TEST_ALLOWED_DIRS_ENV_VAR,
            _TEST_LOGGER,
        )


def test_resolve_allowed_file_path_rejects_prefix_confusable_sibling_dir(
    tmp_path, monkeypatch
):
    """Regression guard: an allowed dir "ws" must not accidentally admit a
    sibling "ws_evil" just because it starts with the same string —
    containment has to be a real path-relative check (is_relative_to), not
    a naive string prefix comparison."""
    allowed = tmp_path / "ws"
    allowed.mkdir()
    evil = tmp_path / "ws_evil"
    evil.mkdir()
    outside = evil / "secret.txt"
    outside.write_text("secret")
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, str(allowed))

    with pytest.raises(PermissionError, match="allowed directories"):
        utils.resolve_allowed_file_path(
            str(outside), _TEST_ALLOWED_DIRS_ENV_VAR, _TEST_LOGGER
        )


def test_resolve_allowed_file_path_rejects_symlink_escaping_allowed_dir(
    tmp_path, monkeypatch
):
    """A symlink physically located inside the allowed directory but
    pointing outside it must not grant access to its target — resolve()
    follows the symlink to its real location before the containment check
    runs, so the escape is caught rather than trusted because the link
    itself lives in an allowed place."""
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("secret")
    link = allowed / "escape_link.txt"
    link.symlink_to(secret_file)
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, str(allowed))

    with pytest.raises(PermissionError, match="allowed directories"):
        utils.resolve_allowed_file_path(
            str(link), _TEST_ALLOWED_DIRS_ENV_VAR, _TEST_LOGGER
        )


def test_resolve_allowed_file_path_rejects_relative_path_resolved_against_cwd(
    tmp_path, monkeypatch
):
    """A relative file_path resolves against this process's own cwd, not
    the allowed directory — so it must be rejected when cwd isn't itself
    inside the allowlist, exactly as the docstring warns."""
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    other_cwd = tmp_path / "somewhere_else"
    other_cwd.mkdir()
    (other_cwd / "report.pdf").write_bytes(b"content")
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, str(allowed))
    monkeypatch.chdir(other_cwd)

    with pytest.raises(PermissionError, match="allowed directories"):
        utils.resolve_allowed_file_path(
            "report.pdf", _TEST_ALLOWED_DIRS_ENV_VAR, _TEST_LOGGER
        )


def test_resolve_allowed_file_path_rejects_missing_file_inside_allowed_dir(
    tmp_path, monkeypatch
):
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, str(allowed))

    with pytest.raises(FileNotFoundError, match="report.pdf"):
        utils.resolve_allowed_file_path(
            str(allowed / "report.pdf"), _TEST_ALLOWED_DIRS_ENV_VAR, _TEST_LOGGER
        )


def test_resolve_allowed_file_path_uses_custom_not_found_label(tmp_path, monkeypatch):
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    monkeypatch.setenv(_TEST_ALLOWED_DIRS_ENV_VAR, str(allowed))

    with pytest.raises(FileNotFoundError, match="Attachment not found"):
        utils.resolve_allowed_file_path(
            str(allowed / "report.pdf"),
            _TEST_ALLOWED_DIRS_ENV_VAR,
            _TEST_LOGGER,
            not_found_label="Attachment",
        )
