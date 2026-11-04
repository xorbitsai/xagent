"""Tests for jira_add_attachment and its local-file guard."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from xagent.core.tools.adapters.vibe.mcp_adapter import _build_mcp_tool_adapter
from xagent.core.workspace import TaskWorkspace
from xagent.web.tools.config import WebToolConfig
from xagent.web.tools.mcp import jira, jira_attachment

_ALLOWED_DIRS_ENV_VAR = "XAGENT_JIRA_FILE_ALLOWED_DIRS"
_ATTACHMENTS_URL = (
    "https://api.atlassian.com/ex/jira/site-a/rest/api/2/issue/ENG-1/attachments"
)


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        text: str = "",
        headers: dict | None = None,
        url: str = _ATTACHMENTS_URL,
    ):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.text = text or (
            json.dumps(self._json_data) if json_data is not None else ""
        )
        self.content = self.text.encode()
        self.headers = headers or {}
        self.url = url

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error for url: {self.url}",
                response=self,
            )


def _created(size: int = 10, /, **overrides):
    entry = {
        "id": "10001",
        "filename": "report.txt",
        "size": size,
        "mimeType": "text/plain",
        "content": "https://acme.atlassian.net/secure/attachment/10001/report.txt",
    }
    entry.update(overrides)
    return [entry]


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("JIRA_ACCESS_TOKEN", "access-token")


@pytest.fixture
def allowed_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "workspace"
    directory.mkdir()
    monkeypatch.setenv(_ALLOWED_DIRS_ENV_VAR, json.dumps([str(directory)]))
    return directory


def _write(directory: Path, name: str = "report.txt", data: bytes = b"hello jira"):
    path = directory / name
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# read_allowed_file: the local-file guard
# ---------------------------------------------------------------------------


def test_read_allowed_file_returns_name_and_bytes(allowed_dir):
    path = _write(allowed_dir, data=b"abc")

    assert jira_attachment.read_allowed_file(str(path)) == ("report.txt", b"abc")


def test_read_allowed_file_follows_a_symlink_inside_the_allowlist(allowed_dir):
    """The tool docstring says the file is stored under its own name and a
    symlink in file_path is followed; the result's `filename` is what the
    agent must use to refer to it."""
    target = _write(allowed_dir, name="real.txt", data=b"real bytes")
    (allowed_dir / "alias.txt").symlink_to(target)

    assert jira_attachment.read_allowed_file(str(allowed_dir / "alias.txt")) == (
        "real.txt",
        b"real bytes",
    )


def test_read_allowed_file_resolves_relative_path_against_cwd(allowed_dir, monkeypatch):
    _write(allowed_dir, data=b"rel")
    monkeypatch.chdir(allowed_dir)

    assert jira_attachment.read_allowed_file("report.txt") == (
        "report.txt",
        b"rel",
    )


def test_read_allowed_file_rejects_path_outside_allowed_dirs(
    allowed_dir, tmp_path, caplog
):
    outside = _write(tmp_path, name="secret.txt")

    with caplog.at_level(logging.WARNING, logger="jira-mcp"):
        with pytest.raises(PermissionError) as excinfo:
            jira_attachment.read_allowed_file(str(outside))

    # The host path stays out of the message that reaches the model, but the
    # operator still gets it in the server log.
    assert str(tmp_path) not in str(excinfo.value)
    assert str(outside) in caplog.text


def test_read_allowed_file_hides_whether_an_outside_path_exists(allowed_dir, tmp_path):
    """A nonexistent path outside the allowlist must fail exactly like an
    existing one: reporting "not found" for one and "outside" for the other
    would let a caller probe the host filesystem."""
    existing = _write(tmp_path, name="exists.txt")
    missing = tmp_path / "missing.txt"

    with pytest.raises(PermissionError) as existing_exc:
        jira_attachment.read_allowed_file(str(existing))
    with pytest.raises(PermissionError) as missing_exc:
        jira_attachment.read_allowed_file(str(missing))

    assert str(existing_exc.value) == str(missing_exc.value)


def test_read_allowed_file_rejects_symlink_escaping_the_allowlist(
    allowed_dir, tmp_path
):
    outside = _write(tmp_path, name="secret.txt")
    link = allowed_dir / "link.txt"
    link.symlink_to(outside)

    with pytest.raises(PermissionError):
        jira_attachment.read_allowed_file(str(link))


def test_read_allowed_file_reports_missing_file_inside_allowed_dir(allowed_dir):
    with pytest.raises(FileNotFoundError):
        jira_attachment.read_allowed_file(str(allowed_dir / "nope.txt"))


def test_read_allowed_file_rejects_directory(allowed_dir):
    subdir = allowed_dir / "sub"
    subdir.mkdir()

    with pytest.raises(ValueError, match="not a regular file"):
        jira_attachment.read_allowed_file(str(subdir))


def test_read_allowed_file_rejects_empty_file(allowed_dir):
    path = _write(allowed_dir, data=b"")

    with pytest.raises(ValueError, match="empty"):
        jira_attachment.read_allowed_file(str(path))


def test_read_allowed_file_rejects_oversized_file(allowed_dir, monkeypatch):
    path = _write(allowed_dir, data=b"x" * 11)
    monkeypatch.setattr(jira_attachment, "MAX_ATTACHMENT_BYTES", 10)

    with pytest.raises(ValueError, match="limit"):
        jira_attachment.read_allowed_file(str(path))


def test_read_allowed_file_accepts_file_exactly_at_the_limit(allowed_dir, monkeypatch):
    path = _write(allowed_dir, data=b"x" * 10)
    monkeypatch.setattr(jira_attachment, "MAX_ATTACHMENT_BYTES", 10)

    assert jira_attachment.read_allowed_file(str(path))[1] == b"x" * 10


@pytest.mark.parametrize("blank", ["", "   "])
def test_read_allowed_file_rejects_blank_path(allowed_dir, blank):
    with pytest.raises(ValueError, match="file_path"):
        jira_attachment.read_allowed_file(blank)


def test_read_allowed_file_rejects_path_with_nul_byte(allowed_dir):
    with pytest.raises(ValueError, match="file_path"):
        jira_attachment.read_allowed_file(str(allowed_dir / "a\0b.txt"))


def test_read_allowed_file_denies_everything_for_an_empty_allowlist(
    tmp_path, monkeypatch
):
    path = _write(tmp_path)
    monkeypatch.setenv(_ALLOWED_DIRS_ENV_VAR, "[]")

    with pytest.raises(PermissionError):
        jira_attachment.read_allowed_file(str(path))


def test_read_allowed_file_rejects_symlink_loop(allowed_dir):
    loop = allowed_dir / "loop"
    loop.symlink_to(loop)

    with pytest.raises(ValueError, match="Could not resolve file_path"):
        jira_attachment.read_allowed_file(str(loop))


def test_read_allowed_file_keeps_the_host_path_out_of_a_read_failure(
    allowed_dir, monkeypatch, caplog
):
    path = _write(allowed_dir)
    real_open = Path.open

    def failing_open(self, *args, **kwargs):
        if self == path:
            raise PermissionError(13, "Permission denied", str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with caplog.at_level(logging.WARNING, logger="jira-mcp"):
        with pytest.raises(ValueError, match="Could not read the file") as excinfo:
            jira_attachment.read_allowed_file(str(path))

    assert str(allowed_dir) not in str(excinfo.value)
    assert str(path) in caplog.text


@pytest.mark.parametrize("method", ["exists", "is_file"])
def test_read_allowed_file_keeps_the_host_path_out_of_a_stat_failure(
    allowed_dir, monkeypatch, caplog, method
):
    """exists()/is_file() raise PermissionError carrying the host path (EACCES
    or ESTALE on stat); it must not reach the model."""
    path = _write(allowed_dir)
    real = getattr(Path, method)

    def failing(self):
        if self == path:
            raise PermissionError(13, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, method, failing)

    with caplog.at_level(logging.WARNING, logger="jira-mcp"):
        with pytest.raises(ValueError, match="Could not read the file") as excinfo:
            jira_attachment.read_allowed_file(str(path))

    assert str(allowed_dir) not in str(excinfo.value)
    assert str(path) in caplog.text


def test_read_allowed_file_rejects_a_file_that_shrinks_while_being_read(
    allowed_dir, monkeypatch
):
    path = _write(allowed_dir, data=b"hello jira")
    # fstat reports 100 bytes, as if the file had been truncated since.
    monkeypatch.setattr(
        jira_attachment,
        "os",
        SimpleNamespace(fstat=lambda fd: SimpleNamespace(st_size=100)),
    )

    with pytest.raises(ValueError, match="changed while"):
        jira_attachment.read_allowed_file(str(path))


# ---------------------------------------------------------------------------
# jira_add_attachment: request shape
# ---------------------------------------------------------------------------


def test_add_attachment_posts_multipart_with_no_check_header(allowed_dir, monkeypatch):
    path = _write(allowed_dir, data=b"hello jira")
    mock_request = Mock(return_value=MockResponse(json_data=_created(10)))
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_add_attachment("ENG-1", str(path), cloud_id="site-a"))

    assert result == {
        "status": "success",
        "attachment": {
            "id": "10001",
            "filename": "report.txt",
            "size": 10,
            "mime_type": "text/plain",
        },
    }
    mock_request.assert_called_once()
    call = mock_request.call_args.kwargs
    assert call["method"] == "POST"
    assert call["url"] == _ATTACHMENTS_URL
    # Bytes, not a file handle: a handle would be left at EOF for the 429
    # retry and re-send an empty file.
    assert call["files"] == {"file": ("report.txt", b"hello jira")}
    assert call["json"] is None
    # requests' timeout bounds the whole body send, so the 30s default would
    # need ~7 Mbit/s for a 25 MiB file.
    assert call["timeout"] == jira_attachment.UPLOAD_TIMEOUT_SECONDS
    assert call["timeout"] > jira.DEFAULT_TIMEOUT_SECONDS
    headers = call["headers"]
    assert headers["X-Atlassian-Token"] == "no-check"
    assert headers["Authorization"] == "Bearer access-token"
    # An explicit JSON Content-Type would replace the multipart boundary.
    assert "Content-Type" not in {name.title() for name in headers}


def test_add_attachment_percent_encodes_issue_key(allowed_dir, monkeypatch):
    path = _write(allowed_dir)
    mock_request = Mock(return_value=MockResponse(json_data=_created(10)))
    monkeypatch.setattr(jira.requests, "request", mock_request)

    jira.jira_add_attachment("ENG-1/../x", str(path), cloud_id="site-a")

    assert mock_request.call_args.kwargs["url"].endswith(
        "/rest/api/2/issue/ENG-1%2F..%2Fx/attachments"
    )


def test_add_attachment_resends_the_full_payload_on_429_retry(allowed_dir, monkeypatch):
    path = _write(allowed_dir, data=b"hello jira")
    monkeypatch.setattr(jira.time, "sleep", lambda s: None)
    mock_request = Mock(
        side_effect=[
            MockResponse(status_code=429, headers={"Retry-After": "1"}),
            MockResponse(json_data=_created(10)),
        ]
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    result = json.loads(jira.jira_add_attachment("ENG-1", str(path), cloud_id="site-a"))

    assert result["status"] == "success"
    assert mock_request.call_count == 2
    for call in mock_request.call_args_list:
        assert call.kwargs["files"] == {"file": ("report.txt", b"hello jira")}
        assert "Content-Type" not in {name.title() for name in call.kwargs["headers"]}


def test_request_absolute_keeps_json_content_type_for_non_multipart(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(jira.requests, "request", mock_request)

    jira._request_absolute("POST", "https://api.atlassian.com/x", json_data={"a": 1})

    headers = mock_request.call_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"
    assert "X-Atlassian-Token" not in headers


# ---------------------------------------------------------------------------
# jira_add_attachment: rejections that must not send any request
# ---------------------------------------------------------------------------


def _assert_error_without_request(result: str, mock_request: Mock) -> dict:
    payload = json.loads(result)
    assert payload["status"] == "error"
    mock_request.assert_not_called()
    return payload


def test_add_attachment_rejects_path_outside_allowed_dirs(
    allowed_dir, tmp_path, monkeypatch
):
    outside = _write(tmp_path, name="secret.txt")
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    payload = _assert_error_without_request(
        jira.jira_add_attachment("ENG-1", str(outside)), mock_request
    )

    assert "allowed" in payload["message"]
    assert str(tmp_path) not in payload["message"]


def test_add_attachment_reports_the_allowlist_error_for_a_missing_path_outside_it(
    allowed_dir, tmp_path, monkeypatch
):
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    payload = _assert_error_without_request(
        jira.jira_add_attachment("ENG-1", str(tmp_path / "missing.txt")),
        mock_request,
    )

    assert "allowed" in payload["message"]
    assert "not found" not in payload["message"].lower()


@pytest.mark.parametrize(
    "scenario, expected",
    [
        ("missing", "not found"),
        ("empty", "empty"),
        ("oversized", "limit"),
    ],
)
def test_add_attachment_rejects_unusable_file_without_a_request(
    allowed_dir, monkeypatch, scenario, expected
):
    if scenario == "missing":
        path = allowed_dir / "nope.txt"
    elif scenario == "empty":
        path = _write(allowed_dir, data=b"")
    else:
        path = _write(allowed_dir, data=b"x" * 11)
        monkeypatch.setattr(jira_attachment, "MAX_ATTACHMENT_BYTES", 10)
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    payload = _assert_error_without_request(
        jira.jira_add_attachment("ENG-1", str(path)), mock_request
    )

    assert expected in payload["message"].lower()


def test_add_attachment_rejects_blank_issue_key_without_a_request(
    allowed_dir, monkeypatch
):
    path = _write(allowed_dir)
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    _assert_error_without_request(jira.jira_add_attachment("", str(path)), mock_request)


def test_add_attachment_reports_invalid_allowlist_configuration(
    tmp_path, monkeypatch, caplog
):
    path = _write(tmp_path)
    monkeypatch.setenv(_ALLOWED_DIRS_ENV_VAR, "[")
    mock_request = Mock()
    monkeypatch.setattr(jira.requests, "request", mock_request)

    with caplog.at_level(logging.WARNING, logger="jira-mcp"):
        payload = _assert_error_without_request(
            jira.jira_add_attachment("ENG-1", str(path)), mock_request
        )

    assert payload["message"] == "Upload directory configuration is invalid"
    assert _ALLOWED_DIRS_ENV_VAR in caplog.text


# ---------------------------------------------------------------------------
# jira_add_attachment: error mapping and response validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status, expected",
    [
        (403, "Create attachments"),
        (404, "cannot view"),
        (404, "cloud_id"),
        (413, "too large"),
        # A gateway answering after the body was sent leaves the outcome open.
        (502, "may have completed"),
        (503, "may have completed"),
        (504, "may have completed"),
    ],
)
def test_add_attachment_maps_http_errors_to_actionable_messages(
    allowed_dir, monkeypatch, status, expected
):
    path = _write(allowed_dir)
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=status,
                json_data={"errorMessages": ["Jira says no"]},
            )
        ),
    )

    payload = json.loads(
        jira.jira_add_attachment("ENG-1", str(path), cloud_id="site-a")
    )

    assert payload["status"] == "error"
    assert expected in payload["message"]
    assert str(status) in payload["message"]
    # Jira's own detail is kept alongside the hint.
    assert "Jira says no" in payload["message"]


def test_add_attachment_passes_other_errors_through_unchanged(allowed_dir, monkeypatch):
    path = _write(allowed_dir)
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(return_value=MockResponse(status_code=500, text="boom")),
    )

    payload = json.loads(
        jira.jira_add_attachment("ENG-1", str(path), cloud_id="site-a")
    )

    assert payload["status"] == "error"
    assert "500" in payload["message"]
    assert "boom" in payload["message"]


def test_add_attachment_gives_no_upload_hint_for_a_site_lookup_failure(
    allowed_dir, monkeypatch
):
    """With an empty cloud_id the site lookup runs first. A 403 there is not
    about attachment permissions, so the upload hint must not be attached."""
    path = _write(allowed_dir)
    mock_request = Mock(
        return_value=MockResponse(
            status_code=403,
            text="forbidden",
            url="https://api.atlassian.com/oauth/token/accessible-resources",
        )
    )
    monkeypatch.setattr(jira.requests, "request", mock_request)

    payload = json.loads(jira.jira_add_attachment("ENG-1", str(path)))

    assert payload["status"] == "error"
    assert "403" in payload["message"]
    assert "Create attachments" not in payload["message"]
    # Only the site lookup was attempted; no upload was sent.
    assert mock_request.call_count == 1


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        "unexpected",
        [None],
        [{"filename": "report.txt", "size": 10}],
        [{"id": "", "filename": "report.txt", "size": 10}],
        [{"id": " 1 ", "filename": "report.txt", "size": 10}],
    ],
)
def test_add_attachment_does_not_report_success_without_a_confirmed_attachment(
    allowed_dir, monkeypatch, body
):
    path = _write(allowed_dir, data=b"hello jira")
    monkeypatch.setattr(
        jira.requests, "request", Mock(return_value=MockResponse(json_data=body))
    )

    payload = json.loads(
        jira.jira_add_attachment("ENG-1", str(path), cloud_id="site-a")
    )

    assert payload["status"] == "error"
    assert "did not confirm" in payload["message"]


@pytest.mark.parametrize("reported_size", [0, 9, 11, None, "10"])
def test_add_attachment_flags_a_size_mismatch_and_names_the_created_attachment(
    allowed_dir, monkeypatch, reported_size
):
    path = _write(allowed_dir, data=b"hello jira")
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            return_value=MockResponse(json_data=_created(10, size=reported_size)),
        ),
    )

    payload = json.loads(
        jira.jira_add_attachment("ENG-1", str(path), cloud_id="site-a")
    )

    assert payload["status"] == "error"
    # The attachment exists on the issue, so the message must say so -- an
    # agent that retries blindly would otherwise attach a duplicate.
    assert "10001" in payload["message"]
    assert "before retrying" in payload["message"]


def test_add_attachment_redacts_credentials_from_errors(allowed_dir, monkeypatch):
    path = _write(allowed_dir)
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=500,
                text="gateway echoed Authorization: Bearer sk-abc123XYZ",
            )
        ),
    )

    payload = json.loads(
        jira.jira_add_attachment("ENG-1", str(path), cloud_id="site-a")
    )

    assert "sk-abc123XYZ" not in payload["message"]


@pytest.mark.parametrize(
    "error",
    [
        requests.ReadTimeout("read timed out"),
        # A send-side timeout surfaces as a ConnectionError, not a Timeout.
        requests.ConnectionError("Connection aborted.", TimeoutError("timed out")),
        # Nothing was sent here, but a refused connection or a failed TLS
        # handshake cannot be told apart from a drop mid-send, so it gets the
        # hint too: checking once is cheaper than a duplicate attachment.
        requests.exceptions.SSLError("handshake failed"),
        requests.exceptions.ChunkedEncodingError("response ended prematurely"),
        requests.exceptions.JSONDecodeError("Expecting value", "<html>", 0),
    ],
    ids=[
        "read-timeout",
        "send-timeout-as-connection-error",
        "tls-failure-treated-as-possibly-sent",
        "truncated-response",
        "non-json-response",
    ],
)
def test_add_attachment_says_the_upload_may_have_completed_when_the_outcome_is_unknown(
    allowed_dir, monkeypatch, error
):
    """The file may already be on the issue when the connection drops or the
    answer is unusable, so a blind retry would attach a duplicate."""
    path = _write(allowed_dir)
    mock_request = Mock(side_effect=error)
    monkeypatch.setattr(jira.requests, "request", mock_request)

    payload = json.loads(
        jira.jira_add_attachment("ENG-1", str(path), cloud_id="site-a")
    )

    assert payload["status"] == "error"
    assert "may have completed" in payload["message"]
    assert "check the issue" in payload["message"]
    assert mock_request.call_count == 1


def test_add_attachment_gives_no_completion_hint_for_a_connect_timeout(
    allowed_dir, monkeypatch
):
    """A ConnectTimeout is the one failure known to have sent nothing."""
    path = _write(allowed_dir)
    monkeypatch.setattr(
        jira.requests,
        "request",
        Mock(side_effect=requests.ConnectTimeout("connect timed out")),
    )

    payload = json.loads(
        jira.jira_add_attachment("ENG-1", str(path), cloud_id="site-a")
    )

    assert payload["status"] == "error"
    assert "may have completed" not in payload["message"]


def test_add_attachment_gives_no_completion_hint_for_a_site_lookup_connection_error(
    allowed_dir, monkeypatch
):
    """With an empty cloud_id the site lookup runs first. A dropped connection
    there happened before any upload, so claiming the upload may have
    completed would be false."""
    path = _write(allowed_dir)
    mock_request = Mock(side_effect=requests.ConnectionError("lookup dropped"))
    monkeypatch.setattr(jira.requests, "request", mock_request)

    payload = json.loads(jira.jira_add_attachment("ENG-1", str(path)))

    assert payload["status"] == "error"
    assert "may have completed" not in payload["message"]
    assert "lookup dropped" in payload["message"]
    # Only the site lookup was attempted; no upload was sent.
    assert mock_request.call_count == 1


def test_a_staged_chat_file_passes_the_jira_allowlist_guard(tmp_path, monkeypatch):
    """The headline case: the user sends a file in chat and asks for it to be
    attached. _DURABLE_UPLOAD_FIELDS makes the host stage the registered file
    into the task workspace, and the connector's guard must accept that path
    under the allowlist WebToolConfig injects, or the file is rejected."""
    workspace = TaskWorkspace(id="web_task_9999", base_dir=str(tmp_path))
    source = workspace.output_dir / "report.pdf"
    source.write_bytes(b"%PDF-1.4 chat file")
    monkeypatch.setattr(workspace, "resolve_file_id_detached", lambda file_id: source)
    staged = workspace.stage_file_for_external_upload("chat-file-id")

    config = WebToolConfig(
        db=None,
        request=None,
        task_id="web_task_9999",
        workspace_base_dir=str(tmp_path),
    )
    monkeypatch.setenv(
        _ALLOWED_DIRS_ENV_VAR, json.dumps(config._mcp_file_allowed_dir_paths())
    )

    assert jira_attachment.read_allowed_file(str(staged)) == (
        "report.pdf",
        b"%PDF-1.4 chat file",
    )


@pytest.mark.asyncio
async def test_the_jira_adapter_stages_a_chat_file_id_into_file_path():
    """The wiring _DURABLE_UPLOAD_FIELDS exists for: the adapter built for the
    Jira connector advertises that a file_id may be passed, and swaps a
    registered file:<id> for a staged workspace path before the tool runs."""
    staged_path = Path("/task/temp/.xagent-internal/mcp-upload/abc/report.pdf")

    class FakeWorkspace:
        def stage_file_for_external_upload(self, file_id):
            return staged_path

    mcp_tool = SimpleNamespace(
        name="jira_add_attachment",
        description="Attach a local file to an issue.",
        inputSchema={
            "type": "object",
            "properties": {
                "issue_key": {"type": "string"},
                "file_path": {"type": "string"},
            },
        },
    )
    adapter = _build_mcp_tool_adapter(
        "Jira",
        {"transport": "stdio", "command": "python", "args": []},
        mcp_tool,
        workspace=FakeWorkspace(),
    )

    assert "file_id" in adapter.description
    prepared, staged, sources = await adapter._stage_external_upload_args(
        {
            "issue_key": "ENG-1",
            "file_path": "file:31218a9f-1497-4216-9f75-1fc8d51368c4",
        }
    )
    assert prepared == {"issue_key": "ENG-1", "file_path": str(staged_path)}
    assert staged == [staged_path]
