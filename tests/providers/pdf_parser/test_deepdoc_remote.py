"""Tests for the remote DeepDoc ``task=parse`` client.

Every test here is a pure unit test: an ``httpx.MockTransport`` is injected
through the ``_transport`` seam of :func:`parse_document_remote`, so nothing
touches the network, and ``save_image`` is a plain callable, so no DeepDoc
parser is imported or constructed and no ONNX model cache is needed.

The failure cases matter more than the happy path. The client's contract is
that *every* remote problem surfaces as :class:`DeepDocRemoteError` and nothing
else, because the caller's single ``except DeepDocRemoteError`` clause is what
makes the fallback to local parsing unconditional. A leaked
``httpx.ConnectError`` or ``binascii.Error`` would escape that clause and break
parsing outright, so each failure test asserts the type is not merely "an
exception" but exactly that one.

Two shapes are asserted against the live server contract rather than a guess:
the whole task configuration travels as one JSON ``kwargs`` form field, and
``image_base64`` is *omitted* for elements without a crop rather than nulled.
"""

import base64
import binascii
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import pytest

from xagent.providers.pdf_parser.deepdoc_remote import (
    IMAGE_SCOPE,
    MAX_ZOOMIN,
    PARSE_ENDPOINT,
    TOKEN_ENDPOINT,
    DeepDocRemoteError,
    _build_headers,
    _normalize_elements,
    is_remote_configured,
    parse_document_remote,
)

BASE_URL = "http://gpu-host.internal:9997"
API_KEY_ENV = "XAGENT_DEEPDOC_XINFERENCE_API_KEY"
SHARED_API_KEY_ENV = "XINFERENCE_API_KEY"
URL_ENV = "XAGENT_DEEPDOC_XINFERENCE_URL"
TIMEOUT_ENV = "XAGENT_DEEPDOC_XINFERENCE_TIMEOUT_SECONDS"
MODEL_UID_ENV = "XAGENT_DEEPDOC_XINFERENCE_MODEL_UID"
USERNAME_ENV = "XAGENT_DEEPDOC_XINFERENCE_USERNAME"
PASSWORD_ENV = "XAGENT_DEEPDOC_XINFERENCE_PASSWORD"

# Smallest possible real PNG, so the decoded bytes are a plausible image rather
# than arbitrary base64 padding.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB"
    "/wFa1z9CAAAAAElFTkSuQmCC"
)
PNG_1X1_BASE64 = base64.b64encode(PNG_1X1).decode("ascii")


@pytest.fixture(autouse=True)
def remote_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure remote parsing and drop any ambient Xinference credentials.

    ``tests/conftest.py`` loads ``.env``/``example.env``, so a developer key in
    the environment would otherwise decide whether an ``Authorization`` header
    is sent. Every credential variable is cleared here and set explicitly by
    the tests that care.
    """
    monkeypatch.setenv(URL_ENV, BASE_URL)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(SHARED_API_KEY_ENV, raising=False)
    monkeypatch.delenv(TIMEOUT_ENV, raising=False)
    monkeypatch.delenv(MODEL_UID_ENV, raising=False)
    monkeypatch.delenv(USERNAME_ENV, raising=False)
    monkeypatch.delenv(PASSWORD_ENV, raising=False)


class RecordingSaveImage:
    """Stand-in for the parser's image writer, recording what it was handed."""

    def __init__(self, path: str = "/artifacts/table_0.png") -> None:
        self.path = path
        self.calls: list[bytes] = []

    def __call__(self, image_bytes: bytes) -> str:
        self.calls.append(image_bytes)
        return self.path


def json_transport(
    payload: Any,
    *,
    status_code: int = 200,
    sink: Optional[list[httpx.Request]] = None,
) -> httpx.MockTransport:
    """Return a transport answering every request with ``payload`` as JSON."""

    def handler(request: httpx.Request) -> httpx.Response:
        if sink is not None:
            sink.append(request)
        return httpx.Response(status_code, json=payload, request=request)

    return httpx.MockTransport(handler)


def routed_transport(
    parse_payload: Any,
    *,
    token: str = "jwt-token",
    token_status: int = 200,
    token_payload: Any = None,
    sink: Optional[list[httpx.Request]] = None,
) -> httpx.MockTransport:
    """Return a transport that answers ``/token`` and the parse endpoint apart.

    The JWT exchange and the parse call share one transport, so a single handler
    has to serve both; routing on the path is what lets a test assert the
    ordering and the token actually reaching the parse request.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if sink is not None:
            sink.append(request)
        if request.url.path == TOKEN_ENDPOINT:
            body = (
                token_payload
                if token_payload is not None
                else {"access_token": token, "token_type": "bearer"}
            )
            return httpx.Response(token_status, json=body, request=request)
        return httpx.Response(200, json=parse_payload, request=request)

    return httpx.MockTransport(handler)


def sample_payload() -> dict[str, Any]:
    """A live-shaped response: an image-less title, and a table carrying a PNG.

    The title element omits ``image_base64`` entirely rather than sending
    ``null``, which is what the server actually does for elements without a
    crop.
    """
    return {
        "task": "parse",
        "elements": [
            {
                "type": "title",
                "text": "Sample Document",
                "metadata": {
                    "x0": 70.666,
                    "x1": 256.333,
                    "top": 77.333,
                    "bottom": 96.333,
                    "page_number": 1,
                    "layout_type": "title",
                    "layoutno": "title-0",
                    "col_id": 0,
                    "positions": [[1, 70.7, 256.3, 77.3, 96.3]],
                },
            },
            {
                "type": "table",
                "text": "<table><caption>T1</caption><tr><th>Cell</th></tr></table>",
                "image_base64": PNG_1X1_BASE64,
                "metadata": {
                    "page_number": 2,
                    "layout_type": "table",
                    "positions": [[2, 20.0, 400.0, 50.0, 200.0]],
                },
            },
        ],
    }


def form_fields(request: httpx.Request) -> dict[str, str]:
    """Extract the simple (non-file) multipart form fields from a request.

    The client sends its task configuration as form fields next to the upload,
    so several tests need to read them back. httpx offers no multipart decoder,
    hence the deliberate minimal parse: split on the boundary and keep the parts
    that carry a ``name`` but no ``filename``.
    """
    body = request.content.decode("utf-8", errors="replace")
    fields: dict[str, str] = {}
    for part in body.split("--"):
        match = re.search(r'name="([^"]+)"', part)
        if match is None or "filename=" in part:
            continue
        _, _, value = part.partition("\r\n\r\n")
        fields[match.group(1)] = value.strip("\r\n")
    return fields


def parse_kwargs(request: httpx.Request) -> dict[str, Any]:
    """Return the decoded ``kwargs`` JSON object the client sent."""
    decoded = json.loads(form_fields(request)["kwargs"])
    assert isinstance(decoded, dict)
    return decoded


class TestParseDocumentRemoteSuccess:
    """Happy-path element shape and request shape."""

    def test_elements_are_translated_to_local_shape(self, tmp_path: Path) -> None:
        """image_base64 becomes an ``image`` path, or None, and metadata survives."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        save_image = RecordingSaveImage()

        elements = parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=save_image,
            _transport=json_transport(sample_payload()),
        )

        assert len(elements) == 2
        title, table = elements

        # image_base64 is consumed, never forwarded.
        assert "image_base64" not in title
        assert "image_base64" not in table

        assert title["type"] == "title"
        assert title["text"] == "Sample Document"
        assert title["image"] is None
        # Document-wide coordinates and positions reach the translator intact.
        assert title["metadata"]["positions"] == [[1, 70.7, 256.3, 77.3, 96.3]]
        assert title["metadata"]["col_id"] == 0
        assert title["metadata"]["layout_type"] == "title"

        assert table["type"] == "table"
        assert table["text"].startswith("<table><caption>")
        assert table["image"] == save_image.path
        # col_id is absent on this element; nothing invents one.
        assert "col_id" not in table["metadata"]

        # Only the table carried an image, so exactly one write happened.
        assert save_image.calls == [PNG_1X1]

    def test_a_single_http_call_is_made(self, tmp_path: Path) -> None:
        """task=parse runs the whole pipeline server-side; there is nothing to stitch."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport(sample_payload(), sink=requests),
        )

        assert len(requests) == 1
        assert str(requests[0].url) == f"{BASE_URL}{PARSE_ENDPOINT}"

    def test_callback_reports_start_and_completion(self, tmp_path: Path) -> None:
        """The completion notice keeps the ``message (1.23s)`` shape local DeepDoc emits."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        progress: list[tuple[float, str]] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            callback=lambda fraction, message: progress.append((fraction, message)),
            _transport=json_transport(sample_payload()),
        )

        assert [fraction for fraction, _ in progress] == [0.05, 1.0]
        assert progress[-1][1].startswith("Remote DeepDoc parse finished (")
        assert progress[-1][1].endswith("s)")

    def test_request_carries_the_pdf_model_and_task_kwargs(
        self, tmp_path: Path
    ) -> None:
        """The upload is the ``image`` part, typed as a PDF, beside model and kwargs."""
        source = tmp_path / "quarterly report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            zoomin=5,
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert len(requests) == 1
        request = requests[0]
        assert request.method == "POST"
        assert str(request.url) == f"{BASE_URL}{PARSE_ENDPOINT}"

        body = request.content.decode("utf-8", errors="replace")
        # The server reads the upload from the "image" field, not "file".
        assert 'name="image"' in body
        assert 'name="file"' not in body
        assert 'filename="quarterly report.pdf"' in body
        assert "application/pdf" in body
        assert "%PDF-1.7 fake" in body

        assert form_fields(request)["model"] == "DeepDoc"
        assert parse_kwargs(request) == {
            "task": "parse",
            "zoomin": 5,
            "image_scope": IMAGE_SCOPE,
        }

    def test_configured_timeout_reaches_read_and_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The configured timeout bounds the upload and the parse, not the connect.

        A whole-document parse can legitimately take many minutes, so ``read``
        and ``write`` follow the configured value while ``connect`` and ``pool``
        stay short -- an unreachable host should fail fast rather than hang for
        the parse budget.
        """
        monkeypatch.setenv(TIMEOUT_ENV, "1234")
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        seen: list[httpx.Timeout] = []

        real_client = httpx.Client

        def capturing_client(*args: Any, **kwargs: Any) -> httpx.Client:
            timeout = kwargs.get("timeout")
            if isinstance(timeout, httpx.Timeout):
                seen.append(timeout)
            return real_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", capturing_client)

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}),
        )

        assert seen, "no httpx.Client was constructed with an explicit timeout"
        timeout = seen[0]
        assert timeout.read == 1234
        assert timeout.write == 1234
        assert timeout.connect == 10
        assert timeout.pool == 10

    def test_zoomin_defaults_to_three(self, tmp_path: Path) -> None:
        """The default matches the local parser's parse_into_bboxes(zoomin=3)."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert parse_kwargs(requests[0])["zoomin"] == 3

    def test_model_uid_is_configurable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.setenv(MODEL_UID_ENV, "deepdoc-gpu-1")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert form_fields(requests[0])["model"] == "deepdoc-gpu-1"

    def test_api_key_is_sent_as_the_bearer_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.setenv(API_KEY_ENV, "secret-key")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        # No token exchange when a key is configured: one request, one header.
        assert len(requests) == 1
        assert requests[0].headers["authorization"] == "Bearer secret-key"

    def test_credentials_are_exchanged_for_a_jwt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """POST /token first, then the parse call carrying the minted token."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "admin123")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=routed_transport(
                sample_payload(), token="minted-jwt", sink=requests
            ),
        )

        assert [request.url.path for request in requests] == [
            TOKEN_ENDPOINT,
            PARSE_ENDPOINT,
        ]
        token_request, parse_request = requests
        assert json.loads(token_request.content) == {
            "username": "admin",
            "password": "admin123",
        }
        assert parse_request.headers["authorization"] == "Bearer minted-jwt"

    def test_credentials_win_over_a_configured_api_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A JWT is short-lived and scoped; prefer it when both are available."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "admin123")
        monkeypatch.setenv(API_KEY_ENV, "static-key")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=routed_transport(
                sample_payload(), token="minted-jwt", sink=requests
            ),
        )

        assert requests[-1].headers["authorization"] == "Bearer minted-jwt"

    @pytest.mark.parametrize(
        ("username", "password"),
        [
            pytest.param("admin", None, id="password-missing"),
            pytest.param(None, "admin123", id="username-missing"),
        ],
    )
    def test_half_configured_credentials_fall_back_to_the_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        username: Optional[str],
        password: Optional[str],
    ) -> None:
        """An incomplete pair must not send a doomed token request."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        if username is not None:
            monkeypatch.setenv(USERNAME_ENV, username)
        if password is not None:
            monkeypatch.setenv(PASSWORD_ENV, password)
        monkeypatch.setenv(API_KEY_ENV, "static-key")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert [request.url.path for request in requests] == [PARSE_ENDPOINT]
        assert requests[0].headers["authorization"] == "Bearer static-key"

    def test_authorization_header_absent_when_nothing_is_configured(
        self, tmp_path: Path
    ) -> None:
        """An unauthenticated self-hosted Xinference must not receive a bogus header."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert "authorization" not in requests[0].headers

    @pytest.mark.parametrize(
        "configured_url",
        [
            BASE_URL,
            f"{BASE_URL}/",
            f"{BASE_URL}///",
            f"{BASE_URL}  ",
        ],
        ids=["bare", "trailing-slash", "many-slashes", "trailing-space"],
    )
    def test_url_has_no_double_slash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured_url: str
    ) -> None:
        """The config getter strips the trailing slash, so the path joins cleanly."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.setenv(URL_ENV, configured_url)
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        url = str(requests[0].url)
        assert url == f"{BASE_URL}{PARSE_ENDPOINT}"
        assert "//v1" not in url

    def test_bytesio_uploads_whole_buffer_and_restores_position(self) -> None:
        """An in-memory PDF uploads as ``document.pdf`` and is left as it was found."""
        stream = BytesIO(b"%PDF-1.7 in-memory document body")
        stream.seek(4)
        requests: list[httpx.Request] = []

        parse_document_remote(
            stream,
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}, sink=requests),
        )

        body = requests[0].content.decode("utf-8", errors="replace")
        assert 'filename="document.pdf"' in body
        assert "application/pdf" in body
        # The whole buffer went up, not just the tail after the cursor.
        assert "%PDF-1.7 in-memory document body" in body
        # The caller may still read from its own buffer after a fallback.
        assert stream.tell() == 4

    def test_empty_element_list_is_a_valid_response(self, tmp_path: Path) -> None:
        """A document with no extractable content is a success, not a failure."""
        source = tmp_path / "empty.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        save_image = RecordingSaveImage()

        elements = parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=save_image,
            _transport=json_transport({"task": "parse", "elements": []}),
        )

        assert elements == []
        assert save_image.calls == []


def error_transport(exc: Exception) -> httpx.MockTransport:
    """Return a transport whose handler raises ``exc`` instead of responding."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return httpx.MockTransport(handler)


def raw_transport(
    body: bytes, *, status_code: int = 200, content_type: str = "application/json"
) -> httpx.MockTransport:
    """Return a transport answering with an exact byte body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            content=body,
            headers={"content-type": content_type},
            request=request,
        )

    return httpx.MockTransport(handler)


def failing_save_image(exc: Exception) -> Callable[[bytes], str]:
    """Return a ``save_image`` that raises ``exc`` when the client writes an image."""

    def save_image(image_bytes: bytes) -> str:
        raise exc

    return save_image


class TestParseDocumentRemoteFailures:
    """Every remote problem must surface as DeepDocRemoteError and nothing else."""

    @pytest.mark.parametrize(
        "transport_factory",
        [
            pytest.param(
                lambda: json_transport({"detail": "inference failed"}, status_code=500),
                id="http-500",
            ),
            pytest.param(
                lambda: json_transport({"detail": "bad token"}, status_code=401),
                id="http-401",
            ),
            pytest.param(
                lambda: json_transport(
                    {"detail": "task 'parse' requires a PDF"}, status_code=400
                ),
                id="http-400",
            ),
            pytest.param(
                lambda: error_transport(
                    httpx.ConnectError("connection refused"),
                ),
                id="connection-error",
            ),
            pytest.param(
                lambda: error_transport(httpx.ReadTimeout("read timed out")),
                id="read-timeout",
            ),
            pytest.param(
                lambda: raw_transport(b"<html>502 Bad Gateway</html>"),
                id="non-json-body",
            ),
            pytest.param(lambda: raw_transport(b""), id="empty-body"),
            pytest.param(
                lambda: json_transport([{"type": "text", "text": "x"}]),
                id="top-level-list",
            ),
            pytest.param(
                lambda: json_transport({"task": "parse"}),
                id="elements-missing",
            ),
            pytest.param(
                lambda: json_transport({"elements": None}),
                id="elements-none",
            ),
            pytest.param(
                lambda: json_transport({"elements": {"type": "text"}}),
                id="elements-not-a-list",
            ),
            pytest.param(
                lambda: json_transport({"elements": ["just a string"]}),
                id="element-not-a-dict",
            ),
            pytest.param(
                lambda: json_transport({"elements": [None]}),
                id="element-is-none",
            ),
            pytest.param(
                lambda: json_transport({"elements": [{"type": "text"}]}),
                id="element-missing-text",
            ),
            pytest.param(
                lambda: json_transport({"elements": [{"text": "x"}]}),
                id="element-missing-type",
            ),
            pytest.param(
                lambda: json_transport(
                    {
                        "elements": [
                            {
                                "type": "table",
                                "text": "<table></table>",
                                "image_base64": "a",
                            }
                        ]
                    }
                ),
                id="undecodable-image",
            ),
        ],
    )
    def test_failure_raises_deepdoc_remote_error(
        self, tmp_path: Path, transport_factory: Callable[[], httpx.MockTransport]
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=transport_factory(),
            )

    @pytest.mark.parametrize(
        "transport_factory",
        [
            pytest.param(
                lambda: routed_transport(sample_payload(), token_status=401),
                id="token-rejected",
            ),
            pytest.param(
                lambda: routed_transport(sample_payload(), token_status=500),
                id="token-server-error",
            ),
            pytest.param(
                lambda: routed_transport(sample_payload(), token_payload={}),
                id="token-field-missing",
            ),
            pytest.param(
                lambda: routed_transport(
                    sample_payload(), token_payload={"access_token": ""}
                ),
                id="token-empty",
            ),
            pytest.param(
                lambda: routed_transport(
                    sample_payload(), token_payload={"access_token": 42}
                ),
                id="token-not-a-string",
            ),
            pytest.param(
                lambda: routed_transport(sample_payload(), token_payload=["a"]),
                id="token-body-not-an-object",
            ),
        ],
    )
    def test_token_exchange_failure_raises_deepdoc_remote_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        transport_factory: Callable[[], httpx.MockTransport],
    ) -> None:
        """A broken JWT exchange falls back locally like any other remote failure."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "admin123")

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=transport_factory(),
            )

    def test_a_failed_token_exchange_makes_no_parse_request(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Uploading a whole PDF with a credential the server rejected is pure waste."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "wrong")
        requests: list[httpx.Request] = []

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=routed_transport(
                    sample_payload(), token_status=401, sink=requests
                ),
            )

        assert [request.url.path for request in requests] == [TOKEN_ENDPOINT]

    @pytest.mark.parametrize(
        "ext",
        [".docx", ".xlsx", ".csv", ".md", ".txt", ".json", ".html", ""],
    )
    def test_non_pdf_is_rejected_without_an_http_call(
        self, tmp_path: Path, ext: str
    ) -> None:
        """task=parse consumes PDFs only, so anything else must not reach the wire."""
        source = tmp_path / f"document{ext or '.bin'}"
        source.write_bytes(b"not a pdf")
        requests: list[httpx.Request] = []

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=ext,
                save_image=RecordingSaveImage(),
                _transport=json_transport(sample_payload(), sink=requests),
            )

        assert requests == []

    @pytest.mark.parametrize(
        "zoomin",
        [
            pytest.param(0, id="zero"),
            pytest.param(-1, id="negative"),
            pytest.param(7, id="above-cap"),
            pytest.param(9, id="well-above-cap"),
            pytest.param(3.0, id="float"),
            pytest.param(True, id="bool"),
            pytest.param("3", id="string"),
        ],
    )
    def test_out_of_range_zoomin_is_rejected_without_an_http_call(
        self, tmp_path: Path, zoomin: Any
    ) -> None:
        """The server caps zoomin at 6; uploading a doomed PDF is pure waste."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                zoomin=zoomin,
                _transport=json_transport(sample_payload(), sink=requests),
            )

        assert requests == []

    @pytest.mark.parametrize("zoomin", [1, 3, MAX_ZOOMIN])
    def test_in_range_zoomin_is_accepted(self, tmp_path: Path, zoomin: int) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        requests: list[httpx.Request] = []

        parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            zoomin=zoomin,
            _transport=json_transport({"elements": []}, sink=requests),
        )

        assert parse_kwargs(requests[0])["zoomin"] == zoomin

    def test_uppercase_pdf_extension_is_accepted(self, tmp_path: Path) -> None:
        """The caller lower-cases the suffix, but the guard must not depend on that."""
        source = tmp_path / "REPORT.PDF"
        source.write_bytes(b"%PDF-1.7 fake")

        elements = parse_document_remote(
            str(source),
            ext=".PDF",
            save_image=RecordingSaveImage(),
            _transport=json_transport({"elements": []}),
        )

        assert elements == []

    @pytest.mark.parametrize(
        "exc",
        [
            pytest.param(RuntimeError("artifact dir vanished"), id="runtime-error"),
            pytest.param(OSError("disk full"), id="os-error"),
            pytest.param(
                binascii.Error("re-raised decode failure"), id="binascii-error"
            ),
        ],
    )
    def test_save_image_failure_raises_deepdoc_remote_error(
        self, tmp_path: Path, exc: Exception
    ) -> None:
        """save_image is caller-supplied, so any exception from it must be wrapped."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=failing_save_image(exc),
                _transport=json_transport(sample_payload()),
            )

    @pytest.mark.parametrize(
        ("transport_factory", "leaked_type"),
        [
            pytest.param(
                lambda: error_transport(httpx.ConnectError("connection refused")),
                httpx.ConnectError,
                id="connection-error",
            ),
            pytest.param(
                lambda: error_transport(httpx.ReadTimeout("read timed out")),
                httpx.ReadTimeout,
                id="read-timeout",
            ),
            pytest.param(
                lambda: json_transport({"detail": "boom"}, status_code=500),
                httpx.HTTPStatusError,
                id="http-500",
            ),
            pytest.param(
                lambda: raw_transport(b"<html>oops</html>"),
                ValueError,
                id="non-json-body",
            ),
            pytest.param(
                lambda: json_transport({"elements": [None]}),
                ValueError,
                id="element-is-none",
            ),
        ],
    )
    def test_underlying_exception_types_do_not_escape(
        self,
        tmp_path: Path,
        transport_factory: Callable[[], httpx.MockTransport],
        leaked_type: type[Exception],
    ) -> None:
        """Guards the parametrized suite above against a passing-for-the-wrong-reason bug.

        ``DeepDocRemoteError`` derives straight from ``Exception``, so it is not
        an instance of any of these. Asserting that keeps the failure suite
        honest: were the client to stop wrapping, the raised type would satisfy
        neither this check nor ``pytest.raises(DeepDocRemoteError)``.
        """
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError) as excinfo:
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=transport_factory(),
            )

        assert not isinstance(excinfo.value, leaked_type)
        assert isinstance(excinfo.value.__cause__, leaked_type)

    def test_missing_source_file_raises_deepdoc_remote_error(
        self, tmp_path: Path
    ) -> None:
        """An unreadable local file must fall back rather than raise OSError."""
        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(tmp_path / "does-not-exist.pdf"),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport(sample_payload()),
            )

    def test_unconfigured_url_raises_deepdoc_remote_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Calling the client without configuration is a caller bug, still wrapped."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        monkeypatch.delenv(URL_ENV, raising=False)

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport(sample_payload()),
            )

    def test_failure_does_not_invoke_the_completion_callback(
        self, tmp_path: Path
    ) -> None:
        """A failed parse must not report progress it did not make."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        progress: list[tuple[float, str]] = []

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                callback=lambda fraction, message: progress.append((fraction, message)),
                _transport=json_transport({"detail": "boom"}, status_code=500),
            )

        assert [fraction for fraction, _ in progress] == [0.05]

    def test_bytesio_position_is_restored_after_a_failure(self) -> None:
        """The buffer must be reusable by the local fallback path."""
        stream = BytesIO(b"%PDF-1.7 in-memory body")
        stream.seek(7)

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                stream,
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport({"detail": "boom"}, status_code=500),
            )

        assert stream.tell() == 7


class TestNormalizeElements:
    """Direct coverage of the translator, where wrapping is not in the way."""

    def test_absent_image_base64_still_yields_an_image_key(self) -> None:
        """The server omits the field entirely, and the caller always reads ``image``."""
        save_image = RecordingSaveImage()

        elements = _normalize_elements(
            {"elements": [{"type": "text", "text": "hello"}]}, save_image
        )

        assert elements == [{"type": "text", "text": "hello", "image": None}]
        assert save_image.calls == []

    @pytest.mark.parametrize("empty", [None, "", 0], ids=["none", "empty-str", "zero"])
    def test_falsy_image_base64_is_treated_as_no_image(self, empty: Any) -> None:
        save_image = RecordingSaveImage()

        elements = _normalize_elements(
            {"elements": [{"type": "text", "text": "hi", "image_base64": empty}]},
            save_image,
        )

        assert elements[0]["image"] is None
        assert save_image.calls == []

    def test_metadata_is_passed_through_untouched(self) -> None:
        """Coordinates, layoutno and positions all reach the downstream translator."""
        metadata = {
            "x0": 70.666,
            "top": 77.333,
            "page_number": 1,
            "layout_type": "title",
            "layoutno": "title-0",
            "positions": [[1, 70.7, 256.3, 77.3, 96.3]],
        }

        elements = _normalize_elements(
            {"elements": [{"type": "title", "text": "T", "metadata": metadata}]},
            RecordingSaveImage(),
        )

        assert elements[0]["metadata"] == metadata

    def test_source_payload_is_not_mutated(self) -> None:
        """Elements are copied, so a caller retrying locally sees its own data intact."""
        payload = {
            "elements": [
                {
                    "type": "table",
                    "text": "<table></table>",
                    "image_base64": PNG_1X1_BASE64,
                }
            ]
        }

        _normalize_elements(payload, RecordingSaveImage())

        assert payload["elements"][0]["image_base64"] == PNG_1X1_BASE64
        assert "image" not in payload["elements"][0]

    @pytest.mark.parametrize(
        "payload",
        [
            pytest.param({}, id="elements-missing"),
            pytest.param({"elements": None}, id="elements-none"),
            pytest.param({"elements": "text"}, id="elements-str"),
            pytest.param({"elements": [42]}, id="element-int"),
            pytest.param({"elements": [{"type": "text"}]}, id="missing-text"),
            pytest.param({"elements": [{"text": "x"}]}, id="missing-type"),
            # Present-but-null and wrongly-typed values must be rejected here.
            # Left through, they reach the translator and raise a pydantic
            # ValidationError, which the caller does not catch -- so the local
            # fallback would be skipped and the whole parse would fail.
            pytest.param({"elements": [{"type": None, "text": "x"}]}, id="null-type"),
            pytest.param(
                {"elements": [{"type": "text", "text": None}]}, id="null-text"
            ),
            pytest.param({"elements": [{"type": 1, "text": "x"}]}, id="int-type"),
            pytest.param({"elements": [{"type": "text", "text": 1}]}, id="int-text"),
        ],
    )
    def test_malformed_payloads_raise_value_error(self, payload: Any) -> None:
        """ValueError is what parse_document_remote translates into its own error."""
        with pytest.raises(ValueError):
            _normalize_elements(payload, RecordingSaveImage())

    def test_undecodable_image_raises_binascii_error(self) -> None:
        with pytest.raises(binascii.Error):
            _normalize_elements(
                {"elements": [{"type": "table", "text": "t", "image_base64": "abcde"}]},
                RecordingSaveImage(),
            )


class TestBuildHeaders:
    """Auth header construction: JWT exchange, API key, and the shared-key fallback."""

    def test_nothing_configured_yields_no_headers(self) -> None:
        assert _build_headers(BASE_URL, None) == {}

    def test_dedicated_key_is_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(API_KEY_ENV, "dedicated")
        assert _build_headers(BASE_URL, None) == {"Authorization": "Bearer dedicated"}

    def test_shared_key_is_the_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SHARED_API_KEY_ENV, "shared")
        assert _build_headers(BASE_URL, None) == {"Authorization": "Bearer shared"}

    def test_dedicated_key_wins_over_the_shared_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_KEY_ENV, "dedicated")
        monkeypatch.setenv(SHARED_API_KEY_ENV, "shared")
        assert _build_headers(BASE_URL, None) == {"Authorization": "Bearer dedicated"}

    def test_credentials_produce_a_minted_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "admin123")
        transport = routed_transport({"elements": []}, token="minted")

        assert _build_headers(BASE_URL, transport) == {"Authorization": "Bearer minted"}


class TestIsRemoteConfigured:
    """Configuration detection must never raise; a typo means local parsing."""

    def test_unset_url_is_not_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(URL_ENV, raising=False)
        assert is_remote_configured() is False

    @pytest.mark.parametrize(
        "value",
        ["", "   "],
        ids=["empty", "whitespace"],
    )
    def test_blank_url_is_not_configured(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(URL_ENV, value)
        assert is_remote_configured() is False

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("ftp://host", id="wrong-scheme"),
            pytest.param("gpu-host.internal:9997", id="no-scheme"),
            pytest.param("http://host:9997?token=x", id="query-string"),
            pytest.param("http://host:9997#frag", id="fragment"),
            pytest.param("http://", id="no-netloc"),
        ],
    )
    def test_malformed_url_degrades_to_local_without_raising(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """The config getter raises ValueError; is_remote_configured must swallow it."""
        monkeypatch.setenv(URL_ENV, value)
        assert is_remote_configured() is False

    def test_malformed_url_logs_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(URL_ENV, "ftp://host")

        with caplog.at_level("WARNING"):
            assert is_remote_configured() is False

        assert "parsing locally" in caplog.text

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param("http://host:9997", id="http"),
            pytest.param("https://host", id="https"),
            pytest.param("http://host:9997/", id="trailing-slash"),
            pytest.param("http://host:9997/base/path", id="with-path"),
        ],
    )
    def test_valid_url_is_configured(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(URL_ENV, value)
        assert is_remote_configured() is True


class TestBlankPasswordDoesNotHijackTheApiKey:
    """A whitespace-only password must not be mistaken for a credential.

    The password is deliberately not stripped, because whitespace can be
    significant in a secret. But a password of only spaces is a misconfigured
    variable, and treating it as real would take the JWT branch, ignore a
    working API key, and 401 forever — surfacing only as a permanent silent
    fallback to local parsing.
    """

    @pytest.mark.parametrize("password", ["", "   ", "\t\n"])
    def test_blank_password_falls_through_to_the_api_key(
        self, monkeypatch: pytest.MonkeyPatch, password: str
    ) -> None:
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, password)
        monkeypatch.setenv(API_KEY_ENV, "configured-key")

        assert _build_headers(BASE_URL, None) == {
            "Authorization": "Bearer configured-key"
        }

    def test_a_password_of_significant_whitespace_is_still_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Padding around a real secret is preserved, not stripped away."""
        monkeypatch.setenv(USERNAME_ENV, "admin")
        monkeypatch.setenv(PASSWORD_ENV, "  secret  ")
        sent: list[httpx.Request] = []

        headers = _build_headers(
            BASE_URL,
            routed_transport({"elements": []}, token="jwt-abc", sink=sent),
        )

        assert headers == {"Authorization": "Bearer jwt-abc"}
        body = json.loads(sent[0].content)
        assert body["password"] == "  secret  "


class TestProgressSinkCannotBreakTheParse:
    """A raising progress sink must never cost a parse.

    Both callbacks sit outside the function's own ``try``, so without their own
    guard the exception escapes as its own type and bypasses the caller's
    ``except DeepDocRemoteError``. On the entry call that skips the fallback; on
    the completion call it is worse, discarding a *successful* parse along with
    every crop already written to disk.
    """

    def test_raising_sink_on_entry_does_not_prevent_the_parse(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        def exploding_callback(progress: float, message: str) -> None:
            raise RuntimeError("sink is broken")

        elements = parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            callback=exploding_callback,
            _transport=json_transport(sample_payload()),
        )

        assert len(elements) == 2

    def test_raising_sink_on_completion_does_not_discard_the_result(
        self, tmp_path: Path
    ) -> None:
        """The completion notice fires after the crops are written; it must not undo them."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        save_image = RecordingSaveImage()
        calls: list[float] = []

        def late_exploding_callback(progress: float, message: str) -> None:
            calls.append(progress)
            if progress >= 1.0:
                raise RuntimeError("sink broke at the finish line")

        elements = parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=save_image,
            callback=late_exploding_callback,
            _transport=json_transport(sample_payload()),
        )

        assert calls == [0.05, 1.0]
        assert len(elements) == 2
        assert len(save_image.calls) == 1


class TestEnvelopedResponses:
    """The managed gateway wraps the parse result; a self-hosted server does not.

    Verified against the live SaaS endpoint, which answers
    ``{"code": 0, "message": "Request successful.", "data": {"task": ..., "elements": [...]}}``
    while a self-hosted Xinference returns that inner object directly. Both are
    legitimate deployments of the same contract, so the client accepts either.
    """

    def test_enveloped_payload_is_unwrapped(self, tmp_path: Path) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")
        save_image = RecordingSaveImage()

        elements = parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=save_image,
            _transport=json_transport(
                {
                    "code": 0,
                    "message": "Request successful.",
                    "data": sample_payload(),
                }
            ),
        )

        assert len(elements) == 2
        assert elements[0]["text"] == "Sample Document"
        assert elements[1]["image"] == save_image.path

    def test_direct_payload_still_works(self, tmp_path: Path) -> None:
        """The unwrapping must not regress the self-hosted shape."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        elements = parse_document_remote(
            str(source),
            ext=".pdf",
            save_image=RecordingSaveImage(),
            _transport=json_transport(sample_payload()),
        )

        assert len(elements) == 2

    def test_nonzero_code_is_surfaced_not_swallowed(self, tmp_path: Path) -> None:
        """The gateway reports some failures with HTTP 200 and a non-zero code."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError, match="code 40301"):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport(
                    {"code": 40301, "message": "quota exceeded", "data": None}
                ),
            )

    def test_envelope_without_elements_is_rejected(self, tmp_path: Path) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport({"code": 0, "data": {"task": "parse"}}),
            )


class TestNonStringImageBase64:
    """A JSON number or array in ``image_base64`` must name the offending field.

    Left to ``b64decode`` it raises ``TypeError``, which the caller reports as
    "argument should be a bytes-like object" — true, but useless for working out
    which element of which response was malformed.
    """

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(12345, id="int"),
            pytest.param(["a"], id="list"),
            pytest.param({"a": 1}, id="dict"),
            pytest.param(True, id="bool"),
        ],
    )
    def test_non_string_base64_names_the_field(
        self, tmp_path: Path, value: Any
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError, match="image_base64"):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport(
                    {
                        "elements": [
                            {"type": "table", "text": "t", "image_base64": value}
                        ]
                    }
                ),
            )


class TestEnvelopeCodeCheckedFirst:
    """A gateway reporting a failure is authoritative, whatever it left in ``data``."""

    def test_nonzero_code_wins_over_a_present_element_list(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError, match="40301"):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport(
                    {
                        "code": 40301,
                        "message": "quota exceeded",
                        "data": {"elements": [{"type": "text", "text": "x"}]},
                    }
                ),
            )

    def test_non_int_code_is_still_reported(self, tmp_path: Path) -> None:
        """A string code would previously fall through to the generic message."""
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-1.7 fake")

        with pytest.raises(DeepDocRemoteError, match="ERR_QUOTA"):
            parse_document_remote(
                str(source),
                ext=".pdf",
                save_image=RecordingSaveImage(),
                _transport=json_transport({"code": "ERR_QUOTA", "data": None}),
            )
