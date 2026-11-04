"""Thumbnail API, validation and output behavior with an HTTPX mock transport."""

import asyncio
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from PIL import Image

from xagent.core.utils import security
from xagent.web.tools.mcp import google_slides

# A real PNG produced by Pillow.  Decodable image bytes keep the "the saved
# file is a valid image" assertion meaningful; the signature-only and truncated
# bodies in the invalid-image tests are rejected by the integrity check.
_PNG_BYTES = io.BytesIO()
Image.new("RGB", (2, 3), (10, 20, 30)).save(_PNG_BYTES, format="PNG")
_PNG_BYTES = _PNG_BYTES.getvalue()
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_THUMB_HOST = "lh3.googleusercontent.com"
_THUMB_URL = f"https://{_THUMB_HOST}/pages/thumb?token=thumb-secret-token"


def _thumbnail_tool():
    tool = getattr(google_slides, "google_slides_get_slide_thumbnail", None)
    assert tool is not None, (
        "google_slides_get_slide_thumbnail is not defined on the connector module"
    )
    return tool


@pytest.fixture(autouse=True)
def _google_env(monkeypatch, tmp_path):
    # The connector only needs a token to build its (mocked) API client; the
    # image download must never use it.  Every test starts from a task
    # workspace output directory and with no ambient proxy configuration.
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", str(tmp_path))
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


class _SlidesApi:
    """A mocked Slides API service with an inspectable thumbnail call."""

    def __init__(self):
        self.service = Mock()
        self.presentations = Mock()
        self.pages = Mock()
        self.service.presentations.return_value = self.presentations
        self.presentations.pages.return_value = self.pages
        self.get_thumbnail = self.pages.getThumbnail
        self.set_thumbnail()
        self.set_slides("s1", "s2", "s3", "s4", "s5")

    def set_thumbnail(self, payload=None, error=None):
        request = self.get_thumbnail.return_value
        if error is not None:
            request.execute.side_effect = error
        else:
            request.execute.return_value = (
                {
                    "contentUrl": _THUMB_URL,
                    "width": 1920,
                    "height": 1080,
                }
                if payload is None
                else payload
            )

    def set_slides(self, *object_ids, error=None):
        request = self.presentations.get.return_value
        if error is not None:
            request.execute.side_effect = error
        else:
            request.execute.return_value = {
                "slides": [{"objectId": object_id} for object_id in object_ids]
            }


def _install_slides_api(monkeypatch):
    api = _SlidesApi()
    monkeypatch.setattr(google_slides, "get_slides_service", lambda: api.service)
    return api


def _install_http(monkeypatch, responder):
    """Exercise the shared fetcher with an isolated HTTPX mock transport."""
    state = {"calls": [], "sessions": []}

    async def resolve(url):
        return ["93.184.216.34"]

    def respond(request):
        state["calls"].append((request, {}))
        return responder(len(state["calls"]) - 1, request, state)

    @asynccontextmanager
    async def client(proxy, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as session:
            state["sessions"].append(session)
            yield session

    monkeypatch.setattr(security, "validate_public_http_url", resolve)
    monkeypatch.setattr(google_slides, "public_fetch_client", client)
    return state


def _http_response(body=b"", status=200, headers=None, url=_THUMB_URL):
    return httpx.Response(status, content=body, headers=headers)


def _call_tool(**kwargs):
    return json.loads(asyncio.run(_thumbnail_tool()(**kwargs)))


def _result_path(result):
    return Path(result["path"])


def test_thumbnail_uses_the_slide_object_id_and_png_with_default_large(monkeypatch):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "success"
    api.get_thumbnail.assert_called_once_with(
        presentationId="pres1",
        pageObjectId="s1",
        thumbnailProperties_mimeType="PNG",
        thumbnailProperties_thumbnailSize="LARGE",
    )
    api.presentations.get.assert_called_once_with(
        presentationId="pres1", fields="slides.objectId"
    )
    assert result["mime_type"] == "image/png"
    assert result["size"] == len(_PNG_BYTES)
    assert result["slide_id"] == "s1"
    assert result["page_number"] is None
    assert result["width"] == 1920
    assert result["height"] == 1080
    assert _result_path(result).read_bytes() == _PNG_BYTES
    assert len(http["calls"]) == 1


@pytest.mark.parametrize("size", ["LARGE", "MEDIUM", "SMALL"])
def test_thumbnail_passes_the_requested_image_size(monkeypatch, size):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="s1", size=size)

    assert result["status"] == "success"
    assert (
        api.get_thumbnail.call_args.kwargs["thumbnailProperties_thumbnailSize"] == size
    )


async def test_thumbnail_normalizes_size_case_and_whitespace(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = await _call_tool_via_mcp(
        {"presentation_id": "pres1", "slide_id": "s1", "size": " medium "}
    )

    assert result["status"] == "success"
    assert (
        api.get_thumbnail.call_args.kwargs["thumbnailProperties_thumbnailSize"]
        == "MEDIUM"
    )


async def _call_tool_via_mcp(arguments):
    tool = _thumbnail_tool()
    content, _result = await google_slides.mcp.call_tool(tool.__name__, arguments)
    return json.loads(content[0].text)


@pytest.mark.parametrize(
    ("page_number", "expected_object_id"),
    [(1, "s1"), (3, "s3"), (5, "s5")],
)
def test_thumbnail_resolves_a_one_based_page_number(
    monkeypatch, page_number, expected_object_id
):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", page_number=page_number)

    assert result["status"] == "success"
    assert result["page_number"] == page_number
    assert result["slide_id"] == expected_object_id
    assert api.get_thumbnail.call_args.kwargs["pageObjectId"] == expected_object_id
    api.presentations.get.assert_called_once_with(
        presentationId="pres1", fields="slides.objectId"
    )


async def test_thumbnail_uses_default_large_and_png_through_the_mcp_layer(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = await _call_tool_via_mcp({"presentation_id": "pres1", "slide_id": "s1"})

    assert result["status"] == "success"
    api.get_thumbnail.assert_called_once_with(
        presentationId="pres1",
        pageObjectId="s1",
        thumbnailProperties_mimeType="PNG",
        thumbnailProperties_thumbnailSize="LARGE",
    )


async def test_thumbnail_resolves_page_number_through_the_mcp_layer(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = await _call_tool_via_mcp({"presentation_id": "pres1", "page_number": 3})

    assert result["status"] == "success"
    assert api.get_thumbnail.call_args.kwargs["pageObjectId"] == "s3"
    api.presentations.get.assert_called_once_with(
        presentationId="pres1", fields="slides.objectId"
    )


def test_thumbnail_treats_numeric_looking_slide_id_as_an_object_id(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    api.set_slides("3")
    result = _call_tool(presentation_id="pres1", slide_id="3")

    assert result["status"] == "success"
    api.presentations.get.assert_called_once_with(
        presentationId="pres1", fields="slides.objectId"
    )
    assert api.get_thumbnail.call_args.kwargs["pageObjectId"] == "3"


def test_thumbnail_resolves_a_full_presentation_url(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(
        presentation_id=(
            "https://docs.google.com/presentation/d/presentation-Abc_123/edit#slide=id.p"
        ),
        slide_id="s1",
    )

    assert result["status"] == "success"
    assert result["presentation_id"] == "presentation-Abc_123"
    assert (
        api.get_thumbnail.call_args.kwargs["presentationId"] == "presentation-Abc_123"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"slide_id": "s1", "page_number": 2},
    ],
    ids=["neither", "both"],
)
def test_thumbnail_requires_exactly_one_slide_locator(monkeypatch, kwargs):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", **kwargs)

    assert result["status"] == "error"
    assert "exactly one" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize("page_number", [0, -1, 6, 1000])
def test_thumbnail_rejects_out_of_range_page_numbers(monkeypatch, page_number):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", page_number=page_number)

    assert result["status"] == "error"
    assert "page_number" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


def test_thumbnail_rejects_a_page_number_for_an_empty_deck(monkeypatch):
    api = _install_slides_api(monkeypatch)
    api.set_slides()
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", page_number=1)

    assert result["status"] == "error"
    assert "no slides" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize("slide_id", ["", "   "])
def test_thumbnail_rejects_blank_slide_ids(monkeypatch, slide_id):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id=slide_id)

    assert result["status"] == "error"
    assert "slide_id" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"presentation_id": "pres1", "page_number": True},
        {"presentation_id": "pres1", "page_number": "2"},
        {"presentation_id": "pres1", "page_number": 2.0},
        {"presentation_id": "pres1", "page_number": 2.5},
        {"presentation_id": "pres1", "page_number": [2]},
        {"presentation_id": "pres1", "page_number": {"n": 2}},
        {"presentation_id": "pres1", "slide_id": 3},
        {"presentation_id": "pres1", "slide_id": ["s1"]},
        {"presentation_id": "pres1", "slide_id": "s1", "size": "HUGE"},
        {"presentation_id": "pres1", "slide_id": "s1", "size": 3},
    ],
)
async def test_thumbnail_schema_validation_never_coerces_a_bad_locator(
    monkeypatch, arguments
):
    from mcp.server.fastmcp.exceptions import ToolError

    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    with pytest.raises(ToolError, match="validation error"):
        await google_slides.mcp.call_tool(
            "google_slides_get_slide_thumbnail", arguments
        )

    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


async def test_thumbnail_rejects_conflicting_locators_through_the_mcp_layer(
    monkeypatch,
):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = await _call_tool_via_mcp(
        {"presentation_id": "pres1", "slide_id": "s1", "page_number": 2}
    )

    assert result["status"] == "error"
    assert "exactly one" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


def test_thumbnail_writes_a_unique_file_per_download(monkeypatch):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    first = _call_tool(presentation_id="pres1", slide_id="s1")
    second = _call_tool(presentation_id="pres1", slide_id="s1")

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert first["path"] != second["path"]
    assert _result_path(first).read_bytes() == _PNG_BYTES
    assert _result_path(second).read_bytes() == _PNG_BYTES
    assert _result_path(first).parent == _result_path(second).parent


def test_thumbnail_keeps_the_written_file_inside_the_output_directory(
    monkeypatch, tmp_path
):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    google_slides.get_slides_service().presentations().get.return_value.execute.return_value = {
        "slides": [{"objectId": "../../escape"}]
    }
    result = _call_tool(presentation_id="pres1", slide_id="../../escape")

    assert result["status"] == "success"
    path = _result_path(result)
    assert path.is_file()
    assert path.parent == tmp_path / "output"
    assert ".." not in path.parts
    assert path.read_bytes() == _PNG_BYTES


def test_thumbnail_requires_a_task_workspace(monkeypatch):
    _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )
    monkeypatch.delenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR")

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "error"
    assert "task workspace" in result["message"]
    assert http["calls"] == []


@pytest.mark.parametrize(
    "content_url",
    [
        "http://lh3.googleusercontent.com/pages/thumb?token=t",
        "https://user:pass@lh3.googleusercontent.com/pages/thumb",
        "https://googleusercontent.com.evil.example/pages/thumb",
        "https://evilgoogleusercontent.com/pages/thumb",
        "https://evil.example/pages/thumb",
    ],
)
def test_thumbnail_rejects_a_disallowed_content_url_without_requesting_it(
    monkeypatch, content_url
):
    api = _install_slides_api(monkeypatch)
    api.set_thumbnail({"contentUrl": content_url, "width": 1, "height": 1})
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "error"
    assert http["calls"] == []


def test_thumbnail_follows_a_redirect_to_another_allowed_google_host(monkeypatch):
    _install_slides_api(monkeypatch)
    redirected_url = "https://lh7-rt.googleusercontent.com/pages/thumb?token=two"

    def responder(index, request, state):
        if index == 0:
            return _http_response(
                status=302,
                headers={"Location": redirected_url},
                url=request.url,
            )
        return _http_response(_PNG_BYTES, url=request.url)

    http = _install_http(monkeypatch, responder)

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "success"
    assert len(http["calls"]) == 2
    second_request = http["calls"][1][0]
    assert second_request.headers["host"] == "lh7-rt.googleusercontent.com"
    assert second_request.url.path == "/pages/thumb"


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example/pages/thumb",
        "http://lh3.googleusercontent.com/pages/thumb",
        "https://googleusercontent.com.evil.example/pages/thumb",
        "https://user:pass@lh3.googleusercontent.com/pages/thumb",
    ],
)
def test_thumbnail_rejects_a_redirect_target_that_fails_validation(
    monkeypatch, location
):
    _install_slides_api(monkeypatch)

    def responder(index, request, state):
        if index == 0:
            return _http_response(
                status=302, headers={"Location": location}, url=request.url
            )
        raise AssertionError(
            "a redirect target that fails validation must not be requested"
        )

    http = _install_http(monkeypatch, responder)

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "error"
    assert len(http["calls"]) == 1


@pytest.mark.parametrize("body", [b"not a png at all", b"<html>error</html>"])
def test_thumbnail_rejects_a_non_png_response(monkeypatch, tmp_path, body):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(body))

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "error"
    assert "PNG" in result["message"]
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_download_does_not_carry_credentials(monkeypatch):
    _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    first = _call_tool(presentation_id="pres1", slide_id="s1")
    second = _call_tool(presentation_id="pres1", page_number=2)
    assert first["status"] == "success"
    assert second["status"] == "success"

    for request, _kwargs in http["calls"]:
        header_names = {name.lower() for name in request.headers}
        assert "authorization" not in header_names
        assert "cookie" not in header_names
        assert "access-token" not in json.dumps(dict(request.headers))
    sessions = http["sessions"]
    for session in sessions:
        assert session.auth is None
        assert len(session.cookies) == 0


def test_thumbnail_never_returns_or_logs_the_short_lived_content_url(
    monkeypatch, caplog
):
    caplog.set_level(logging.INFO)
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "success"
    assert "thumb-secret-token" not in json.dumps(result)
    assert "contentUrl" not in json.dumps(result)
    assert "thumb-secret-token" not in caplog.text


@pytest.mark.parametrize("failure", [RuntimeError("api boom"), Exception("api boom")])
def test_thumbnail_api_failures_are_not_reported_as_success(monkeypatch, failure):
    api = _install_slides_api(monkeypatch)
    api.set_thumbnail(error=failure)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "error"
    assert http["calls"] == []


def test_thumbnail_missing_content_url_is_an_error(monkeypatch):
    api = _install_slides_api(monkeypatch)
    api.set_thumbnail({"width": 1920, "height": 1080})
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "error"
    assert http["calls"] == []


def test_thumbnail_page_lookup_failure_is_not_a_success(monkeypatch):
    api = _install_slides_api(monkeypatch)
    api.set_slides(error=RuntimeError("presentation get failed"))
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", page_number=2)

    assert result["status"] == "error"
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


async def test_thumbnail_tool_is_registered_with_its_limits_and_failure_contract():
    tools = {tool.name: tool for tool in await google_slides.mcp.list_tools()}
    tool = tools.get("google_slides_get_slide_thumbnail")
    assert tool is not None
    description = tool.description
    for phrase in (
        "exactly one of slide_id or page_number",
        "page_number",
        "1-based",
        "PNG",
        "LARGE",
        "task workspace",
        "file_ref",
        "error",
    ):
        assert phrase in description
    schema = tool.inputSchema
    assert set(schema["properties"]) == {
        "presentation_id",
        "slide_id",
        "page_number",
        "size",
    }
    assert schema["properties"]["size"]["default"] == "LARGE"
    assert schema["properties"]["size"]["enum"] == ["SMALL", "MEDIUM", "LARGE"]
    assert schema["required"] == ["presentation_id"]
    # Existing connector tools stay registered.
    assert "google_slides_get_presentation" in tools
    assert "google_slides_import_pptx" in tools


def test_thumbnail_does_not_follow_a_symlink_in_the_output_directory(
    monkeypatch, tmp_path
):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "outside.png"
    (output_dir / "slide_s1.png").symlink_to(outside)

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "success"
    path = _result_path(result)
    assert path.parent == output_dir
    assert path.name != "slide_s1.png"
    assert not outside.exists()
    assert path.read_bytes() == _PNG_BYTES


def test_thumbnail_forced_open_collision_preserves_existing_file(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "slide_s1.png"
    existing.write_bytes(b"existing output")
    original = os.open
    attempted = []

    def collide_once(path, flags, *args, **kwargs):
        if flags & os.O_EXCL:
            attempted.append(path)
            if len(attempted) == 1:
                raise FileExistsError("forced collision")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", collide_once)
    result = _call_tool(presentation_id="pres1", slide_id="s1")
    assert result["status"] == "success"
    assert len(attempted) == 2
    assert existing.read_bytes() == b"existing output"
    assert _result_path(result).read_bytes() == _PNG_BYTES
    assert _result_path(result) != existing
    assert len(list(output.iterdir())) == 2


@pytest.mark.parametrize(
    "body",
    [_PNG_SIGNATURE, _PNG_SIGNATURE + b"junk", _PNG_BYTES[:33]],
    ids=["signature_only", "signature_and_junk", "truncated"],
)
def test_thumbnail_rejects_png_like_bodies_that_cannot_be_decoded(
    monkeypatch, tmp_path, body
):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(body))

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "error"
    assert "PNG" in result["message"]
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_saves_a_decodable_png(monkeypatch):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "success"
    with Image.open(_result_path(result)) as image:
        image.load()
        assert image.format == "PNG"


def test_thumbnail_rejects_a_symlinked_output_directory(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (tmp_path / "output").symlink_to(outside, target_is_directory=True)

    result = _call_tool(presentation_id="pres1", slide_id="s1")

    assert result["status"] == "error"
    assert not list(outside.glob("*.png"))
