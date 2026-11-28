import json
from unittest.mock import Mock

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image
from requests import Response

from xagent.web.tools.mcp import linkedin


class MockResponse:
    def __init__(self, json_data=None, headers=None, text="", status_code=200):
        self._json_data = json_data or {}
        self.headers = headers or {}
        self.text = text
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text or f"HTTP {self.status_code}")


def test_sanitize_post_text_escapes_ascii_parentheses():
    text = "🤖 Claude (Anthropic) — Prompt-to-prototype thinking"

    assert (
        linkedin._sanitize_post_text(text)
        == "🤖 Claude \\(Anthropic\\) — Prompt-to-prototype thinking"
    )


@pytest.mark.asyncio
async def test_create_post_schema_accepts_optional_image_path():
    tools = await linkedin.list_tools()
    create_post = next(tool for tool in tools if tool.name == "create_post")

    properties = create_post.inputSchema["properties"]
    assert create_post.inputSchema["required"] == ["text"]
    assert "image_path" in properties
    assert "altText" in properties
    assert "generated image" in create_post.description


@pytest.mark.asyncio
async def test_create_post_without_image_keeps_text_only_flow(monkeypatch):
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "token")
    mock_get = Mock(
        return_value=MockResponse(
            json_data={"sub": "person-1"}, text='{"sub":"person-1"}'
        )
    )
    mock_post = Mock(
        return_value=MockResponse(headers={"x-restli-id": "urn:li:share:post-1"})
    )
    mock_put = Mock()

    monkeypatch.setattr(linkedin.requests, "get", mock_get)
    monkeypatch.setattr(linkedin.requests, "post", mock_post)
    monkeypatch.setattr(linkedin.requests, "put", mock_put)

    result = await linkedin.call_tool("create_post", {"text": "hello"})

    assert result[0].text == "Post created successfully! URN: urn:li:share:post-1"
    mock_put.assert_not_called()
    assert mock_post.call_count == 1
    post_body = mock_post.call_args.kwargs["json"]
    assert post_body["commentary"] == "hello"
    assert "content" not in post_body


@pytest.mark.asyncio
async def test_create_article_post_escapes_parentheses_in_all_user_facing_fields(
    monkeypatch,
):
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "token")
    mock_get = Mock(
        return_value=MockResponse(
            json_data={"sub": "person-1"}, text='{"sub":"person-1"}'
        )
    )
    mock_post = Mock(
        return_value=MockResponse(headers={"x-restli-id": "urn:li:share:post-1"})
    )

    monkeypatch.setattr(linkedin.requests, "get", mock_get)
    monkeypatch.setattr(linkedin.requests, "post", mock_post)

    result = await linkedin.call_tool(
        "create_article_post",
        {
            "text": "Launch (recap)",
            "articleUrl": "https://example.com/post",
            "articleTitle": "Claude (Anthropic)",
            "articleDescription": "Notes (draft)",
        },
    )

    assert (
        result[0].text == "Article post created successfully! URN: urn:li:share:post-1"
    )
    post_body = mock_post.call_args.kwargs["json"]
    assert post_body["commentary"] == "Launch \\(recap\\)"
    assert post_body["content"]["article"] == {
        "source": "https://example.com/post",
        "title": "Claude \\(Anthropic\\)",
        "description": "Notes \\(draft\\)",
    }


@pytest.mark.asyncio
async def test_create_post_with_image_uploads_and_attaches_media(monkeypatch, tmp_path):
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "token")
    monkeypatch.setenv("XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS", str(tmp_path))
    image_path = tmp_path / "generated.png"
    Image.new("RGB", (2, 2)).save(image_path)

    mock_get = Mock(
        return_value=MockResponse(
            json_data={"sub": "person-1"}, text='{"sub":"person-1"}'
        )
    )

    def mock_post(url, **kwargs):
        if url == linkedin.IMAGES_URL:
            assert kwargs["params"] == {"action": "initializeUpload"}
            assert kwargs["json"] == {
                "initializeUploadRequest": {
                    "owner": "urn:li:person:person-1",
                }
            }
            return MockResponse(
                json_data={
                    "value": {
                        "uploadUrl": "https://upload.linkedin.example/image",
                        "image": "urn:li:image:image-1",
                    }
                }
            )
        if url == linkedin.POSTS_URL:
            return MockResponse(headers={"x-restli-id": "post-1"})
        raise AssertionError(f"Unexpected POST URL: {url}")

    mock_post_fn = Mock(side_effect=mock_post)

    def upload_bytes(url, **kwargs):
        assert kwargs["data"].read() == image_path.read_bytes()
        return MockResponse()

    mock_put = Mock(side_effect=upload_bytes)

    monkeypatch.setattr(linkedin.requests, "get", mock_get)
    monkeypatch.setattr(linkedin.requests, "post", mock_post_fn)
    monkeypatch.setattr(linkedin.requests, "put", mock_put)

    result = await linkedin.call_tool(
        "create_post",
        {
            "text": "hello with image",
            "image_path": str(image_path),
            "altText": "Generated AI visual",
        },
    )

    assert result[0].text == "Post created successfully! URN: urn:li:share:post-1"
    assert mock_post_fn.call_count == 2
    mock_put.assert_called_once()

    put_kwargs = mock_put.call_args.kwargs
    assert mock_put.call_args.args[0] == "https://upload.linkedin.example/image"
    assert put_kwargs["headers"]["Authorization"] == "Bearer token"
    assert put_kwargs["headers"]["Content-Type"] == "image/png"

    final_post_body = mock_post_fn.call_args.kwargs["json"]
    assert final_post_body["commentary"] == "hello with image"
    assert final_post_body["content"] == {
        "media": {
            "id": "urn:li:image:image-1",
            "altText": "Generated AI visual",
        }
    }


@pytest.mark.asyncio
async def test_create_post_rejects_image_outside_allowed_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "token")
    allowed_dir = tmp_path / "allowed"
    blocked_dir = tmp_path / "blocked"
    allowed_dir.mkdir()
    blocked_dir.mkdir()
    image_path = blocked_dir / "secret.png"
    image_path.write_bytes(b"png-bytes")
    monkeypatch.setenv("XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS", str(allowed_dir))

    mock_get = Mock(
        return_value=MockResponse(
            json_data={"sub": "person-1"}, text='{"sub":"person-1"}'
        )
    )
    mock_post = Mock(return_value=MockResponse())
    mock_put = Mock(return_value=MockResponse())

    monkeypatch.setattr(linkedin.requests, "get", mock_get)
    monkeypatch.setattr(linkedin.requests, "post", mock_post)
    monkeypatch.setattr(linkedin.requests, "put", mock_put)

    result = await _call_over_mcp(
        "create_post", {"text": "hello with image", "image_path": str(image_path)}
    )

    assert result.isError
    assert "outside allowed directories" in result.content[0].text
    assert str(image_path) in result.content[0].text
    assert str(allowed_dir) not in result.content[0].text
    mock_post.assert_not_called()
    mock_put.assert_not_called()


@pytest.mark.asyncio
async def test_create_comment_escapes_parentheses_before_publishing(monkeypatch):
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "token")
    mock_get = Mock(
        return_value=MockResponse(
            json_data={"sub": "person-1"}, text='{"sub":"person-1"}'
        )
    )
    mock_post = Mock(return_value=MockResponse(headers={"x-restli-id": "comment-1"}))

    monkeypatch.setattr(linkedin.requests, "get", mock_get)
    monkeypatch.setattr(linkedin.requests, "post", mock_post)

    result = await linkedin.call_tool(
        "create_comment",
        {
            "post_urn": "urn:li:share:post-1",
            "text": "Nice work (team)!",
        },
    )

    assert result[0].text == "Comment created successfully! ID: comment-1"
    comment_body = mock_post.call_args.kwargs["json"]
    assert comment_body["message"]["text"] == "Nice work \\(team\\)!"


async def _call_over_mcp(name, arguments):
    async with create_connected_server_and_client_session(linkedin.app) as session:
        return await session.call_tool(name, arguments)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("disguised", "valid JPEG, PNG, or GIF"),
        ("truncated", "valid JPEG, PNG, or GIF"),
        ("pixels", "fewer than 36,152,320 pixels"),
        ("frames", "at most 250 frames"),
    ],
)
async def test_invalid_image_does_not_initialize_upload(
    monkeypatch, tmp_path, case, message
):
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "token")
    monkeypatch.setenv("XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setattr(
        linkedin.requests, "get", Mock(return_value=MockResponse({"sub": "person-1"}))
    )
    post = Mock()
    put = Mock()
    monkeypatch.setattr(linkedin.requests, "post", post)
    monkeypatch.setattr(linkedin.requests, "put", put)

    image_path = tmp_path / "image.png"
    if case == "disguised":
        Image.new("RGB", (2, 2)).save(image_path, format="BMP")
    elif case == "truncated":
        Image.new("RGB", (32, 32)).save(image_path, format="JPEG")
        image_path.write_bytes(image_path.read_bytes()[:-8])
    elif case == "pixels":
        monkeypatch.setattr(linkedin, "_IMAGE_PIXEL_LIMIT", 16)
        Image.new("RGB", (4, 4)).save(image_path)
    elif case == "frames":
        frames = [Image.new("RGB", (2, 2), (i, 0, 0)) for i in range(251)]
        frames[0].save(
            image_path, format="GIF", save_all=True, append_images=frames[1:]
        )

    result = await _call_over_mcp(
        "create_post", {"text": "hello", "image_path": str(image_path)}
    )

    assert result.isError
    assert message in result.content[0].text
    post.assert_not_called()
    put.assert_not_called()


def test_validate_image_accepts_gif_at_frame_limit(tmp_path):
    image_path = tmp_path / "animation.gif"
    frames = [Image.new("RGB", (2, 2), (i, 0, 0)) for i in range(250)]
    frames[0].save(image_path, save_all=True, append_images=frames[1:])

    with image_path.open("rb") as image_file:
        assert linkedin._validate_image(image_file) == "image/gif"
        assert image_file.tell() == 0


def _http_error(status):
    response = Response()
    response.status_code = status
    return linkedin.requests.HTTPError("raw LinkedIn response", response=response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "exc", "expected_error"),
    [
        ("author", linkedin.requests.Timeout(), True),
        ("publish", _http_error(422), True),
        ("publish", linkedin.requests.Timeout(), False),
        ("publish", _http_error(503), False),
    ],
)
async def test_create_post_failure_classification(
    monkeypatch, stage, exc, expected_error
):
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "token")
    get = Mock(return_value=MockResponse({"sub": "person-1"}))
    post = Mock()
    (get if stage == "author" else post).side_effect = exc
    monkeypatch.setattr(linkedin.requests, "get", get)
    monkeypatch.setattr(linkedin.requests, "post", post)

    result = await _call_over_mcp("create_post", {"text": "hello"})

    assert result.isError is expected_error
    text = result.content[0].text
    assert "raw LinkedIn response" not in text
    if not expected_error:
        payload = json.loads(text)
        assert payload["status"] == "indeterminate"
        assert payload["retryable"] is False
        assert payload["mutation_may_have_completed"] is True
        assert "Check LinkedIn before retrying" in payload["message"]
    assert post.call_count == int(stage == "publish")


@pytest.mark.asyncio
async def test_image_upload_failure_prevents_publishing(monkeypatch, tmp_path):
    monkeypatch.setenv("LINKEDIN_ACCESS_TOKEN", "token")
    monkeypatch.setenv("XAGENT_LINKEDIN_IMAGE_ALLOWED_DIRS", str(tmp_path))
    image_path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image_path)
    monkeypatch.setattr(
        linkedin.requests, "get", Mock(return_value=MockResponse({"sub": "person-1"}))
    )
    post = Mock(
        return_value=MockResponse(
            {
                "value": {
                    "uploadUrl": "https://upload.linkedin.example/image",
                    "image": "urn:li:image:1",
                }
            }
        )
    )
    monkeypatch.setattr(linkedin.requests, "post", post)
    monkeypatch.setattr(
        linkedin.requests, "put", Mock(side_effect=linkedin.requests.Timeout())
    )

    result = await _call_over_mcp(
        "create_post", {"text": "hello", "image_path": str(image_path)}
    )

    assert result.isError
    assert "timed out" in result.content[0].text
    assert [call.args[0] for call in post.call_args_list] == [linkedin.IMAGES_URL]
