import json

import aiohttp
import pytest

from xagent.core.model import VideoModelConfig
from xagent.core.model.video.adapter import create_video_model, retry_on
from xagent.core.model.video.xinference import (
    XinferenceVideoModel,
    _size_from_resolution_ratio,
)


class FakeSession:
    pass


class FakeErrorResponse:
    status = 503
    history = ()
    headers = {}
    request_info = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def json(self):
        return {"error": "temporarily unavailable"}


class FakeErrorSession:
    def request(self, *args, **kwargs):
        return FakeErrorResponse()


class FakeClientSession:
    def __init__(self, *args, **kwargs):
        self.session = FakeSession()

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return None


class FakeVideoTransport:
    def __init__(self):
        self.text_calls = []
        self.image_calls = []
        self.flf_calls = []

    async def post_json(self, model, session, path, body):
        self.text_calls.append(
            {
                "session": session,
                "path": path,
                "body": body,
                "headers": model._headers(),
            }
        )
        return {"data": [{"url": "https://example.com/result.mp4"}]}

    async def post_form(self, model, session, path, fields, files):
        call = {
            "session": session,
            "path": path,
            "fields": fields,
            "files": files,
            "headers": model._headers(),
        }
        if path.endswith("/image"):
            self.image_calls.append(call)
            return {"data": [{"b64_json": "dmlkZW8="}]}
        self.flf_calls.append(call)
        return {"data": [{"url": "https://example.com/flf.mp4"}]}


@pytest.fixture
def fake_video_transport(monkeypatch):
    transport = FakeVideoTransport()

    async def post_json(model, session, path, body):
        return await transport.post_json(model, session, path, body)

    async def post_form(model, session, path, fields, files):
        return await transport.post_form(model, session, path, fields, files)

    monkeypatch.setattr(
        "xagent.core.model.video.xinference.aiohttp.ClientSession",
        FakeClientSession,
    )
    monkeypatch.setattr(
        XinferenceVideoModel,
        "_post_video_json",
        post_json,
    )
    monkeypatch.setattr(
        XinferenceVideoModel,
        "_post_video_form",
        post_form,
    )
    return transport


@pytest.mark.asyncio
async def test_xinference_text_to_video_uses_openai_compatible_params(
    fake_video_transport,
):
    model = XinferenceVideoModel(
        model_name="Wan2.1-1.3B",
        base_url="http://localhost:9997",
        api_key="token",
    )

    result = await model.generate_video(
        prompt="A camera push through a city",
        seconds=5,
        size="1280x720",
        n=2,
    )

    assert result["status"] == "succeeded"
    assert result["video_url"] == "https://example.com/result.mp4"
    assert fake_video_transport.text_calls == [
        {
            "session": fake_video_transport.text_calls[0]["session"],
            "path": "/v1/video/generations",
            "body": {
                "model": "Wan2.1-1.3B",
                "prompt": "A camera push through a city",
                "n": 2,
                "kwargs": json.dumps({"seconds": 5, "size": "1280x720"}),
            },
            "headers": {"Authorization": "Bearer token"},
        }
    ]


@pytest.mark.asyncio
async def test_xinference_generate_video_rejects_response_without_video_url(
    monkeypatch,
):
    async def post_json(model, session, path, body):
        _ = model, session, path, body
        return {"data": [{"id": "task-no-video"}]}

    monkeypatch.setattr(
        "xagent.core.model.video.xinference.aiohttp.ClientSession",
        FakeClientSession,
    )
    monkeypatch.setattr(XinferenceVideoModel, "_post_video_json", post_json)

    model = XinferenceVideoModel(
        model_name="Wan2.1-1.3B",
        base_url="http://localhost:9997",
    )

    with pytest.raises(RuntimeError, match="did not include"):
        await model.generate_video(prompt="A camera push through a city")


@pytest.mark.asyncio
async def test_xinference_image_to_video_accepts_reference_file(
    tmp_path, fake_video_transport
):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"image bytes")
    model = XinferenceVideoModel(
        model_name="Wan2.1-i2v-14B-480p",
        base_url="http://localhost:9997",
    )

    result = await model.generate_video(
        prompt="Animate this frame",
        input_reference=str(image_path),
        negative_prompt="blur",
        resolution="480p",
        ratio="9:16",
        allowed_local_media_roots=[tmp_path],
    )

    assert result["video_url"] == "data:video/mp4;base64,dmlkZW8="
    assert fake_video_transport.image_calls == [
        {
            "session": fake_video_transport.image_calls[0]["session"],
            "path": "/v1/video/generations/image",
            "fields": {
                "model": "Wan2.1-i2v-14B-480p",
                "prompt": "Animate this frame",
                "negative_prompt": "blur",
                "n": 1,
                "kwargs": json.dumps({"size": "480x853"}),
            },
            "files": {"image": b"image bytes"},
            "headers": {},
        }
    ]


@pytest.mark.asyncio
async def test_xinference_image_reference_accepts_file_uri_with_escaped_path(
    tmp_path,
):
    image_path = tmp_path / "frame with space.png"
    image_path.write_bytes(b"image bytes")
    model = XinferenceVideoModel(
        model_name="Wan2.1-i2v-14B-480p",
        base_url="http://localhost:9997",
    )

    data = await model._read_media_bytes(
        image_path.as_uri(),
        allowed_local_media_roots=[tmp_path],
    )

    assert data == b"image bytes"


@pytest.mark.asyncio
async def test_xinference_image_reference_resolves_relative_path_against_allowed_root(
    tmp_path,
):
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"image bytes")
    model = XinferenceVideoModel(
        model_name="Wan2.1-i2v-14B-480p",
        base_url="http://localhost:9997",
    )

    data = await model._read_media_bytes(
        "frame.png",
        allowed_local_media_roots=[tmp_path],
    )

    assert data == b"image bytes"


def test_xinference_size_from_resolution_ratio_rejects_zero_ratio():
    assert _size_from_resolution_ratio("480p", "0:16") is None
    assert _size_from_resolution_ratio("480p", "16:0") is None


def test_video_retry_filter_only_retries_transient_errors():
    def wrap(exc):
        try:
            raise RuntimeError("wrapped provider error") from exc
        except RuntimeError as wrapped:
            return wrapped

    assert retry_on(aiohttp.ServerTimeoutError()) is True
    assert retry_on(wrap(aiohttp.ServerTimeoutError())) is True
    assert (
        retry_on(
            wrap(
                aiohttp.ClientResponseError(
                    request_info=None,
                    history=(),
                    status=429,
                    message="Too many requests",
                    headers=None,
                )
            )
        )
        is True
    )
    assert retry_on(TimeoutError("poll timed out")) is False
    assert retry_on(RuntimeError("terminal task failure")) is False
    assert retry_on(ValueError("invalid request")) is False


@pytest.mark.asyncio
async def test_xinference_request_json_preserves_http_status_for_retry():
    model = XinferenceVideoModel(model_name="Wan2.1-1.3B")

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await model._request_json(FakeErrorSession(), "POST", "/v1/video/generations")

    assert exc_info.value.status == 503
    assert retry_on(RuntimeError("wrapped")) is False
    assert retry_on(exc_info.value) is True


@pytest.mark.asyncio
async def test_xinference_generate_video_uses_effective_timeout_without_mutating_model(
    monkeypatch,
    fake_video_transport,
):
    read_timeouts = []
    model = XinferenceVideoModel(
        model_name="Wan2.1-i2v-14B-480p",
        base_url="http://localhost:9997",
        timeout=120.0,
    )

    async def read_media_bytes(
        self, media, allowed_local_media_roots=None, timeout=None
    ):
        _ = self, media, allowed_local_media_roots
        read_timeouts.append(timeout)
        return b"image bytes"

    monkeypatch.setattr(XinferenceVideoModel, "_read_media_bytes", read_media_bytes)

    await model.generate_video(
        prompt="Animate this frame",
        input_reference="https://example.com/frame.png",
        timeout=7.0,
    )

    assert model.timeout == 120.0
    assert read_timeouts == [7.0]
    assert fake_video_transport.image_calls


@pytest.mark.asyncio
async def test_xinference_image_reference_rejects_local_file_outside_allowlist(
    tmp_path, fake_video_transport
):
    allowed_root = tmp_path / "workspace"
    allowed_root.mkdir()
    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(b"image bytes")
    model = XinferenceVideoModel(
        model_name="Wan2.1-i2v-14B-480p",
        base_url="http://localhost:9997",
    )

    with pytest.raises(RuntimeError, match="Access denied"):
        await model.generate_video(
            prompt="Animate this frame",
            input_reference=str(outside_image),
            allowed_local_media_roots=[allowed_root],
        )

    assert fake_video_transport.image_calls == []


def test_video_adapter_routes_to_xinference():
    config = VideoModelConfig(
        id="wan",
        model_name="Wan2.1-1.3B",
        model_provider="xinference",
        base_url="http://localhost:9997",
        abilities=["generate"],
    )

    model = create_video_model(config)

    assert isinstance(getattr(model, "_inner", model), XinferenceVideoModel)
