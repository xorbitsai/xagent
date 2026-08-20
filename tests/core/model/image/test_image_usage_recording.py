"""Image providers record media usage into the active token context."""

import pytest

from xagent.core.model.chat.token_context import (
    TokenContextManager,
    aggregate_media_usage_by_model,
)
from xagent.core.model.image.usage import record_image_usage


def test_record_image_usage_gemini_style_tokens() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"image_url": "x", "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
            model_name="gemini-image",
            call_type="generate_image",
            resolution="2K",
        )
        usage = manager.get_usage()

    assert usage.media_calls == 1
    entry = usage.details[0]
    assert entry["type"] == "media"
    assert entry["unit"] == "images"
    assert entry["quantity"] == 1.0
    assert entry["provider_tokens"] == 8
    # Provider-reported, not a local estimate — billing may price these.
    assert entry["tokens_estimated"] is False
    assert entry["call_type"] == "generate_image"


def test_record_image_usage_honours_n_and_model_id() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {}},
            model_name="dall-e",
            model_id="img-1",
            image_count=4,
            resolution="1024x1024",
        )
        entry = manager.get_usage().details[0]

    # n>1 must bill 4 images, not 1.
    assert entry["quantity"] == 4.0
    assert entry["model_id"] == "img-1"
    assert entry["resolution"] == "1024x1024"


def test_record_image_usage_records_resolution_tier() -> None:
    # Resolution tier is recorded so cloud can price by (model, resolution),
    # while the real image tokens let a token-based price take precedence.
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": 5, "completion_tokens": 3}},
            model_name="gemini-image",
            call_type="generate_image",
            resolution="2K",
        )
        entry = manager.get_usage().details[0]

    assert entry["resolution"] == "2K"
    assert entry["provider_input_tokens"] == 5
    assert entry["provider_output_tokens"] == 3


def test_record_image_usage_empty_or_missing_usage() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"image_url": "x", "usage": {}},
            model_name="sdxl",
            call_type="edit_image",
        )
        record_image_usage({"image_url": "x"}, model_name="foo")
        usage = manager.get_usage()

    assert usage.media_calls == 2
    assert all(entry["provider_tokens"] == 0 for entry in usage.details)


def test_record_image_usage_never_raises_on_garbage() -> None:
    with TokenContextManager() as manager:
        # Not a dict / no usage / None usage must all be tolerated.
        record_image_usage({}, model_name="m")  # type: ignore[arg-type]
        record_image_usage({"usage": None}, model_name="m")
        usage = manager.get_usage()

    assert usage.media_calls == 2


def test_image_usage_shows_in_media_aggregation() -> None:
    with TokenContextManager() as manager:
        record_image_usage({"usage": {}}, model_name="sd", call_type="generate_image")
        record_image_usage({"usage": {}}, model_name="sd", call_type="generate_image")
        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    assert len(groups) == 1
    assert groups[0]["model_name"] == "sd"
    assert groups[0]["unit"] == "images"
    assert groups[0]["quantity"] == 2.0
    assert groups[0]["calls"] == 2


class _FakeResponse:
    """Minimal stand-in for an httpx/aiohttp 200 response."""

    status_code = 200
    status = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


# A 200 whose usageMetadata is real but whose candidate carries no image:
# Google has billed this, and retry_on only matches 429/5xx so it is final.
_GEMINI_SAFETY_BLOCKED = {
    "candidates": [{"finishReason": "SAFETY", "content": {"parts": [{"text": "no"}]}}],
    "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 5},
}


@pytest.mark.asyncio
async def test_gemini_meters_billed_response_with_no_image(monkeypatch) -> None:
    from xagent.core.model.image import gemini as gemini_mod

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **kw):
            return _FakeResponse(_GEMINI_SAFETY_BLOCKED)

    monkeypatch.setattr(gemini_mod.httpx, "AsyncClient", lambda **kw: _Client())
    model = gemini_mod.GeminiImageModel(model_name="gemini-image", api_key="k")

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await model.generate_image(prompt="p", n=3)
        usage = manager.get_usage()

    # The call was billed by the provider, so it must be metered even though
    # the response body failed validation and the caller saw an error.
    assert usage.media_calls == 1
    entry = usage.details[0]
    assert entry["unit"] == "images"
    # 1, not 3: Gemini has no multi-image parameter and this client drops `n`
    # before the request, so billing 3 would charge for images that were never
    # requested of the provider or returned by it.
    assert entry["quantity"] == 1
    assert entry["provider_tokens"] == 16


@pytest.mark.asyncio
async def test_dashscope_meters_billed_unparseable_response(monkeypatch) -> None:
    from xagent.core.model.image import dashscope as ds_mod

    payload = {"usage": {"input_tokens": 7}}  # 200, billed, but no `output`

    class _Post:
        async def __aenter__(self):
            return _AsyncResp()

        async def __aexit__(self, *exc):
            return False

    class _AsyncResp:
        status = 200

        async def json(self):
            return payload

        async def text(self):
            return ""

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, *a, **kw):
            return _Post()

    monkeypatch.setattr(ds_mod.aiohttp, "ClientSession", lambda **kw: _Session())
    model = ds_mod.DashScopeImageModel(model_name="wanx", api_key="k")

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await model.generate_image(prompt="p", n=4)
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["unit"] == "images"
    assert usage.details[0]["quantity"] == 4


@pytest.mark.asyncio
async def test_gemini_edit_does_not_bill_unsupported_n(monkeypatch) -> None:
    """Gemini must never bill `n`: it cannot honour it.

    `edit_image` builds `generationConfig` with only `imageConfig`, so a
    caller-supplied `n` is dropped before the request goes out, and the parser
    returns a single image. Billing `n` would charge for images the provider
    was never asked to make and never returned.
    """
    from xagent.core.model.image import gemini as gemini_mod

    calls = []
    sent = {}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **kw):
            sent.update(kw.get("json", {}))
            return _FakeResponse(
                {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {
                                "parts": [
                                    {"text": "![Image](https://example.com/edit.png)"}
                                ]
                            },
                        }
                    ],
                    "usageMetadata": {},
                }
            )

    monkeypatch.setattr(gemini_mod.httpx, "AsyncClient", lambda **kw: _Client())
    monkeypatch.setattr(
        gemini_mod,
        "record_image_usage",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    model = gemini_mod.GeminiImageModel(
        model_name="gemini-3-pro-image-preview-2k",
        api_key="k",
        abilities=["generate", "edit"],
    )

    await model.edit_image(
        image_url="data:image/png;base64,iVBORw0KGgo=",
        prompt="edit",
        n=3,
    )

    # n was dropped on the way to the provider...
    assert "n" not in sent.get("generationConfig", {})
    # ...so image_count is not passed and defaults to 1 rather than billing 3.
    assert "image_count" not in calls[0]


@pytest.mark.asyncio
async def test_dashscope_edit_forwards_image_count(monkeypatch) -> None:
    from xagent.core.model.image import dashscope as ds_mod

    calls = []
    payload = {
        "usage": {},
        "output": {
            "choices": [
                {"message": {"content": [{"image": "https://example.com/edit.png"}]}}
            ]
        },
    }

    class _Post:
        async def __aenter__(self):
            return _AsyncResp()

        async def __aexit__(self, *exc):
            return False

    class _AsyncResp:
        status = 200

        async def json(self):
            return payload

        async def text(self):
            return ""

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, *a, **kw):
            return _Post()

    monkeypatch.setattr(ds_mod.aiohttp, "ClientSession", lambda **kw: _Session())
    monkeypatch.setattr(
        ds_mod,
        "record_image_usage",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    model = ds_mod.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"]
    )

    await model.edit_image(
        image_url="https://example.com/source.png",
        prompt="edit",
        n=4,
    )

    assert calls[0]["image_count"] == 4


@pytest.mark.asyncio
async def test_openai_edit_closes_files_when_later_open_fails(monkeypatch) -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    class _OpenedFile:
        closed = False

        def close(self):
            self.closed = True

    opened_file = _OpenedFile()

    def fake_open(path, mode):
        if path == "missing.png":
            raise FileNotFoundError(path)
        return opened_file

    model = OpenAIImageModel(api_key="k")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    monkeypatch.setattr(model, "_client", object())
    monkeypatch.setattr("builtins.open", fake_open)

    with pytest.raises(FileNotFoundError):
        await model.edit_image(
            image_url=["first.png", "missing.png"],
            prompt="edit",
        )

    assert opened_file.closed is True


@pytest.mark.asyncio
async def test_gemini_does_not_bill_unsupported_n(monkeypatch) -> None:
    # The Gemini API has no multi-image parameter, and this client forwards only
    # `temperature` out of **kwargs, so a caller-supplied n never reaches the
    # provider — one image is requested, generated, and parsed. Billing n would
    # charge for images that do not exist.
    from xagent.core.model.image import gemini as gemini_mod

    payload = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [{"inlineData": {"mimeType": "image/png", "data": "eA=="}}]
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1},
    }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **kw):
            # n must not have been forwarded to the provider.
            assert "n" not in kw.get("json", {}).get("generationConfig", {})
            return _FakeResponse(payload)

    monkeypatch.setattr(gemini_mod.httpx, "AsyncClient", lambda **kw: _Client())
    model = gemini_mod.GeminiImageModel(model_name="gemini-image", api_key="k")

    with TokenContextManager() as manager:
        await model.generate_image(prompt="p", n=4)
        entries = [d for d in manager.get_usage().details if d.get("type") == "media"]

    assert len(entries) == 1
    assert entries[0]["quantity"] == 1.0  # not 4.0


@pytest.mark.asyncio
async def test_dashscope_bills_n_it_actually_forwards(monkeypatch) -> None:
    # DashScope spreads **kwargs into the request `parameters`, so n does reach
    # the provider and is billed by it. The recorded count must match.
    from xagent.core.model.image import dashscope as ds_mod

    seen: dict = {}
    payload = {
        "usage": {},
        "output": {
            "choices": [{"message": {"content": [{"image": "https://x/1.png"}]}}]
        },
    }

    class _AsyncResp:
        status = 200

        async def json(self):
            return payload

        async def text(self):
            return ""

    class _Post:
        async def __aenter__(self):
            return _AsyncResp()

        async def __aexit__(self, *exc):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, *a, **kw):
            seen.update(kw.get("json", {}))
            return _Post()

    monkeypatch.setattr(ds_mod.aiohttp, "ClientSession", lambda **kw: _Session())
    model = ds_mod.DashScopeImageModel(model_name="wanx", api_key="k")

    with TokenContextManager() as manager:
        await model.generate_image(prompt="p", n=3)
        entries = [d for d in manager.get_usage().details if d.get("type") == "media"]

    # n really was sent to the provider...
    assert seen["parameters"]["n"] == 3
    # ...so the billed quantity matches the provider's own invoice.
    assert entries[0]["quantity"] == 3.0
