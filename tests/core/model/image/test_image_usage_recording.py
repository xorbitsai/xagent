"""Image providers record media usage into the active token context."""

import decimal
import fractions
import math

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
    # Resolution tier is retained so downstream grouping can separate a model's
    # resolutions into distinct billable line items, and provider tokens ride
    # along as raw metadata on the same row. Neither is a pricing rule: the row
    # carries no price-basis discriminator and the aggregate groups purely by
    # (model, unit, call_type, resolution). Token-vs-resolution precedence is
    # tracked in #1461.
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
# Google has billed this, and the production retry policy refuses to retry the
# InvalidImageResponseError it raises, so the charge and the row are final.
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


# ---------------------------------------------------------------------------
# Raw provider values must reach the shared write boundary uncoerced.
#
# Pre-coercing here defeated guards that exist one layer down: int(True) is 1,
# so a JSON boolean billed a token the boundary deliberately reads as 0, and
# int(float("inf")) raises OverflowError inside record_image_usage's own
# best-effort handler -- losing the whole billable row to salvage one bad field.
# ---------------------------------------------------------------------------


def test_boolean_provider_tokens_are_not_billed_as_one() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": True, "completion_tokens": True}},
            model_name="m",
        )
        entry = manager.get_usage().details[0]

    # 0, not 1: a provider JSON boolean is malformed metadata, not a count.
    assert entry["provider_input_tokens"] == 0
    assert entry["provider_output_tokens"] == 0
    assert entry["provider_tokens"] == 0


def test_non_finite_provider_tokens_keep_the_billable_row() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": float("inf")}},
            model_name="m",
            image_count=2,
        )
        usage = manager.get_usage()

    # The row survives: the call was billed, so losing it to one bad token
    # field would silently drop real usage.
    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 2.0
    assert usage.details[0]["provider_tokens"] == 0


def test_overflowing_provider_tokens_keep_the_billable_row() -> None:
    # json.loads of a 400-digit integer literal yields exactly this.
    with TokenContextManager() as manager:
        record_image_usage({"usage": {"prompt_tokens": 10**400}}, model_name="m")
        usage = manager.get_usage()

    assert usage.media_calls == 1


@pytest.mark.parametrize("bad_count", [float("inf"), float("nan"), -3, "many", None])
def test_unusable_image_count_keeps_the_row(bad_count: object) -> None:
    with TokenContextManager() as manager:
        record_image_usage({"usage": {}}, model_name="m", image_count=bad_count)
        usage = manager.get_usage()

    # A zero-quantity row is the boundary's "billed but unmeasured" convention,
    # and is strictly better than no row at all.
    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 0.0


def test_boolean_image_count_is_not_billed_as_one() -> None:
    with TokenContextManager() as manager:
        record_image_usage({"usage": {}}, model_name="m", image_count=True)
        entry = manager.get_usage().details[0]

    assert entry["quantity"] == 0.0


def test_omitted_image_count_bills_one() -> None:
    with TokenContextManager() as manager:
        record_image_usage({"usage": {}}, model_name="m")
        entry = manager.get_usage().details[0]

    # Absent `n` is the "provider neither accepts nor reports a count" case.
    assert entry["quantity"] == 1.0


def test_record_image_usage_tolerates_non_dict_result() -> None:
    with TokenContextManager() as manager:
        # A list, not {}: `{}` takes the same isinstance(dict) branch as a real
        # payload, so it never exercised the defensive non-dict path.
        record_image_usage(["not", "a", "dict"], model_name="m")  # type: ignore[arg-type]
        record_image_usage("nope", model_name="m")  # type: ignore[arg-type]
        record_image_usage(None, model_name="m")  # type: ignore[arg-type]
        usage = manager.get_usage()

    assert usage.media_calls == 3
    assert all(entry["provider_tokens"] == 0 for entry in usage.details)


def test_usage_payload_whose_attribute_access_raises() -> None:
    class _Hostile:
        @property
        def prompt_tokens(self) -> int:
            raise RuntimeError("boom")

    with TokenContextManager() as manager:
        record_image_usage({"usage": _Hostile()}, model_name="m")
        usage = manager.get_usage()

    # Reading a field must not cost the row.
    assert usage.media_calls == 1


# ---------------------------------------------------------------------------
# Retry policy: a billed invalid 200 must be charged and metered exactly once.
#
# These go through the production `create_image_model` wrapper, not a bare
# provider: the defect lived in the wrapper's default retry predicate
# (`lambda _: True`), so a provider-only test could not see it.
# ---------------------------------------------------------------------------


def _image_config(provider: str, *, model_id: str = "cfg-1", max_retries: int = 4):
    from xagent.core.model import ImageModelConfig

    return ImageModelConfig(
        id=model_id,
        model_name=f"{provider}-image",
        model_provider=provider,
        api_key="k",
        base_url=None,
        timeout=5.0,
        abilities=["generate", "edit"],
        max_retries=max_retries,
    )


@pytest.fixture
def instant_retries(monkeypatch):
    """Zero the backoff so a retry-count assertion does not sleep."""
    from xagent.core.retry.strategy import ExponentialBackoff

    monkeypatch.setattr(ExponentialBackoff, "get_delay", lambda self, attempt: 0.0)


def _gemini_client_factory(payload: dict, attempts: dict):
    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *a, **kw):
            attempts["n"] += 1
            return _FakeResponse(payload)

    return lambda **kw: _Client()


@pytest.mark.asyncio
async def test_billed_invalid_gemini_response_is_charged_once(
    monkeypatch, instant_retries
) -> None:
    from xagent.core.model.image import gemini as gemini_mod
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.image.base import InvalidImageResponseError

    attempts = {"n": 0}
    monkeypatch.setattr(
        gemini_mod.httpx,
        "AsyncClient",
        _gemini_client_factory(_GEMINI_SAFETY_BLOCKED, attempts),
    )
    model = create_image_model(_image_config("gemini"))

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await model.generate_image(prompt="p")
        usage = manager.get_usage()

    # One provider call and one billing row. Retrying a safety block cannot
    # change the outcome, so each retry only bought another charge and another
    # row -- with max_retries=4 this was 4 and 4.
    assert attempts["n"] == 1
    assert usage.media_calls == 1


@pytest.mark.asyncio
async def test_billed_invalid_gemini_edit_response_is_charged_once(
    monkeypatch, instant_retries
) -> None:
    from xagent.core.model.image import gemini as gemini_mod
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.image.base import InvalidImageResponseError

    attempts = {"n": 0}
    monkeypatch.setattr(
        gemini_mod.httpx,
        "AsyncClient",
        _gemini_client_factory(_GEMINI_SAFETY_BLOCKED, attempts),
    )
    model = create_image_model(_image_config("gemini"))

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await model.edit_image(
                image_url="data:image/png;base64,iVBORw0KGgo=", prompt="edit"
            )
        usage = manager.get_usage()

    assert attempts["n"] == 1
    assert usage.media_calls == 1
    assert usage.details[0]["call_type"] == "edit_image"


@pytest.mark.asyncio
async def test_transient_gemini_failures_are_still_retried(
    monkeypatch, instant_retries
) -> None:
    """The narrow exclusion must not stop retrying real transient failures.

    Gemini flattens timeouts into a plain RuntimeError, so a predicate that
    only allowed named transient types would silently stop retrying them.
    """
    from xagent.core.model.image import gemini as gemini_mod
    from xagent.core.model.image.adapter import create_image_model

    attempts = {"n": 0}
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
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise gemini_mod.httpx.TimeoutException("transient")
            return _FakeResponse(payload)

    monkeypatch.setattr(gemini_mod.httpx, "AsyncClient", lambda **kw: _Client())
    model = create_image_model(_image_config("gemini"))

    with TokenContextManager() as manager:
        result = await model.generate_image(prompt="p")
        usage = manager.get_usage()

    assert result["image_url"]
    assert attempts["n"] == 3
    # Only the attempt that reached a 200 was billed, so only it is metered.
    assert usage.media_calls == 1


def _dashscope_session_factory(payload: dict, attempts: dict, status: int = 200):
    class _Resp:
        def __init__(self) -> None:
            self.status = status

        async def json(self):
            return payload

        async def text(self):
            return ""

    class _Post:
        async def __aenter__(self):
            attempts["n"] += 1
            return _Resp()

        async def __aexit__(self, *exc):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, *a, **kw):
            return _Post()

    return lambda **kw: _Session()


@pytest.mark.asyncio
async def test_billed_invalid_dashscope_edit_response_is_charged_once(
    monkeypatch, instant_retries
) -> None:
    from xagent.core.model.image import dashscope as ds_mod
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.image.base import InvalidImageResponseError

    attempts = {"n": 0}
    monkeypatch.setattr(
        ds_mod.aiohttp,
        "ClientSession",
        # 200, billed, but no `output` -- unparsable.
        _dashscope_session_factory({"usage": {"input_tokens": 7}}, attempts),
    )
    model = create_image_model(_image_config("dashscope"))

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await model.edit_image(image_url="https://x/s.png", prompt="edit")
        usage = manager.get_usage()

    assert attempts["n"] == 1
    assert usage.media_calls == 1
    assert usage.details[0]["call_type"] == "edit_image"


@pytest.mark.asyncio
async def test_billed_invalid_dashscope_generate_response_is_charged_once(
    monkeypatch, instant_retries
) -> None:
    from xagent.core.model.image import dashscope as ds_mod
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.image.base import InvalidImageResponseError

    attempts = {"n": 0}
    monkeypatch.setattr(
        ds_mod.aiohttp,
        "ClientSession",
        _dashscope_session_factory({"usage": {}}, attempts),
    )
    model = create_image_model(_image_config("dashscope"))

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await model.generate_image(prompt="p", n=2)
        usage = manager.get_usage()

    assert attempts["n"] == 1
    assert usage.media_calls == 1


def test_invalid_image_response_is_not_retryable() -> None:
    import aiohttp

    from xagent.core.model.image.adapter import retry_image_call
    from xagent.core.model.image.base import InvalidImageResponseError

    assert retry_image_call(InvalidImageResponseError("blocked")) is False
    # Everything else stays as retryable as the previous default, including the
    # plain RuntimeErrors that Gemini/DashScope flatten transient failures into.
    assert retry_image_call(RuntimeError("Image generation timeout")) is True
    assert retry_image_call(aiohttp.ServerTimeoutError()) is True
    assert retry_image_call(ValueError("boom")) is True


def test_invalid_image_response_is_still_a_runtime_error() -> None:
    """Callers documented to catch RuntimeError must keep working."""
    from xagent.core.model.image.base import InvalidImageResponseError

    assert issubclass(InvalidImageResponseError, RuntimeError)


# ---------------------------------------------------------------------------
# Configured model identity must reach the row the provider writes.
#
# model_service stamps the id on the outer retry wrapper, but the inner provider
# is what records, so the id has to live on the provider itself. The aggregator
# groups on `model_id or model`, so without it two same-name configurations
# collapse into one billing group.
# ---------------------------------------------------------------------------


class _StubOpenAIImage:
    url = "https://example.com/1.png"
    b64_json = None


class _StubOpenAIResponse:
    data = [_StubOpenAIImage()]
    usage: dict = {}
    id = "req-1"


class _StubOpenAIImagesClient:
    async def generate(self, **kwargs):
        return _StubOpenAIResponse()

    async def edit(self, **kwargs):
        return _StubOpenAIResponse()


def _stub_openai_transport(model, monkeypatch) -> None:
    """Replace only the network client, keeping the real accounting path."""
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    monkeypatch.setattr(
        model,
        "_client",
        type("_C", (), {"images": _StubOpenAIImagesClient()})(),
    )


class _DBRow:
    """Minimal stand-in for the image `Model` row get_image_model_instance reads."""

    def __init__(self, model_id: str, model_name: str = "gpt-image-1") -> None:
        self.model_id = model_id
        self.model_name = model_name
        self.model_provider = "openai"
        self.api_key = "k"
        self.base_url = None
        self.abilities = ["generate", "edit"]
        self.timeout = 5.0
        self.max_retries = 1


@pytest.mark.asyncio
async def test_same_name_configured_models_bill_separately(monkeypatch) -> None:
    from xagent.core.model.image.adapter import get_image_model_instance

    with TokenContextManager() as manager:
        for configured_id in ("img-a", "img-b"):
            wrapper = get_image_model_instance(_DBRow(configured_id))
            _stub_openai_transport(wrapper._inner, monkeypatch)
            await wrapper.generate_image(prompt="p")
        groups = aggregate_media_usage_by_model(manager.get_usage().details)

    # Two configured models sharing a provider-facing name are two billing
    # identities, not one: different endpoints or pricing, same model_name.
    assert len(groups) == 2
    assert {group["model_id"] for group in groups} == {"img-a", "img-b"}
    assert all(group["quantity"] == 1.0 for group in groups)


@pytest.mark.asyncio
async def test_create_image_model_stamps_configured_id_on_the_row(
    monkeypatch,
) -> None:
    from xagent.core.model.image.adapter import create_image_model

    model = create_image_model(_image_config("openai", model_id="cfg-77"))
    _stub_openai_transport(model._inner, monkeypatch)

    with TokenContextManager() as manager:
        await model.generate_image(prompt="p")
        entry = manager.get_usage().details[0]

    assert entry["model_id"] == "cfg-77"


def test_get_image_model_instance_uses_the_rows_own_id() -> None:
    from xagent.core.model.image.adapter import get_image_model_instance

    wrapper = get_image_model_instance(_DBRow("row-id-9"))
    # Not a "name-provider" composite: both halves are non-unique, so the
    # composite still collapses two same-name configurations into one group.
    assert wrapper._inner.model_id == "row-id-9"


def test_get_image_model_instance_falls_back_when_row_has_no_id() -> None:
    from xagent.core.model.image.adapter import get_image_model_instance

    row = _DBRow("")
    wrapper = get_image_model_instance(row)
    # An identity is always recorded, even for a row carrying no model_id.
    assert wrapper._inner.model_id == "gpt-image-1-openai"


# ---------------------------------------------------------------------------
# Provider-level accounting through a real TokenContextManager.
#
# The pre-existing edit tests monkeypatched record_image_usage, so they stayed
# green if the accounting call were deleted or moved after validation. These
# assert on the persisted row instead.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_generate_records_through_the_real_context(monkeypatch) -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    model = OpenAIImageModel(api_key="k", model_id="oa-1")
    _stub_openai_transport(model, monkeypatch)

    with TokenContextManager() as manager:
        await model.generate_image(prompt="p", size="1024*1024", n=2)
        entry = manager.get_usage().details[0]

    assert entry["type"] == "media"
    assert entry["call_type"] == "generate_image"
    assert entry["quantity"] == 2.0
    assert entry["model_id"] == "oa-1"
    assert entry["resolution"] == "1024x1024"


@pytest.mark.asyncio
async def test_openai_edit_records_through_the_real_context(monkeypatch) -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    model = OpenAIImageModel(api_key="k", model_id="oa-1")
    _stub_openai_transport(model, monkeypatch)
    monkeypatch.setattr("builtins.open", lambda path, mode: _CloseableFile())

    with TokenContextManager() as manager:
        await model.edit_image(image_url="local.png", prompt="edit", n=3)
        entry = manager.get_usage().details[0]

    assert entry["call_type"] == "edit_image"
    assert entry["quantity"] == 3.0
    assert entry["model_id"] == "oa-1"


class _CloseableFile:
    def close(self) -> None:
        return None


class _XinferenceImage:
    url = "https://example.com/x.png"


class _XinferenceResult:
    data = [_XinferenceImage()]
    usage: dict = {}
    id = "xin-1"


@pytest.mark.asyncio
async def test_xinference_generate_records_through_the_real_context(
    monkeypatch,
) -> None:
    from xagent.core.model.image.xinference import XinferenceImageModel

    model = XinferenceImageModel(model_name="sdxl", model_id="xin-cfg")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    monkeypatch.setattr(
        model,
        "_model_handle",
        type("_H", (), {"text_to_image": lambda self, **kw: _XinferenceResult()})(),
    )

    with TokenContextManager() as manager:
        await model.generate_image(prompt="p", size="1024*1024", n=2)
        entry = manager.get_usage().details[0]

    assert entry["call_type"] == "generate_image"
    assert entry["quantity"] == 2.0
    assert entry["model_id"] == "xin-cfg"


@pytest.mark.asyncio
async def test_xinference_edit_records_through_the_real_context(monkeypatch) -> None:
    from xagent.core.model.image.xinference import XinferenceImageModel

    model = XinferenceImageModel(
        model_name="sdxl", model_id="xin-cfg", abilities=["generate", "edit"]
    )
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    monkeypatch.setattr(
        model,
        "_model_handle",
        type("_H", (), {"image_to_image": lambda self, **kw: _XinferenceResult()})(),
    )

    with TokenContextManager() as manager:
        await model.edit_image(
            image_url="https://example.com/s.png", prompt="edit", n=2
        )
        entry = manager.get_usage().details[0]

    assert entry["call_type"] == "edit_image"
    assert entry["quantity"] == 2.0
    assert entry["model_id"] == "xin-cfg"


# ---------------------------------------------------------------------------
# DashScope: the provider's own reported usage is authoritative.
#
# A 200 can succeed partially -- a two-image request returning one successful
# image reports usage.image_count == 1 -- so billing the request's `n` overcharges.
# The request values remain the documented fallback when nothing is reported.
# ---------------------------------------------------------------------------


async def _dashscope_generate(monkeypatch, usage: dict, **call_kwargs):
    from xagent.core.model.image import dashscope as ds_mod

    payload = {
        "usage": usage,
        "output": {
            "choices": [{"message": {"content": [{"image": "https://x/1.png"}]}}]
        },
    }
    monkeypatch.setattr(
        ds_mod.aiohttp, "ClientSession", _dashscope_session_factory(payload, {"n": 0})
    )
    model = ds_mod.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"]
    )
    with TokenContextManager() as manager:
        await model.generate_image(prompt="p", size="1024*1024", **call_kwargs)
        return manager.get_usage().details[0]


@pytest.mark.asyncio
async def test_dashscope_prefers_reported_count_over_requested_n(monkeypatch) -> None:
    entry = await _dashscope_generate(monkeypatch, {"image_count": 1}, n=2)
    # 1, not 2: only one image succeeded, so billing `n` overcharged by one.
    assert entry["quantity"] == 1.0


@pytest.mark.asyncio
async def test_dashscope_honours_a_reported_zero_count(monkeypatch) -> None:
    entry = await _dashscope_generate(monkeypatch, {"image_count": 0}, n=2)
    # A zero-quantity row is the boundary's "billed but produced nothing"
    # convention; substituting `n` would bill images the provider disclaims.
    assert entry["quantity"] == 0.0


@pytest.mark.asyncio
async def test_dashscope_reads_output_prefixed_usage_aliases(monkeypatch) -> None:
    entry = await _dashscope_generate(
        monkeypatch,
        {"output_image_count": 3, "output_width": 2048, "output_height": 1024},
        n=1,
    )
    assert entry["quantity"] == 3.0
    assert entry["resolution"] == "2048x1024"


@pytest.mark.asyncio
async def test_dashscope_falls_back_to_requested_n_when_unreported(
    monkeypatch,
) -> None:
    entry = await _dashscope_generate(monkeypatch, {}, n=2)
    # The documented fallback: correct for a fully successful request.
    assert entry["quantity"] == 2.0
    assert entry["resolution"] == "1024x1024"


@pytest.mark.parametrize("bad", ["many", True, -1, 1.5, float("inf")])
@pytest.mark.asyncio
async def test_dashscope_ignores_unusable_reported_counts(monkeypatch, bad) -> None:
    entry = await _dashscope_generate(monkeypatch, {"image_count": bad}, n=2)
    # A malformed count says nothing, so the request value stands rather than
    # billing 0 or letting a bool bill 1.
    assert entry["quantity"] == 2.0


@pytest.mark.asyncio
async def test_dashscope_groups_under_reported_dimensions(monkeypatch) -> None:
    entry = await _dashscope_generate(monkeypatch, {"width": 512, "height": 512}, n=1)
    # Resolution is part of the aggregate key, so the produced dimensions -- not
    # the requested ones -- decide the billable line item.
    assert entry["resolution"] == "512x512"


@pytest.mark.asyncio
async def test_dashscope_keeps_requested_size_on_half_reported_dimensions(
    monkeypatch,
) -> None:
    entry = await _dashscope_generate(monkeypatch, {"width": 512}, n=1)
    # A width with no height says nothing about the real dimensions.
    assert entry["resolution"] == "1024x1024"


@pytest.mark.asyncio
async def test_dashscope_resolution_is_normalised_across_both_branches(
    monkeypatch,
) -> None:
    """One physical resolution must not produce two aggregate keys.

    The requested size is DashScope `W*H` vocabulary while a reported pair is
    `WxH`, so returning the request verbatim would bill 1024*1024 and
    1024x1024 as separate line items purely by whether the provider reported.
    """
    reported = await _dashscope_generate(
        monkeypatch, {"width": 1024, "height": 1024}, n=1
    )
    fallback = await _dashscope_generate(monkeypatch, {}, n=1)
    assert reported["resolution"] == fallback["resolution"] == "1024x1024"


@pytest.mark.asyncio
async def test_dashscope_edit_prefers_reported_count(monkeypatch) -> None:
    from xagent.core.model.image import dashscope as ds_mod

    payload = {
        "usage": {"image_count": 1, "width": 768, "height": 768},
        "output": {
            "choices": [{"message": {"content": [{"image": "https://x/e.png"}]}}]
        },
    }
    monkeypatch.setattr(
        ds_mod.aiohttp, "ClientSession", _dashscope_session_factory(payload, {"n": 0})
    )
    model = ds_mod.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"]
    )

    with TokenContextManager() as manager:
        await model.edit_image(
            image_url="https://x/s.png", prompt="edit", n=3, size="1024*1024"
        )
        entry = manager.get_usage().details[0]

    # The same rule applies to edits, which had only the request values before.
    assert entry["call_type"] == "edit_image"
    assert entry["quantity"] == 1.0
    assert entry["resolution"] == "768x768"


@pytest.mark.asyncio
async def test_dashscope_edit_falls_back_to_requested_values(monkeypatch) -> None:
    from xagent.core.model.image import dashscope as ds_mod

    payload = {
        "usage": {},
        "output": {
            "choices": [{"message": {"content": [{"image": "https://x/e.png"}]}}]
        },
    }
    monkeypatch.setattr(
        ds_mod.aiohttp, "ClientSession", _dashscope_session_factory(payload, {"n": 0})
    )
    model = ds_mod.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"]
    )

    with TokenContextManager() as manager:
        await model.edit_image(
            image_url="https://x/s.png", prompt="edit", n=2, size="512*512"
        )
        entry = manager.get_usage().details[0]

    assert entry["quantity"] == 2.0
    assert entry["resolution"] == "512x512"


@pytest.mark.asyncio
async def test_gemini_edit_records_one_image_through_the_real_context(
    monkeypatch,
) -> None:
    """Companion to the kwargs-level seam test above.

    That test monkeypatches record_image_usage, so it stays green if the
    accounting call is deleted or moved after validation. This one asserts the
    persisted row instead.
    """
    from xagent.core.model.image import gemini as gemini_mod

    payload = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [{"text": "![Image](https://example.com/edit.png)"}]
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4},
    }
    monkeypatch.setattr(
        gemini_mod.httpx, "AsyncClient", _gemini_client_factory(payload, {"n": 0})
    )
    model = gemini_mod.GeminiImageModel(
        model_name="gemini-3-pro-image-preview-2k",
        api_key="k",
        abilities=["generate", "edit"],
        model_id="gem-cfg",
    )

    with TokenContextManager() as manager:
        await model.edit_image(
            image_url="data:image/png;base64,iVBORw0KGgo=", prompt="edit", n=3
        )
        entry = manager.get_usage().details[0]

    assert entry["call_type"] == "edit_image"
    # 1, not 3: Gemini cannot honour `n`, so billing it would charge for images
    # the provider was never asked to make.
    assert entry["quantity"] == 1.0
    assert entry["provider_tokens"] == 13
    assert entry["model_id"] == "gem-cfg"


@pytest.mark.asyncio
async def test_dashscope_edit_records_through_the_real_context(monkeypatch) -> None:
    """Companion to the kwargs-level DashScope edit seam test above."""
    from xagent.core.model.image import dashscope as ds_mod

    payload = {
        "usage": {"input_tokens": 6},
        "output": {
            "choices": [{"message": {"content": [{"image": "https://x/e.png"}]}}]
        },
    }
    monkeypatch.setattr(
        ds_mod.aiohttp, "ClientSession", _dashscope_session_factory(payload, {"n": 0})
    )
    model = ds_mod.DashScopeImageModel(
        model_name="wanx",
        api_key="k",
        abilities=["generate", "edit"],
        model_id="ds-cfg",
    )

    with TokenContextManager() as manager:
        await model.edit_image(image_url="https://x/s.png", prompt="edit", n=4)
        entry = manager.get_usage().details[0]

    assert entry["call_type"] == "edit_image"
    assert entry["quantity"] == 4.0
    assert entry["provider_tokens"] == 6
    assert entry["model_id"] == "ds-cfg"


def test_every_provider_constructor_accepts_a_configured_id() -> None:
    """The raw construction path in model_service builds providers directly.

    That path previously stamped model_id after construction, on providers that
    had no such attribute at all -- so a provider recording usage before the
    stamp, or reading an attribute the class never declared, billed under the
    non-unique provider-facing name. Every provider must accept and keep it.
    """
    from xagent.core.model.image.dashscope import DashScopeImageModel
    from xagent.core.model.image.gemini import GeminiImageModel
    from xagent.core.model.image.openai import OpenAIImageModel
    from xagent.core.model.image.xinference import XinferenceImageModel

    for factory in (
        lambda: GeminiImageModel(api_key="k", model_id="cfg"),
        lambda: DashScopeImageModel(api_key="k", model_id="cfg"),
        lambda: OpenAIImageModel(api_key="k", model_id="cfg"),
        lambda: XinferenceImageModel(model_id="cfg"),
    ):
        assert factory().model_id == "cfg"


def test_providers_default_to_an_empty_configured_id() -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    # "" not None: the shared boundary strips strings, and the aggregator falls
    # back to the model name when the id is empty.
    assert OpenAIImageModel(api_key="k").model_id == ""


@pytest.mark.parametrize("first", ["bad", True, float("inf"), None])
def test_unusable_first_alias_does_not_shadow_a_valid_second(first: object) -> None:
    """Alias order must survive a malformed value.

    Forwarding raw values must not mean stopping at the first present key: a
    payload reporting `prompt_tokens: "bad"` alongside a valid `input_tokens: 7`
    has to bill 7, not 0.
    """
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": first, "input_tokens": 7}}, model_name="m"
        )
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == 7


def test_a_huge_integer_alias_does_not_shadow_a_real_one() -> None:
    # int(10**400) succeeds, so this value is *parseable* -- but it is not a
    # plausible token count, and letting it through both shadowed the usable
    # `input_tokens` alias and put a 401-digit number in the persisted row.
    # Bounded like a reported count instead. (An earlier revision of this test
    # asserted the opposite; that premise was wrong.)
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": 10**400, "input_tokens": 7}}, model_name="m"
        )
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == 7


def test_a_reported_zero_is_not_treated_as_a_missing_alias() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": 0, "input_tokens": 7}}, model_name="m"
        )
        entry = manager.get_usage().details[0]

    # An explicit provider zero is a real measurement, so the later alias must
    # not override it.
    assert entry["provider_input_tokens"] == 0


# ---------------------------------------------------------------------------
# Malformed *elements*, not just malformed containers.
#
# The structural checks validate that `content`/`candidates`/`parts` are
# non-empty lists, never that their elements are dicts. Walking such a body
# fails implicitly -- TypeError on `"image" not in 123`, AttributeError on
# `"str".get(...)` -- so before these were classified positionally they reached
# the blanket handler as plain RuntimeErrors and were retried, re-billing an
# already-billed 200 once per attempt.
# ---------------------------------------------------------------------------


def _dashscope_body(content: object) -> dict:
    return {
        "usage": {},
        "output": {"choices": [{"message": {"content": content}}]},
    }


@pytest.mark.parametrize(
    "payload",
    [
        _dashscope_body([None]),
        _dashscope_body([123]),
        {"usage": {}, "output": {"choices": [123]}},
        {"usage": {}, "output": "not-a-dict"},
        {"usage": {}, "output": {"choices": [{"message": 7}]}},
    ],
)
@pytest.mark.asyncio
async def test_malformed_dashscope_body_is_billed_once(
    monkeypatch, instant_retries, payload
) -> None:
    from xagent.core.model.image import dashscope as ds_mod
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.image.base import InvalidImageResponseError

    attempts = {"n": 0}
    monkeypatch.setattr(
        ds_mod.aiohttp, "ClientSession", _dashscope_session_factory(payload, attempts)
    )
    model = create_image_model(_image_config("dashscope"))

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await model.generate_image(prompt="p")
        usage = manager.get_usage()

    assert attempts["n"] == 1
    assert usage.media_calls == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"usageMetadata": {"promptTokenCount": 5}, "candidates": ["oops"]},
        {"usageMetadata": {"promptTokenCount": 5}, "candidates": [7]},
        {"usageMetadata": {"promptTokenCount": 5}, "candidates": {"a": 1}},
        {
            "usageMetadata": {"promptTokenCount": 5},
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": ["not-a-dict"]}}
            ],
        },
    ],
)
@pytest.mark.asyncio
async def test_malformed_gemini_body_is_billed_once(
    monkeypatch, instant_retries, payload
) -> None:
    from xagent.core.model.image import gemini as gemini_mod
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.image.base import InvalidImageResponseError

    attempts = {"n": 0}
    monkeypatch.setattr(
        gemini_mod.httpx, "AsyncClient", _gemini_client_factory(payload, attempts)
    )
    model = create_image_model(_image_config("gemini"))

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await model.generate_image(prompt="p")
        usage = manager.get_usage()

    assert attempts["n"] == 1
    assert usage.media_calls == 1


@pytest.mark.asyncio
async def test_malformed_body_on_the_edit_path_is_billed_once(
    monkeypatch, instant_retries
) -> None:
    from xagent.core.model.image import dashscope as ds_mod
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.image.base import InvalidImageResponseError

    attempts = {"n": 0}
    monkeypatch.setattr(
        ds_mod.aiohttp,
        "ClientSession",
        _dashscope_session_factory(_dashscope_body([None]), attempts),
    )
    model = create_image_model(_image_config("dashscope"))

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await model.edit_image(image_url="https://x/s.png", prompt="edit")
        usage = manager.get_usage()

    assert attempts["n"] == 1
    assert usage.media_calls == 1


def test_reclassified_errors_keep_the_original_cause() -> None:
    """The original failure must remain diagnosable from the traceback."""
    from xagent.core.model.image.base import (
        InvalidImageResponseError,
        invalid_response_from,
    )

    original = TypeError("argument of type 'int' is not iterable")
    reclassified = invalid_response_from(original, "Invalid response format")

    assert isinstance(reclassified, InvalidImageResponseError)
    assert "TypeError" in str(reclassified)
    assert "not iterable" in str(reclassified)


def test_absurd_provider_counts_are_rejected_not_billed_as_zero() -> None:
    """An unbounded provider integer must not reach the quantity slot.

    The write boundary folds a value too large for a float to quantity 0.0, so
    forwarding one would bill nothing for a call that returned a real image --
    worse than falling back to the request's own bounded count. The same bound
    keeps a provider-reported dimension from becoming a several-hundred-character
    aggregate key.
    """
    from xagent.core.model.image.usage import usable_image_count

    assert usable_image_count(10**400) is None
    assert usable_image_count(2**53 + 1) is None
    assert usable_image_count(10**30) is None
    # The bound is well above any real image response.
    assert usable_image_count(1_000_000) == 1_000_000
    assert usable_image_count(1_000_001) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, 3),
        (0, 0),
        (3.0, 3),
        ("3", 3),
        (-1, None),
        (1.5, None),
        (True, None),
        (False, None),
        (float("inf"), None),
        (float("nan"), None),
        (None, None),
        ("x", None),
        ([], None),
    ],
)
def test_usable_image_count_contract(value: object, expected: object) -> None:
    from xagent.core.model.image.usage import usable_image_count

    assert usable_image_count(value) == expected


@pytest.mark.parametrize("size", [2048, 0, None, ["1024", "1024"], {"w": 1}])
@pytest.mark.asyncio
async def test_dashscope_drops_a_non_string_requested_size(monkeypatch, size) -> None:
    """A non-string size must not become a half-resolution aggregate key.

    edit_image reads `size` straight from caller kwargs with no normalisation,
    so an int 2048 would otherwise be recorded as the key "2048" -- which joins
    no price table and is indistinguishable from a real tier.
    """
    from xagent.core.model.image import dashscope as ds_mod

    payload = {
        "usage": {},
        "output": {
            "choices": [{"message": {"content": [{"image": "https://x/e.png"}]}}]
        },
    }
    monkeypatch.setattr(
        ds_mod.aiohttp, "ClientSession", _dashscope_session_factory(payload, {"n": 0})
    )
    model = ds_mod.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"]
    )

    with TokenContextManager() as manager:
        await model.edit_image(image_url="https://x/s.png", prompt="edit", size=size)
        entry = manager.get_usage().details[0]

    assert entry["resolution"] == ""


@pytest.mark.asyncio
async def test_dashscope_reported_dimensions_win_over_a_non_string_size(
    monkeypatch,
) -> None:
    from xagent.core.model.image import dashscope as ds_mod

    payload = {
        "usage": {"width": 512, "height": 512},
        "output": {
            "choices": [{"message": {"content": [{"image": "https://x/e.png"}]}}]
        },
    }
    monkeypatch.setattr(
        ds_mod.aiohttp, "ClientSession", _dashscope_session_factory(payload, {"n": 0})
    )
    model = ds_mod.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"]
    )

    with TokenContextManager() as manager:
        await model.edit_image(image_url="https://x/s.png", prompt="edit", size=2048)
        entry = manager.get_usage().details[0]

    assert entry["resolution"] == "512x512"


class _HostileEq:
    """A provider value whose equality comparison raises."""

    def __eq__(self, other: object) -> bool:
        raise ValueError("comparison boom")

    def __hash__(self) -> int:
        return 0


class _ArrayLikeEq:
    """A provider value whose comparison is non-boolean, as arrays are."""

    def __eq__(self, other: object) -> list:  # type: ignore[override]
        return [False]

    def __hash__(self) -> int:
        return 0

    def __bool__(self) -> bool:
        raise ValueError("truth value is ambiguous")


@pytest.mark.parametrize("first", [_HostileEq(), _ArrayLikeEq()])
def test_two_unusable_aliases_still_keep_the_row(first: object) -> None:
    """Tracking the fallback must not compare a raw provider value to 0.

    Once an unusable value is held as the fallback, `fallback == 0` invokes that
    value's own ``__eq__``, which escapes into record_image_usage's handler and
    drops the whole billable row -- the failure this helper exists to prevent.
    Needs *two* unusable aliases: with one, the comparison still runs against
    the initial int and cannot reach the provider value.

    Only ``_HostileEq`` detects a regression here; ``_ArrayLikeEq`` is a
    contract case rather than a second detector, because ``bool([False])`` is
    True and the buggy comparison happens to take the same branch. It is kept
    so a future rewrite that does depend on the comparison's truthiness is
    covered, but it is not evidence on its own.
    """
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": first, "input_tokens": "bad"}}, model_name="m"
        )
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["provider_input_tokens"] == 0


def test_a_hostile_first_alias_does_not_shadow_a_valid_second() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": _HostileEq(), "input_tokens": 7}},
            model_name="m",
        )
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == 7


# ---------------------------------------------------------------------------
# Provider payloads must be walked inside record_image_usage's swallow.
#
# An argument expression runs in the *caller's* frame, so walking the usage
# payload there put a provider-controlled `.get` outside the only protection
# that exists -- turning a successful 200 that returned a real image into a
# failed call with no billing row.
# ---------------------------------------------------------------------------


class _HostileGet(dict):
    """A usage payload whose lookups raise, as a proxy-mangled body might."""

    def get(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("payload boom")


_DASHSCOPE_GOOD_OUTPUT = {
    "choices": [{"message": {"content": [{"image": "https://x/1.png"}]}}]
}


@pytest.mark.asyncio
async def test_hostile_usage_payload_does_not_fail_a_successful_call(
    monkeypatch,
) -> None:
    from xagent.core.model.image import dashscope as ds_mod

    monkeypatch.setattr(
        ds_mod.aiohttp,
        "ClientSession",
        _dashscope_session_factory(
            {"usage": _HostileGet(), "output": _DASHSCOPE_GOOD_OUTPUT}, {"n": 0}
        ),
    )
    model = ds_mod.DashScopeImageModel(model_name="wanx", api_key="k")

    with TokenContextManager() as manager:
        result = await model.generate_image(prompt="p", size="1024*1024", n=2)
        usage = manager.get_usage()

    # The image came back, so the call must succeed and be billed. Accounting
    # is best-effort; it may not take the user's result down with it.
    assert result["image_url"] == "https://x/1.png"
    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 2.0


@pytest.mark.asyncio
async def test_absurd_reported_count_falls_back_instead_of_billing_zero(
    monkeypatch,
) -> None:
    """A count too large for a float must not bill 0 for a real image.

    The write boundary folds such a value to quantity 0.0, so forwarding it
    would record nothing billable for a call that did return an image. The
    request's own bounded count is the better answer.
    """
    from xagent.core.model.image import dashscope as ds_mod

    monkeypatch.setattr(
        ds_mod.aiohttp,
        "ClientSession",
        _dashscope_session_factory(
            {"usage": {"image_count": 10**400}, "output": _DASHSCOPE_GOOD_OUTPUT},
            {"n": 0},
        ),
    )
    model = ds_mod.DashScopeImageModel(model_name="wanx", api_key="k")

    with TokenContextManager() as manager:
        await model.generate_image(prompt="p", size="1024*1024", n=1)
        entry = manager.get_usage().details[0]

    assert entry["quantity"] == 1.0


@pytest.mark.asyncio
async def test_absurd_reported_dimensions_do_not_become_the_aggregate_key(
    monkeypatch,
) -> None:
    from xagent.core.model.image import dashscope as ds_mod

    monkeypatch.setattr(
        ds_mod.aiohttp,
        "ClientSession",
        _dashscope_session_factory(
            {
                "usage": {"width": 10**60, "height": 10**60},
                "output": _DASHSCOPE_GOOD_OUTPUT,
            },
            {"n": 0},
        ),
    )
    model = ds_mod.DashScopeImageModel(model_name="wanx", api_key="k")

    with TokenContextManager() as manager:
        await model.generate_image(prompt="p", size="1024*1024")
        entry = manager.get_usage().details[0]

    # The requested tier, not a 123-character number pair that joins no price
    # table.
    assert entry["resolution"] == "1024x1024"


# ---------------------------------------------------------------------------
# A billed 200 must be metered even when its own usage block is malformed.
#
# The metering call sits after the first reads of the parsed body, so those
# reads must not raise: the resulting error is classified as an already-billed
# invalid response, which asserts the row exists and tells the retry policy not
# to try again. If the row was never written, the call is charged, unmetered and
# unretried at once.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metadata", [[1, 2], "nope", 7, None])
@pytest.mark.asyncio
async def test_gemini_malformed_usage_metadata_still_records_the_call(
    monkeypatch, metadata
) -> None:
    from xagent.core.model.image import gemini as gemini_mod

    monkeypatch.setattr(
        gemini_mod.httpx,
        "AsyncClient",
        _gemini_client_factory({"usageMetadata": metadata, "candidates": []}, {"n": 0}),
    )
    model = gemini_mod.GeminiImageModel(model_name="gemini-image", api_key="k")

    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await model.generate_image(prompt="p")
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["provider_tokens"] == 0


@pytest.mark.parametrize("body", [["not", "a", "dict"], "nope", 7])
@pytest.mark.asyncio
async def test_non_dict_200_body_still_records_the_billed_call(
    monkeypatch, body
) -> None:
    """A JSON array is a legal 200 body, and the provider still charged for it."""
    from xagent.core.model.image import dashscope as ds_mod
    from xagent.core.model.image import gemini as gemini_mod

    monkeypatch.setattr(
        gemini_mod.httpx, "AsyncClient", _gemini_client_factory(body, {"n": 0})
    )
    gemini = gemini_mod.GeminiImageModel(model_name="gemini-image", api_key="k")
    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await gemini.generate_image(prompt="p")
        assert manager.get_usage().media_calls == 1

    monkeypatch.setattr(
        ds_mod.aiohttp, "ClientSession", _dashscope_session_factory(body, {"n": 0})
    )
    dashscope = ds_mod.DashScopeImageModel(model_name="wanx", api_key="k")
    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError):
            await dashscope.generate_image(prompt="p")
        assert manager.get_usage().media_calls == 1


def test_countable_does_not_let_a_raising_int_escape() -> None:
    """_countable runs a frame above the boundary, so it must never raise."""
    from xagent.core.model.image.usage import _countable, usable_image_count

    class _RaisingInt:
        def __int__(self) -> int:
            raise RuntimeError("int boom")

    class _RaisingFloat:
        def __float__(self) -> float:
            raise RuntimeError("float boom")

    assert _countable(_RaisingInt()) is False
    assert usable_image_count(_RaisingFloat()) is None


@pytest.mark.parametrize(
    ("result", "count_from", "size_from"),
    [
        (_HostileGet(), {}, {}),
        ({"usage": {}}, _HostileGet(), {}),
        ({"usage": {}}, {}, _HostileGet()),
        (_HostileGet(), _HostileGet(), _HostileGet()),
    ],
)
def test_a_hostile_payload_at_any_position_keeps_the_row(
    result: object, count_from: object, size_from: object
) -> None:
    """Every provider-controlled read degrades, none of them costs the row.

    The reads are guarded individually rather than only by the outer handler:
    reaching that handler means no row at all, and a row carrying the request's
    own values is strictly better than losing evidence of a billed call.
    """
    with TokenContextManager() as manager:
        record_image_usage(
            result,  # type: ignore[arg-type]
            model_name="m",
            image_count=2,
            resolution="1024x1024",
            reported_count_from=count_from,
            reported_size_from=size_from,
        )
        usage = manager.get_usage()

    assert usage.media_calls == 1
    entry = usage.details[0]
    assert entry["quantity"] == 2.0
    assert entry["resolution"] == "1024x1024"


@pytest.mark.parametrize(
    ("payload", "expected_in", "expected_out"),
    [
        # The boundary floors token fields with max(0, ...), so a negative first
        # alias bills 0 -- it must not stop the search while a usable alias is
        # still to come.
        ({"prompt_tokens": -5, "input_tokens": 9}, 9, 0),
        ({"completion_tokens": -1, "output_tokens": 1290}, 0, 1290),
        ({"prompt_tokens": -7.9, "input_tokens": 9}, 9, 0),
        # An absurd magnitude must not shadow a real alias either, nor land in
        # the persisted row as a several-hundred-digit number.
        ({"prompt_tokens": 10**400, "input_tokens": 7}, 7, 0),
    ],
)
def test_a_value_the_boundary_would_discard_does_not_shadow_a_valid_alias(
    payload: dict, expected_in: int, expected_out: int
) -> None:
    """`_countable` must mirror every reduction the boundary applies.

    The question it answers is not "is this a number" but "would this survive
    the write boundary" -- a value that reaches the boundary and is discarded
    there bills 0 while a usable later alias goes unread.
    """
    with TokenContextManager() as manager:
        record_image_usage({"usage": payload}, model_name="m")
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == expected_in
    assert entry["provider_output_tokens"] == expected_out


def test_an_only_unusable_token_value_still_keeps_the_row() -> None:
    with TokenContextManager() as manager:
        record_image_usage({"usage": {"prompt_tokens": -5}}, model_name="m")
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["provider_input_tokens"] == 0


def test_ordinary_token_counts_are_unchanged() -> None:
    """Guard against the bound or the sign check eating real values."""
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": 11, "completion_tokens": 5}}, model_name="m"
        )
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == 11
    assert entry["provider_output_tokens"] == 5
    assert entry["provider_tokens"] == 16


@pytest.mark.parametrize("negative_fraction", [-0.5, -0.9, -0.25])
def test_a_negative_fraction_does_not_shadow_a_valid_alias(
    negative_fraction: float,
) -> None:
    """The sign must be judged on the value, not on ``int(value)``.

    ``int()`` truncates toward zero, so ``int(-0.5)`` is ``0`` -- a negative
    fraction passes a check written against the truncated result, while the
    boundary still bills 0. The alias scan then stops on a value worth nothing.
    """
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": negative_fraction, "input_tokens": 9}},
            model_name="m",
        )
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == 9


def test_countable_agrees_with_the_write_boundary() -> None:
    """The gate exists only to predict the boundary; drift defeats its purpose.

    Any value the gate calls usable must actually be billed by the boundary.
    Pinned as a property over a spread of shapes rather than one example,
    because every disagreement found so far was a different shape.
    """
    from xagent.core.model.chat.token_context import _coerce_media_tokens
    from xagent.core.model.image.usage import _MAX_TOKENS, _countable

    values: list[object] = [
        -0.5,
        -0.9,
        -0.0,
        0,
        7,
        7.9,
        -7.9,
        -5,
        11,
        _MAX_TOKENS,
        _MAX_TOKENS + 1,
        2**63,
        10**400,
        float("inf"),
        float("nan"),
        "7",
        "123",
        " 7 ",
        b"7",
        "-7",
        "x",
        7.9,
        -0.5,
        1e-09,
        None,
        True,
        False,
        [],
    ]
    for value in values:
        billed = max(0, _coerce_media_tokens(value))
        if _countable(value):
            # "Usable" must mean the boundary bills exactly this, so stopping
            # the alias scan here cannot cost real usage. Zero is the one
            # legitimate exception: a provider-reported 0 is a real measurement.
            assert billed == int(value) or billed == 0, value  # type: ignore[call-overload]
        elif billed > 0 and billed <= _MAX_TOKENS:
            # The other direction, which is what actually regressed twice: a
            # gate STRICTER than the boundary skips a value the boundary would
            # have billed, so a real provider count loses to a later alias.
            # Over-bound values are excluded deliberately -- they are still
            # billed in full when they are the only alias present.
            raise AssertionError(
                f"gate rejects {value!r} but the boundary bills {billed}"
            )


def test_a_hostile_comparison_result_cannot_cost_the_row() -> None:
    """The whole gate predicate must evaluate inside its guard.

    Computing the comparison inside the try and testing it outside still leaves
    an escape: `not (value < 0)` invokes __bool__ on the comparison *result*,
    which a provider object can raise from. Unreachable from today's call sites
    (they carry JSON scalars and SDK ints), but the guard is a one-token
    difference and a refactor should not be able to reopen it.
    """
    from xagent.core.model.image.usage import _countable

    class _SneakyComparison:
        def __int__(self) -> int:
            return 5

        def _unbooleanable(self) -> object:
            class _B:
                def __bool__(self) -> bool:
                    raise ValueError("bool boom")

            return _B()

        def __lt__(self, other: object) -> object:
            return self._unbooleanable()

        def __le__(self, other: object) -> object:
            return self._unbooleanable()

    # Not asserting that the object is rejected -- that was an artefact of an
    # earlier gate that compared `value` directly. The contract is that a
    # hostile comparison cannot cost the row, whichever way the gate decides.
    assert _countable(_SneakyComparison()) in (True, False)

    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": _SneakyComparison(), "input_tokens": 9}},
            model_name="m",
        )
        usage = manager.get_usage()

    assert usage.media_calls == 1
    # Either the object's int() value or the later alias -- both are real
    # numbers the boundary can bill. What must not happen is a lost row.
    assert usage.details[0]["provider_input_tokens"] in (5, 9)


def test_a_sub_one_fraction_does_not_shadow_a_valid_alias() -> None:
    """int() truncates toward zero, so 1e-9 becomes 0 at the boundary."""
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": 1e-09, "input_tokens": 9}}, model_name="m"
        )
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == 9


def test_an_over_bound_sole_alias_is_still_billed_in_full() -> None:
    """The bound gates the scan decision, never the billed amount.

    An absurd value only loses out to a *usable later alias*. When it is all the
    provider reported, it is still what the provider reported, and the boundary
    stores it -- so the bound cannot under-bill real usage.
    """
    with TokenContextManager() as manager:
        record_image_usage({"usage": {"prompt_tokens": 2_000_000_000}}, model_name="m")
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == 2_000_000_000


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        # Numeric strings: the boundary bills these correctly, so rejecting them
        # made a real provider count lose to the later alias.
        ("123", 123),
        (" 7 ", 7),
        (b"7", 7),
        # Fractions: the boundary truncates them, which is a real billed value.
        (7.9, 7),
        (decimal.Decimal("7.5"), 7),
        (fractions.Fraction(15, 2), 7),
    ],
)
def test_the_gate_is_not_stricter_than_the_boundary(
    primary: object, expected: int
) -> None:
    """A gate stricter than the boundary under-bills, same as a looser one.

    Both regressions here came from deriving the decision from `value` rather
    than from what the boundary will actually bill: comparing `value` cannot
    order a numeric string against an int, and a truncation test rejects
    fractions the boundary bills as their integer part. Either way a real
    provider count silently lost to `input_tokens`.
    """
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": primary, "input_tokens": 9}}, model_name="m"
        )
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == expected


@pytest.mark.parametrize("zero", [0, 0.0, -0.0])
def test_an_explicit_provider_zero_stops_the_alias_scan(zero: object) -> None:
    """A reported zero is a measurement, not a missing value.

    "This call used no tokens" is something a provider can state, and a later
    alias must not silently override it. Distinguished from -0.5 and 1e-9,
    which also reach 0 through int() but which the boundary bills 0 while a
    usable alias goes unread.
    """
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"prompt_tokens": zero, "input_tokens": 7}}, model_name="m"
        )
        entry = manager.get_usage().details[0]

    assert entry["provider_input_tokens"] == 0


@pytest.mark.parametrize("fractional", [2.5, 7.9, decimal.Decimal("2.5")])
def test_a_fractional_reported_count_falls_back_to_the_request(
    fractional: object,
) -> None:
    """Counts are discrete, unlike tokens -- the two gates differ on purpose.

    `_countable` deliberately accepts 7.9 for a token field, because the write
    boundary truncates it to 7 and that is a real billed amount. A count of 2.5
    is not "2.5 images"; it is a malformed response, and the request's `n` --
    what the provider was actually asked for -- is the more trustworthy number.

    Pinned so that a future pass "unifying" the two validators cannot quietly
    start billing fractional image counts.
    """
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {}},
            model_name="m",
            image_count=1,
            reported_count_from={"image_count": fractional},
        )
        entry = manager.get_usage().details[0]

    assert entry["quantity"] == 1.0


@pytest.mark.parametrize(("reported", "expected"), [("3", 3.0), (3.0, 3.0)])
def test_an_integral_reported_count_is_still_honoured(
    reported: object, expected: float
) -> None:
    """The fractional rejection must not catch integral values in other types."""
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {}},
            model_name="m",
            image_count=1,
            reported_count_from={"image_count": reported},
        )
        entry = manager.get_usage().details[0]

    assert entry["quantity"] == expected


def test_alias_selection_matches_an_independent_reference_over_all_pairs() -> None:
    """Differential test: every (primary, secondary) alias pair, against a
    reference implementation written from the contract rather than from the code.

    This exists because assertion-shaped tests kept missing real defects here.
    A one-directional property ("everything the gate accepts, the boundary
    bills") passed while numeric strings were being wrongly rejected; making it
    bidirectional caught that, but only because I happened to add the reverse
    half. A reference implementation does not depend on my guessing which
    direction to assert.

    The reference states the contract directly: skip bools (never counts), take
    the first alias the boundary bills nonzero within the deliberate magnitude
    bound, treat an explicit integral zero as authoritative, and otherwise let
    the first present value reach the boundary raw.
    """
    from xagent.core.model.chat.token_context import _coerce_media_tokens
    from xagent.core.model.image.usage import _MAX_TOKENS

    def reference(values: list) -> int:
        for value in values:
            if isinstance(value, bool):
                continue
            billed = max(0, _coerce_media_tokens(value))
            if 0 < billed <= _MAX_TOKENS:
                return billed
            try:
                if int(value) == 0 and value == 0:
                    return 0
            except Exception:  # noqa: BLE001
                pass
        return max(0, _coerce_media_tokens(values[0])) if values else 0

    candidates: list = [
        "123",
        " 7 ",
        b"7",
        "x",
        "",
        7,
        7.0,
        7.9,
        0,
        0.0,
        -0.0,
        -5,
        -0.5,
        1e-09,
        10**9,
        10**9 + 1,
        10**400,
        float("inf"),
        float("nan"),
        True,
        False,
        [],
        decimal.Decimal("7"),
        decimal.Decimal("7.5"),
        fractions.Fraction(15, 2),
    ]

    for primary in candidates:
        for secondary in candidates:
            with TokenContextManager() as manager:
                record_image_usage(
                    {"usage": {"prompt_tokens": primary, "input_tokens": secondary}},
                    model_name="m",
                )
                usage = manager.get_usage()

            # The row must never be lost, whatever the provider sent.
            assert usage.media_calls == 1, (primary, secondary)
            assert usage.details[0]["provider_input_tokens"] == reference(
                [primary, secondary]
            ), (primary, secondary)


@pytest.mark.parametrize("provider", ["openai", "xinference"])
@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.asyncio
async def test_configured_id_reaches_the_row_through_the_wrapper(
    monkeypatch, provider: str, method: str
) -> None:
    """The wrapped path must carry the id too, for generate and edit alike.

    `create_image_model` returns a retry wrapper, and the inner provider is what
    records. Covered per provider and per method because the recording call is
    duplicated at eight sites and a missed one is invisible from any single test.
    """
    from xagent.core.model.image.adapter import create_image_model

    model = create_image_model(_image_config(provider, model_id="cfg-9"))
    inner = model._inner

    if provider == "openai":
        _stub_openai_transport(inner, monkeypatch)
        monkeypatch.setattr("builtins.open", lambda path, mode: _CloseableFile())
    else:
        monkeypatch.setattr(inner, "_ensure_client", lambda: None)
        monkeypatch.setattr(
            inner,
            "_model_handle",
            type(
                "_H",
                (),
                {
                    "text_to_image": lambda self, **kw: _XinferenceResult(),
                    "image_to_image": lambda self, **kw: _XinferenceResult(),
                },
            )(),
        )

    with TokenContextManager() as manager:
        if method == "generate":
            await model.generate_image(prompt="p")
        else:
            await model.edit_image(image_url="local.png", prompt="edit")
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["model_id"] == "cfg-9"
    assert usage.details[0]["call_type"] == (
        "generate_image" if method == "generate" else "edit_image"
    )


@pytest.mark.asyncio
async def test_dashscope_count_and_resolution_match_an_independent_reference(
    monkeypatch,
) -> None:
    """Differential test over the provider-reported vs request cross-product.

    A reference implementation written from the contract, not from the code:
    a usable reported count is non-bool, integral, and within the magnitude
    bound; it overrides the request's `n`; dimensions need both halves usable
    and normalise to WxH; a non-string request size is dropped.

    Kept as a test rather than a one-off script because assertion-shaped tests
    repeatedly missed defects in this file by only checking the direction I
    happened to be thinking about. Verified to fail on injected defects --
    removing the fractional-count rejection, and removing the `*`->`x`
    normalisation -- rather than assumed to be effective.
    """
    from xagent.core.model.image import dashscope as ds_mod
    from xagent.core.model.image.usage import _MAX_REPORTED

    def usable(value: object) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            number = float(value)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return None
        if not math.isfinite(number) or number < 0 or number > _MAX_REPORTED:
            return None
        if number != int(number):
            return None
        return int(number)

    def reference_quantity(usage: object, requested: object) -> float:
        if isinstance(usage, dict):
            for key in ("image_count", "output_image_count"):
                found = usable(usage.get(key))
                if found is not None:
                    return float(found)
        if isinstance(requested, bool):
            return 0.0
        try:
            number = float(requested)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return 0.0
        if not math.isfinite(number) or number < 0:
            return 0.0
        return number

    def reference_resolution(usage: object, size: object) -> str:
        if isinstance(usage, dict):
            for width_key, height_key in (
                ("width", "height"),
                ("output_width", "output_height"),
            ):
                width = usable(usage.get(width_key))
                height = usable(usage.get(height_key))
                if width and height:
                    return f"{width}x{height}"
        if not isinstance(size, str):
            return ""
        return size.replace("*", "x").strip()

    output = {"choices": [{"message": {"content": [{"image": "https://x/1.png"}]}}]}
    counts: list = [None, 0, 1, 3, -1, 2.5, True, "3", 10**7, float("nan")]
    dimensions: list = [None, 0, 512, -5, 2048.0, "512"]
    sizes: list = ["1024*1024", "1024x1024", "", None, 2048]
    requested: list = [1, 2, 0, None]

    for count in counts:
        for width in dimensions:
            for height in dimensions[:3]:
                for size in sizes:
                    for n in requested:
                        usage: dict = {}
                        if count is not None:
                            usage["image_count"] = count
                        if width is not None:
                            usage["width"] = width
                        if height is not None:
                            usage["height"] = height
                        monkeypatch.setattr(
                            ds_mod.aiohttp,
                            "ClientSession",
                            _dashscope_session_factory(
                                {"usage": usage, "output": output}, {"n": 0}
                            ),
                        )
                        model = ds_mod.DashScopeImageModel(
                            model_name="wanx", api_key="k"
                        )
                        kwargs = {} if n is None else {"n": n}
                        with TokenContextManager() as manager:
                            await model.generate_image(prompt="p", size=size, **kwargs)
                            recorded = manager.get_usage()

                        context = (usage, size, n)
                        assert recorded.media_calls == 1, context
                        entry = recorded.details[0]
                        assert entry["quantity"] == reference_quantity(
                            usage, 1 if n is None else n
                        ), context
                        assert entry["resolution"] == reference_resolution(
                            usage, size
                        ), context


class _XinferenceDictResult(dict):
    """The xinference client returns `response.json()` verbatim -- a raw dict."""


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [None]},
        {"data": ["url-as-a-bare-string"]},
        {"data": 5},
        {"data": [[]]},
    ],
)
@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.asyncio
async def test_malformed_xinference_body_is_billed_once(
    monkeypatch, instant_retries, payload: dict, method: str
) -> None:
    """A billed xinference 200 with a malformed body must be charged once.

    xinference_client's text_to_image/image_to_image return `response.json()`
    verbatim -- raw server JSON, despite the annotation -- so a malformed body
    reaches the parsing code directly. Walking it used to raise before the
    metering call, inside the try whose handler rewraps into a plain
    RuntimeError: the row was lost AND the error was retryable, so every attempt
    was billed and none recorded. This was the one provider that received
    neither the metering-before-validation order nor the positional
    classification.
    """
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.image.base import InvalidImageResponseError

    attempts = {"n": 0}
    model = create_image_model(_image_config("xinference", max_retries=5))
    inner = model._inner
    monkeypatch.setattr(inner, "_ensure_client", lambda: None)

    def _respond(*args: object, **kwargs: object) -> dict:
        attempts["n"] += 1
        return payload

    monkeypatch.setattr(
        inner,
        "_model_handle",
        type(
            "_H",
            (),
            {"text_to_image": _respond, "image_to_image": _respond},
        )(),
    )

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            if method == "generate":
                await model.generate_image(prompt="p")
            else:
                await model.edit_image(image_url="https://x/s.png", prompt="e")
        usage = manager.get_usage()

    # One provider call, one billing row -- not five and none.
    assert attempts["n"] == 1
    assert usage.media_calls == 1


@pytest.mark.parametrize(
    ("payload", "expected_url", "expected_tokens"),
    [
        ({"data": [{"url": "https://x/1.png"}]}, "https://x/1.png", 0),
        (
            {"data": [{"b64_json": "eA=="}]},
            "data:image/png;base64,eA==",
            0,
        ),
        ({"data": []}, None, 0),
        # A dict response's usage was previously unreachable: the code read it
        # with getattr(), which never finds a dict key, so provider tokens were
        # always 0 for this shape.
        (
            {"data": [{"url": "https://x/2.png"}], "usage": {"input_tokens": 7}},
            "https://x/2.png",
            7,
        ),
    ],
)
@pytest.mark.asyncio
async def test_wellformed_xinference_dict_responses_are_unaffected(
    monkeypatch, payload: dict, expected_url: object, expected_tokens: int
) -> None:
    """Reordering the metering call must not change the success path."""
    from xagent.core.model.image.xinference import XinferenceImageModel

    model = XinferenceImageModel(model_name="sdxl", model_id="xi-cfg")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    monkeypatch.setattr(
        model,
        "_model_handle",
        type("_H", (), {"text_to_image": lambda self, **kw: payload})(),
    )

    with TokenContextManager() as manager:
        result = await model.generate_image(prompt="p", size="1024*1024", n=2)
        entry = manager.get_usage().details[0]

    assert result["image_url"] == expected_url
    assert entry["quantity"] == 2.0
    assert entry["model_id"] == "xi-cfg"
    assert entry["provider_tokens"] == expected_tokens


_XINFERENCE_SHAPES: list = [
    ({"data": [{"url": "https://x/1.png"}]}, True),
    ({"data": [{"b64_json": "eA=="}]}, True),
    ({"data": []}, True),
    ({}, True),
    ({"data": [{"url": "https://x/1.png"}], "usage": {"input_tokens": 5}}, True),
    ({"data": [None]}, False),
    ({"data": ["url-as-a-bare-string"]}, False),
    ({"data": 5}, False),
    ({"data": [[]]}, False),
    ({"data": [7]}, False),
]


@pytest.mark.parametrize(("payload", "should_succeed"), _XINFERENCE_SHAPES)
@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.asyncio
async def test_every_billed_xinference_200_is_metered_exactly_once(
    monkeypatch, instant_retries, payload: dict, should_succeed: bool, method: str
) -> None:
    """One billed 200, one row, one provider call -- whatever its body.

    The contract does not depend on whether the body is usable: the provider
    charged for the call either way. This exists because xinference was the one
    provider my earlier differential sweeps did not cover, and it turned out to
    have neither the metering-before-validation order nor the positional
    classification -- a malformed body cost max_retries charges and zero rows.
    """
    from xagent.core.model.image.adapter import create_image_model

    attempts = {"n": 0}
    model = create_image_model(_image_config("xinference"))
    inner = model._inner
    monkeypatch.setattr(inner, "_ensure_client", lambda: None)

    def _respond(*args: object, **kwargs: object) -> dict:
        attempts["n"] += 1
        return payload

    monkeypatch.setattr(
        inner,
        "_model_handle",
        type("_H", (), {"text_to_image": _respond, "image_to_image": _respond})(),
    )

    with TokenContextManager() as manager:
        raised = None
        try:
            if method == "generate":
                await model.generate_image(prompt="p")
            else:
                await model.edit_image(image_url="https://x/s.png", prompt="e")
        except Exception as error:  # noqa: BLE001
            raised = error
        usage = manager.get_usage()

    assert attempts["n"] == 1, payload
    assert usage.media_calls == 1, payload
    assert (raised is None) is should_succeed, (payload, raised)


@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.asyncio
async def test_unwrapped_providers_meter_and_surface_invalid_responses(
    monkeypatch, method: str
) -> None:
    """`model_service.get_image_models` builds providers with NO retry wrapper.

    Every other test here drives a wrapped provider, so this combination was
    untested: without a wrapper there is no retry policy, and
    `InvalidImageResponseError` must reach the caller directly while the billed
    row is still written. A provider that only behaved correctly under the
    wrapper would look fine everywhere else.
    """
    from xagent.core.model.image import dashscope as ds_mod
    from xagent.core.model.image.base import InvalidImageResponseError

    malformed = {
        "usage": {"input_tokens": 4},
        "output": {"choices": [{"message": {"content": [None]}}]},
    }
    monkeypatch.setattr(
        ds_mod.aiohttp,
        "ClientSession",
        _dashscope_session_factory(malformed, {"n": 0}),
    )
    # Constructed the way model_service does it: directly, unwrapped.
    model = ds_mod.DashScopeImageModel(
        model_name="wanx",
        api_key="k",
        abilities=["generate", "edit"],
        model_id="raw-cfg",
    )

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            if method == "generate":
                await model.generate_image(prompt="p", n=2)
            else:
                await model.edit_image(image_url="https://x/s.png", prompt="e", n=2)
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["model_id"] == "raw-cfg"
    assert usage.details[0]["provider_tokens"] == 4


@pytest.mark.parametrize("provider", ["dashscope", "gemini", "openai", "xinference"])
def test_every_provider_takes_a_model_id_on_the_raw_path(provider: str) -> None:
    """model_service constructs all four directly, by keyword.

    A provider missing the parameter would raise TypeError there and be
    swallowed by that loop's `except Exception`, silently dropping the model
    from the returned dict rather than failing loudly.
    """
    from xagent.core.model.image.dashscope import DashScopeImageModel
    from xagent.core.model.image.gemini import GeminiImageModel
    from xagent.core.model.image.openai import OpenAIImageModel
    from xagent.core.model.image.xinference import XinferenceImageModel

    builders = {
        "dashscope": DashScopeImageModel,
        "gemini": GeminiImageModel,
        "openai": OpenAIImageModel,
        "xinference": XinferenceImageModel,
    }
    model = builders[provider](
        model_name="n",
        api_key="k",
        base_url="https://example.invalid",
        abilities=["generate", "edit"],
        model_id="raw-id",
    )
    assert model.model_id == "raw-id"


class _OpenAIOddResponse:
    """A response whose `data` is truthy but not indexable.

    The typed SDK is meant to make `data` a list or None -- but that is a
    guarantee from another package, and treating it as unreachable is what left
    this path unguarded. A proxy, a gateway, or a future SDK shape can produce
    it.
    """

    def __init__(self, data: object) -> None:
        self.data = data
        self.usage: dict = {}
        self.id = "r"


@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.asyncio
async def test_openai_non_indexable_data_is_billed_once(
    monkeypatch, instant_retries, method: str
) -> None:
    """A billed OpenAI 200 with an unwalkable body: one call, one row.

    Before, `response.data[0]` raised TypeError *before* the metering call, and
    the retry policy treats a bare TypeError as retryable -- so one billed call
    became max_retries charges with no row recorded. Measured 4 and 0.
    """
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.image.base import InvalidImageResponseError

    attempts = {"n": 0}
    model = create_image_model(_image_config("openai"))
    inner = model._inner
    monkeypatch.setattr(inner, "_ensure_client", lambda: None)

    def _respond(**kwargs: object) -> _OpenAIOddResponse:
        attempts["n"] += 1
        return _OpenAIOddResponse(5)

    class _Images:
        async def generate(self, **kwargs: object) -> _OpenAIOddResponse:
            return _respond(**kwargs)

        async def edit(self, **kwargs: object) -> _OpenAIOddResponse:
            return _respond(**kwargs)

    monkeypatch.setattr(inner, "_client", type("_C", (), {"images": _Images()})())
    monkeypatch.setattr("builtins.open", lambda path, mode: _CloseableFile())

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            if method == "generate":
                await model.generate_image(prompt="p")
            else:
                await model.edit_image(image_url="local.png", prompt="e")
        usage = manager.get_usage()

    assert attempts["n"] == 1
    assert usage.media_calls == 1


@pytest.mark.parametrize(
    ("data", "expected_url"),
    [
        ([_StubOpenAIImage()], "https://example.com/1.png"),
        ([], None),
        (None, None),
    ],
)
@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.asyncio
async def test_openai_success_paths_survive_the_reorder(
    monkeypatch, data: object, expected_url: object, method: str
) -> None:
    """Metering before the body walk must not change what the caller gets."""
    from xagent.core.model.image.openai import OpenAIImageModel

    model = OpenAIImageModel(api_key="k", model_id="oa-1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)

    response = _OpenAIOddResponse(data)

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            return response

        async def edit(self, **kwargs: object) -> object:
            return response

    monkeypatch.setattr(model, "_client", type("_C", (), {"images": _Images()})())
    monkeypatch.setattr("builtins.open", lambda path, mode: _CloseableFile())

    with TokenContextManager() as manager:
        if method == "generate":
            result = await model.generate_image(prompt="p")
        else:
            result = await model.edit_image(image_url="local.png", prompt="e")
        entry = manager.get_usage().details[0]

    assert result["image_url"] == expected_url
    assert entry["model_id"] == "oa-1"


@pytest.mark.parametrize(
    ("data", "expect_invalid"),
    [(5, True), ([], False), (None, False)],
)
@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.asyncio
async def test_unwrapped_openai_meters_and_surfaces_invalid_bodies(
    monkeypatch, data: object, expect_invalid: bool, method: str
) -> None:
    """model_service builds OpenAI directly, with no retry wrapper.

    Without a wrapper there is no policy to absorb the error, so the classified
    exception must reach the caller while the billed row is still written. A
    provider that only behaved correctly under the wrapper would look fine in
    every other test here.
    """
    from xagent.core.model.image.base import InvalidImageResponseError
    from xagent.core.model.image.openai import OpenAIImageModel

    model = OpenAIImageModel(api_key="k", model_id="raw-oa")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    response = _OpenAIOddResponse(data)
    response.usage = {"input_tokens": 3}

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            return response

        async def edit(self, **kwargs: object) -> object:
            return response

    monkeypatch.setattr(model, "_client", type("_C", (), {"images": _Images()})())
    monkeypatch.setattr("builtins.open", lambda path, mode: _CloseableFile())

    with TokenContextManager() as manager:
        raised: Exception | None = None
        try:
            if method == "generate":
                await model.generate_image(prompt="p")
            else:
                await model.edit_image(image_url="local.png", prompt="e")
        except Exception as error:  # noqa: BLE001
            raised = error
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["model_id"] == "raw-oa"
    # The billed tokens are recorded even when the body cannot be walked.
    assert usage.details[0]["provider_tokens"] == 3
    assert isinstance(raised, InvalidImageResponseError) is expect_invalid
