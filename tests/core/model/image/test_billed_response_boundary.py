"""The billing boundary around an already-charged image response.

A provider that answers 200 has billed the caller. Everything from the first
read of that body onwards is therefore an *invalid response* -- never a
retryable failure -- because retrying buys another charge for an outcome that
cannot change.

The boundary was introduced for failures that happen while *walking* a decoded
body. These tests pin the earlier half that was missing: the decode itself, and
the clients that decode internally. Left outside, a body that would not parse
raised a retryable error before any metering ran, so the call was billed on
every attempt and recorded on none -- worse than the multiplied-metering bug the
boundary was built to fix, because that spend leaves no trace at all.

Each test asserts the same two numbers: how many times the provider was called,
and how many billing rows exist afterwards.
"""

import asyncio
import json

import pytest
import requests
from openai import APIResponseValidationError

from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.image.adapter import retry_image_call
from xagent.core.model.image.base import (
    InvalidImageResponseError,
    call_billed_endpoint,
    call_billed_endpoint_async,
    decode_billed_body,
    decode_billed_body_async,
    image_url_from_item,
)
from xagent.core.model.image.usage import record_image_usage


def _run_with_retries(call, max_retries: int = 4):
    """Drive a call through the production retry policy, counting attempts."""
    attempts = 0
    last: Exception | None = None
    for _ in range(max_retries):
        attempts += 1
        try:
            return call(), attempts
        except Exception as error:  # noqa: BLE001
            last = error
            if not retry_image_call(error):
                break
    raise last from None


# --- the decode step itself -------------------------------------------------


def test_undecodable_billed_body_is_not_retried() -> None:
    # An edge proxy answering 200 with HTML is the everyday shape of this.
    def call():
        return decode_billed_body(
            lambda: json.loads("<html>502 Bad Gateway</html>"), "ctx"
        )

    with pytest.raises(InvalidImageResponseError):
        _run_with_retries(call)


def test_undecodable_billed_body_costs_exactly_one_provider_call() -> None:
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        return decode_billed_body(lambda: json.loads("not json"), "ctx")

    with pytest.raises(InvalidImageResponseError):
        _run_with_retries(call)
    # Before the fix this was 4: a JSONDecodeError reached the retry predicate,
    # and each attempt was separately billed by the provider.
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_undecodable_billed_body_async_is_not_retried() -> None:
    calls = {"n": 0}

    async def decode():
        # What aiohttp raises when a 200 is not the declared content type.
        raise ValueError("Attempt to decode JSON with unexpected mimetype")

    async def call():
        calls["n"] += 1
        return await decode_billed_body_async(decode, "ctx")

    attempts = 0
    last: Exception | None = None
    for _ in range(4):
        attempts += 1
        try:
            await call()
        except Exception as error:  # noqa: BLE001
            last = error
            if not retry_image_call(error):
                break
    assert isinstance(last, InvalidImageResponseError)
    assert calls["n"] == 1


def test_a_decodable_body_is_returned_unchanged() -> None:
    assert decode_billed_body(lambda: json.loads('{"a": 1}'), "ctx") == {"a": 1}


# --- clients that decode internally ----------------------------------------
#
# The xinference client raises on a non-200 and then calls response.json()
# itself, so a billed-but-undecodable body surfaces from the call rather than
# from a body walk. The split is by type, and it is exact: every decode failure
# is a ValueError, and no transport failure is one.


@pytest.mark.parametrize(
    "error,retryable",
    [
        (json.JSONDecodeError("x", "<html>", 0), False),
        (requests.exceptions.JSONDecodeError("x", "", 0), False),
        (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"), False),
        (requests.exceptions.Timeout("timed out"), True),
        (requests.exceptions.ConnectionError("connection reset"), True),
        (RuntimeError("Failed to create the images"), True),
    ],
    ids=["json", "requests-json", "unicode", "timeout", "connection", "non-200"],
)
def test_client_decoded_endpoint_splits_billed_from_transport(
    error: Exception, retryable: bool
) -> None:
    def call():
        raise error

    with pytest.raises(Exception) as caught:  # noqa: PT011
        call_billed_endpoint(call, "ctx")
    assert retry_image_call(caught.value) is retryable
    # A billed body is reclassified; a transport failure keeps its own type so
    # genuinely separate calls stay retryable and separately metered.
    assert isinstance(caught.value, InvalidImageResponseError) is not retryable


def test_client_decoded_endpoint_passes_a_good_result_through() -> None:
    assert call_billed_endpoint(lambda: {"data": [1]}, "ctx") == {"data": [1]}


@pytest.mark.asyncio
async def test_openai_response_validation_is_treated_as_billed() -> None:
    # The OpenAI SDK reports a 200 whose body fails validation with its own
    # error type, which is not a ValueError -- so the provider names it
    # explicitly rather than the boundary guessing.
    async def call():
        raise APIResponseValidationError(
            response=requests.Response(), body=None, message="bad body"
        )

    with pytest.raises(InvalidImageResponseError):
        await call_billed_endpoint_async(
            call, "ctx", also_billed=(APIResponseValidationError,)
        )


@pytest.mark.asyncio
async def test_openai_connection_errors_stay_retryable() -> None:
    async def call():
        raise requests.exceptions.ConnectionError("reset")

    with pytest.raises(requests.exceptions.ConnectionError) as caught:
        await call_billed_endpoint_async(
            call, "ctx", also_billed=(APIResponseValidationError,)
        )
    assert retry_image_call(caught.value) is True


# --- identity ---------------------------------------------------------------


@pytest.mark.parametrize("placeholder", ["", "none", "None", "null", "default"])
def test_placeholder_identities_are_never_billed(placeholder: str) -> None:
    # The aggregator groups on `model_id or model`, so a configured id of
    # "default" would become a billing line that unrelated models merge into.
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"completion_tokens": 7}},
            model_name=placeholder,
            model_id=placeholder,
            call_type="generate_image",
        )
        usage = manager.get_usage()

    entry = usage.details[0]
    assert entry["model"] == ""
    assert entry["model_id"] == ""
    # The row is still written: the call happened and was billed, and dropping
    # it would report "no media used" for a run that really did use some.
    assert usage.media_calls == 1


def test_a_real_identity_is_preserved() -> None:
    with TokenContextManager() as manager:
        record_image_usage(
            {"usage": {"completion_tokens": 7}},
            model_name="gpt-image-1",
            model_id="cfg-42",
            call_type="generate_image",
        )
        usage = manager.get_usage()

    assert usage.details[0]["model"] == "gpt-image-1"
    assert usage.details[0]["model_id"] == "cfg-42"


# --- shared url extraction --------------------------------------------------


class _Item:
    def __init__(self, url=None, b64_json=None):
        self.url = url
        self.b64_json = b64_json


def test_a_null_url_falls_through_to_inline_data() -> None:
    # hasattr read `url=None` as present and returned the string "None", which
    # image_tool then tried to download.
    assert image_url_from_item(_Item(url=None, b64_json="QUJD")) == (
        "data:image/png;base64,QUJD"
    )


def test_a_real_url_wins_over_inline_data() -> None:
    assert image_url_from_item(_Item(url="https://x/y.png", b64_json="QUJD")) == (
        "https://x/y.png"
    )


def test_an_item_with_neither_is_absent() -> None:
    assert image_url_from_item(_Item()) is None


# --- model_service construction path ----------------------------------------


@pytest.mark.asyncio
async def test_model_service_publishes_a_model_that_retries_and_meters() -> None:
    """Asserted through the published model's own methods, not its internals.

    An earlier version checked `published._retry_wrapper.retry_on is
    retry_image_call`, which stays green if the wrapper stops delegating
    generate_image, uses the wrong attempt bound, or skips recording entirely.
    What matters is observable: a transient failure is retried, an
    already-billed one is not, and the row lands either way.
    """
    from xagent.core.model.image.openai import OpenAIImageModel
    from xagent.web.services.model_service import _add_image_model_with_id

    class _Row:
        model_id = "cfg-99"
        model_name = "gpt-image-1"
        max_retries = 3

    attempts = {"n": 0}

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            attempts["n"] += 1
            raise APIResponseValidationError(
                response=requests.Response(), body=None, message="bad body"
            )

    provider = OpenAIImageModel(api_key="k")
    provider._ensure_client = lambda: None  # type: ignore[method-assign]
    provider._client = type("_C", (), {"images": _Images()})()  # type: ignore[assignment]

    models: dict[str, object] = {}
    _add_image_model_with_id(models, provider, _Row())
    published = models["cfg-99"]

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await published.generate_image(prompt="p", n=2)
        usage = manager.get_usage()

    # Billed once, so called once -- the policy is applied, not merely attached.
    assert attempts["n"] == 1
    # And the charge left a row, carrying the configured id rather than the
    # provider-facing name. The id has to reach the *recording* provider, which
    # is the inner one; a wrapper-only stamp would bill under "gpt-image-1".
    assert usage.media_calls == 1
    assert usage.details[0]["model_id"] == "cfg-99"


@pytest.mark.asyncio
async def test_the_published_model_retries_a_transient_failure() -> None:
    # The other half: the policy must still retry what is genuinely retryable,
    # or narrowing it would have quietly disabled retries altogether.
    from xagent.core.model.image.openai import OpenAIImageModel
    from xagent.web.services.model_service import _add_image_model_with_id

    class _Row:
        model_id = "cfg-transient"
        model_name = "gpt-image-1"
        max_retries = 3

    attempts = {"n": 0}

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            attempts["n"] += 1
            raise requests.exceptions.ConnectionError("reset")

    provider = OpenAIImageModel(api_key="k")
    provider._ensure_client = lambda: None  # type: ignore[method-assign]
    provider._client = type("_C", (), {"images": _Images()})()  # type: ignore[assignment]

    models: dict[str, object] = {}
    _add_image_model_with_id(models, provider, _Row())

    with TokenContextManager():
        with pytest.raises(requests.exceptions.ConnectionError):
            await models["cfg-transient"].generate_image(prompt="p")

    assert attempts["n"] == 3, "a transient failure must exhaust the attempt bound"


def test_the_wrapped_model_still_answers_as_an_image_model() -> None:
    # Wrapping must not change the published surface: callers read `abilities`
    # and `model_id` straight off whatever get_image_models returned.
    from xagent.core.model.image.openai import OpenAIImageModel
    from xagent.web.services.model_service import _add_image_model_with_id

    class _Row:
        model_id = "cfg-1"
        model_name = "gpt-image-1"

    models: dict[str, object] = {}
    _add_image_model_with_id(models, OpenAIImageModel(api_key="k"), _Row())
    published = models["cfg-1"]
    assert published.model_id == "cfg-1"
    assert "generate" in published.abilities


# --- a classified response must still be metered ----------------------------
#
# Reclassifying a billed failure stops the repeated charges, but the one real
# charge still has to leave a row. Recording only on the success path would
# trade multiplied metering for invisible spend, which is worse: the money is
# gone either way, and this way nothing shows it.


@pytest.mark.asyncio
async def test_openai_meters_a_response_the_sdk_refused_to_return() -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    model = OpenAIImageModel(api_key="k", model_id="cfg-oa")
    model._ensure_client = lambda: None  # type: ignore[method-assign]

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            raise APIResponseValidationError(
                response=requests.Response(), body=None, message="bad body"
            )

    model._client = type("_C", (), {"images": _Images()})()  # type: ignore[assignment]

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await model.generate_image(prompt="p", n=2)
        usage = manager.get_usage()

    assert usage.media_calls == 1
    row = usage.details[0]
    assert row["model_id"] == "cfg-oa"
    # The requested count, and no tokens: the SDK raised instead of handing
    # over a usage payload, so the call is billed but unmeasured.
    assert row["quantity"] == 2.0
    assert row["provider_tokens"] == 0


@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.asyncio
async def test_xinference_meters_a_body_the_client_could_not_decode(
    method: str,
) -> None:
    from xagent.core.model.image.xinference import XinferenceImageModel

    model = XinferenceImageModel(model_name="sd", base_url="http://x", model_id="xi-1")
    model._ensure_client = lambda: None  # type: ignore[method-assign]
    model._abilities = ["generate", "edit"]

    class _Handle:
        def text_to_image(self, **kwargs: object) -> object:
            raise json.JSONDecodeError("x", "<html>", 0)

        def image_to_image(self, **kwargs: object) -> object:
            raise json.JSONDecodeError("x", "<html>", 0)

    model._model_handle = _Handle()

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            if method == "generate":
                await model.generate_image(prompt="p", n=3)
            else:
                await model.edit_image(image_url="u", prompt="e", n=3)
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 3.0
    assert usage.details[0]["model_id"] == "xi-1"


# --- the cross-provider matrix ----------------------------------------------
#
# Every previous round fixed the providers the review named and left the others
# behind: round one metered after the body walk, round two closed the decode for
# all four but only added the record-before-raise half to two of them. These
# tests assert the property across all four at once, so the next gap fails here
# rather than in review.


def _undecodable_gemini(monkeypatch):
    import httpx

    class _Response:
        status_code = 200
        text = "<html>502</html>"

        def json(self):
            raise json.JSONDecodeError("x", "<html>", 0)

        def raise_for_status(self):
            return None

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return _Response()

    from xagent.core.model.image.gemini import GeminiImageModel

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _Client())
    model = GeminiImageModel(model_name="gemini-3-pro-image-preview-2k", api_key="k")
    return model, lambda: model.generate_image(prompt="p")


def _undecodable_dashscope(monkeypatch):
    from xagent.core.model.image import dashscope as ds

    class _Response:
        status = 200

        async def json(self):
            raise ValueError("Attempt to decode JSON with unexpected mimetype")

        async def text(self):
            return "<html>"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(ds.aiohttp, "ClientSession", lambda *a, **k: _Session())
    model = ds.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"], model_id="d1"
    )
    return model, lambda: model.generate_image(prompt="p", n=2)


def _unusable_openai(monkeypatch):
    from xagent.core.model.image.openai import OpenAIImageModel

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            raise APIResponseValidationError(
                response=requests.Response(), body=None, message="bad body"
            )

    model = OpenAIImageModel(api_key="k", model_id="o1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._client = type("_C", (), {"images": _Images()})()
    return model, lambda: model.generate_image(prompt="p", n=2)


def _undecodable_xinference(monkeypatch):
    from xagent.core.model.image.xinference import XinferenceImageModel

    class _Handle:
        def text_to_image(self, **kwargs: object) -> object:
            raise json.JSONDecodeError("x", "<html>", 0)

    model = XinferenceImageModel(model_name="sd", base_url="http://x", model_id="x1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._abilities = ["generate", "edit"]
    model._model_handle = _Handle()
    return model, lambda: model.generate_image(prompt="p", n=2)


@pytest.mark.parametrize(
    "setup",
    [
        _undecodable_gemini,
        _undecodable_dashscope,
        _unusable_openai,
        _undecodable_xinference,
    ],
    ids=["gemini", "dashscope", "openai", "xinference"],
)
@pytest.mark.asyncio
async def test_every_provider_meters_a_billed_but_unreadable_response(
    monkeypatch, setup
) -> None:
    """One charge, one row — on all four providers.

    Classifying the failure as non-retryable is only half the fix. Without the
    row, the single real charge leaves no trace at all, which is worse than the
    multiplied metering the classification prevents.
    """
    _, call = setup(monkeypatch)

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await call()
        usage = manager.get_usage()

    assert usage.media_calls == 1
    assert len(usage.details) == 1


@pytest.mark.parametrize(
    "setup",
    [
        _undecodable_gemini,
        _undecodable_dashscope,
        _unusable_openai,
        _undecodable_xinference,
    ],
    ids=["gemini", "dashscope", "openai", "xinference"],
)
@pytest.mark.asyncio
async def test_a_billed_unreadable_response_is_never_retried(
    monkeypatch, setup
) -> None:
    # The classification half, asserted on the same four so the two properties
    # cannot drift apart per provider.
    _, call = setup(monkeypatch)
    with TokenContextManager():
        with pytest.raises(InvalidImageResponseError) as caught:
            await call()
    assert retry_image_call(caught.value) is False


# --- edit_image honours the size it was asked for ---------------------------
#
# generate_image has always applied `resolution > width+height > size`;
# edit_image read `size` alone, so an edit requested at 1920x1080 was rendered
# and billed at the stale default. Asserted per provider, and on both the value
# sent to the provider and the value recorded, because billing at a size the
# request never used is the half that costs money.


@pytest.mark.asyncio
async def test_openai_edit_honours_resolution(monkeypatch) -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    sent: dict = {}

    class _Images:
        async def edit(self, **kwargs: object) -> object:
            sent.update(kwargs)
            return type("R", (), {"data": [], "usage": None, "id": "x"})()

    model = OpenAIImageModel(api_key="k", model_id="o1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._client = type("_C", (), {"images": _Images()})()
    monkeypatch.setattr(
        "builtins.open",
        lambda *a, **k: type(
            "F", (), {"close": lambda s: None, "read": lambda s: b""}
        )(),
    )

    with TokenContextManager() as manager:
        await model.edit_image(image_url="a.png", prompt="p", resolution="1920x1080")
        usage = manager.get_usage()

    assert sent["size"] == "1920x1080"
    assert usage.details[0]["resolution"] == "1920x1080"


@pytest.mark.asyncio
async def test_xinference_edit_honours_width_and_height(monkeypatch) -> None:
    from xagent.core.model.image.xinference import XinferenceImageModel

    sent: dict = {}

    class _Handle:
        def image_to_image(self, **kwargs: object) -> object:
            sent.update(kwargs)
            return {"data": [{"url": "https://x/y.png"}]}

    model = XinferenceImageModel(model_name="sd", base_url="http://x", model_id="x1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._abilities = ["generate", "edit"]
    model._model_handle = _Handle()

    with TokenContextManager() as manager:
        await model.edit_image(image_url="u", prompt="p", width=1920, height=1080)
        usage = manager.get_usage()

    # Xinference speaks W*H, so the shared helper's WxH is normalised on the way
    # out -- and the recorded key matches what that provider is priced on.
    assert sent["size"] == "1920*1080"
    assert usage.details[0]["resolution"] == "1920*1080"


@pytest.mark.asyncio
async def test_dashscope_edit_honours_resolution(monkeypatch) -> None:
    from xagent.core.model.image import dashscope as ds

    sent: dict = {}

    class _Response:
        status = 200

        async def json(self):
            return {
                "usage": {},
                "output": {
                    "choices": [
                        {"message": {"content": [{"image": "https://x/e.png"}]}}
                    ]
                },
            }

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, headers=None, json=None, **kwargs):
            sent.update((json or {}).get("parameters", {}))
            return _Response()

    monkeypatch.setattr(ds.aiohttp, "ClientSession", lambda *a, **k: _Session())
    model = ds.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"], model_id="d1"
    )
    monkeypatch.setattr(model, "_convert_image_to_base64", lambda x: "b64")

    with TokenContextManager() as manager:
        await model.edit_image(
            image_url="https://x/s.png", prompt="p", resolution="1920x1080"
        )
        usage = manager.get_usage()

    assert sent["size"] == "1920x1080"
    assert usage.details[0]["resolution"] == "1920x1080"


def test_a_non_string_size_does_not_become_a_billing_key() -> None:
    from xagent.core.model.image.base import resolve_requested_size

    # An int 2048 stringified to "2048" would join no price table and read as a
    # real tier. width+height is the typed way to say the same thing.
    assert resolve_requested_size(2048, default="") == ""
    assert resolve_requested_size(None, width=1920, height=1080) == "1920x1080"
    assert resolve_requested_size("1024x1024") == "1024x1024"
    # Precedence, in one line each.
    assert resolve_requested_size("a", resolution="b", width=1, height=2) == "b"
    assert resolve_requested_size("a", width=1, height=2) == "1x2"


# --- the public factory, and the edit half of the wiring ---------------------
#
# The tests above construct through _add_image_model_with_id. That leaves the
# factory itself uncovered: a published model could stop being wrapped, or the
# edit method could lose its wrapper, without any of them noticing.


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *args, **kwargs):
        return _FakeQuery(self._rows)


class _Row:
    id = 1
    model_id = "cfg-public"
    model_name = "gpt-image-1"
    model_provider = "openai"
    api_key = "sk-test"
    base_url = "https://api.openai.com/v1"
    abilities = ["generate", "edit"]
    category = "image"
    is_active = True
    max_retries = 3


@pytest.mark.asyncio
async def test_the_public_factory_publishes_a_wrapped_metering_model(
    monkeypatch,
) -> None:
    from xagent.web.services import model_service

    monkeypatch.setattr(
        model_service, "_is_model_visible_to_user", lambda *a, **k: True
    )

    models = model_service.get_image_models(_FakeSession([_Row()]), user_id=1)
    published = models["cfg-public"]

    attempts = {"n": 0}

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            attempts["n"] += 1
            raise APIResponseValidationError(
                response=requests.Response(), body=None, message="bad body"
            )

    # Reach past the wrapper to stub the transport only; everything asserted
    # below goes through the published object's own method.
    published._inner._ensure_client = lambda: None
    published._inner._client = type("_C", (), {"images": _Images()})()

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await published.generate_image(prompt="p", n=2)
        usage = manager.get_usage()

    assert attempts["n"] == 1, "the factory must apply the retry policy"
    assert usage.media_calls == 1
    assert usage.details[0]["model_id"] == "cfg-public"


@pytest.mark.asyncio
async def test_the_published_edit_method_is_wrapped_and_meters(monkeypatch) -> None:
    # edit_image is in retry_methods too; a wiring change that covered only
    # generate_image would otherwise go unnoticed.
    from xagent.web.services import model_service

    monkeypatch.setattr(
        model_service, "_is_model_visible_to_user", lambda *a, **k: True
    )
    monkeypatch.setattr(
        "builtins.open",
        lambda *a, **k: type(
            "F", (), {"close": lambda s: None, "read": lambda s: b""}
        )(),
    )

    models = model_service.get_image_models(_FakeSession([_Row()]), user_id=1)
    published = models["cfg-public"]

    attempts = {"n": 0}

    class _Images:
        async def edit(self, **kwargs: object) -> object:
            attempts["n"] += 1
            raise APIResponseValidationError(
                response=requests.Response(), body=None, message="bad body"
            )

    published._inner._ensure_client = lambda: None
    published._inner._client = type("_C", (), {"images": _Images()})()

    with TokenContextManager() as manager:
        with pytest.raises(InvalidImageResponseError):
            await published.edit_image(image_url="a.png", prompt="p", n=2)
        usage = manager.get_usage()

    assert attempts["n"] == 1
    assert usage.media_calls == 1
    assert usage.details[0]["call_type"] == "edit_image"


# --- cancellation: recorded only where the charge is certain ----------------
#
# The two providers catch cancellation at different points, and the difference
# is the whole rule. DashScope's catch wraps `response.json()` on a 200 already
# in hand, so the charge has happened. OpenAI's wraps the entire SDK coroutine,
# where a cancellation during DNS, connect, upload or the header wait happens
# before the request is accepted — recording there invents a charge.


@pytest.mark.asyncio
async def test_openai_does_not_record_a_cancelled_call(monkeypatch) -> None:
    from xagent.core.model.image.openai import OpenAIImageModel

    class _Images:
        async def generate(self, **kwargs: object) -> object:
            # Stands for a cancellation anywhere in the SDK coroutine, including
            # before the request reaches OpenAI.
            raise asyncio.CancelledError()

    model = OpenAIImageModel(api_key="k", model_id="o1")
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    model._client = type("_C", (), {"images": _Images()})()

    with TokenContextManager() as manager:
        with pytest.raises(asyncio.CancelledError):
            await model.generate_image(prompt="p", n=2)
        usage = manager.get_usage()

    # No row: the outcome is unknown, and inventing a charge is worse than
    # leaving an ambiguous one to reconciliation (xorbitsai/xagent#2513).
    assert usage.media_calls == 0


@pytest.mark.asyncio
async def test_dashscope_records_a_cancelled_decode(monkeypatch) -> None:
    from xagent.core.model.image import dashscope as ds

    class _Response:
        status = 200

        async def json(self):
            # Cancelled while decoding a 200 that DashScope has already billed.
            raise asyncio.CancelledError()

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(ds.aiohttp, "ClientSession", lambda *a, **k: _Session())
    model = ds.DashScopeImageModel(
        model_name="wanx", api_key="k", abilities=["generate", "edit"], model_id="d1"
    )

    with TokenContextManager() as manager:
        with pytest.raises(asyncio.CancelledError):
            await model.generate_image(prompt="p", n=2)
        usage = manager.get_usage()

    # One row: the 200 arrived, so the charge is certain even though the body
    # was never read. Re-raised unchanged so the caller's shutdown still works.
    assert usage.media_calls == 1
    assert usage.details[0]["quantity"] == 2.0
