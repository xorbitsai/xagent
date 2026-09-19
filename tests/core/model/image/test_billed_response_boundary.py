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


def test_model_service_wraps_image_models_with_the_retry_policy() -> None:
    """The published model carries the policy; the recording provider carries the id.

    Both halves matter and they target different objects. Before this, the
    no-double-billing invariant held on this path only because nothing here
    retried -- not because the policy was applied. The next person to add a
    retry would have silently re-billed every invalid-but-charged 200.

    The id still has to land on the *inner* provider, because that is what calls
    record_image_usage; stamping the wrapper instead leaves the recording layer
    without one and the aggregator groups on the non-unique provider-facing name.
    """
    from xagent.core.model.image.openai import OpenAIImageModel
    from xagent.web.services.model_service import _add_image_model_with_id

    class _Row:
        model_id = "cfg-99"
        model_name = "gpt-image-1"

    provider = OpenAIImageModel(api_key="k")
    models: dict[str, object] = {}
    _add_image_model_with_id(models, provider, _Row())

    published = models["cfg-99"]
    assert published is not provider, "the published model must carry the retry policy"
    assert published._inner is provider
    assert provider.model_id == "cfg-99", "the recording provider must carry the id"
    # The policy is the same one adapter.py installs, so an already-billed
    # invalid response is not retried on either construction path.
    assert published._retry_wrapper.retry_on is retry_image_call


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
