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

from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.image.base import (
    InvalidImageResponseError,
    call_billed_endpoint,
    decode_billed_body,
    decode_billed_body_async,
    image_url_from_item,
    retry_image_call,
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
