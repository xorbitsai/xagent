"""Unit stability for duration-billed media tools.

The billed unit must depend on the modality, never on how complete the
provider's response happened to be — a price table keyed on (model, unit) is
unusable if the same model sometimes reports "seconds" and sometimes
"requests".
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from xagent.core.model.chat import token_context
from xagent.core.model.chat.token_context import (
    MediaCallType,
    MediaUnit,
    TokenContextManager,
    aggregate_media_usage_by_model,
    get_token_usage,
)
from xagent.core.tools.core import zhipu_web_search as zhipu_web_search_module
from xagent.core.tools.core.exa_web_search import ExaWebSearchCore
from xagent.core.tools.core.media_usage import (
    WEB_SEARCH_MODEL_IDS,
    coerce_duration,
    record_media_seconds,
    record_media_usage,
    record_web_search_usage,
    resolve_billing_model,
)
from xagent.core.tools.core.tavily_web_search import TavilyWebSearchCore
from xagent.core.tools.core.web_search import WebSearchCore
from xagent.core.tools.core.zhipu_web_search import ZhipuWebSearchCore


@pytest.mark.parametrize(
    "value,expected",
    [
        (12.5, 12.5),
        ("3", 3.0),
        (0, None),  # zero is "no usable duration", not a real measurement
        (-5, None),
        (None, None),
        (True, None),  # bool must not sneak through as 1.0
        ("nonsense", None),
    ],
)
def test_coerce_duration(value, expected) -> None:
    assert coerce_duration(value) == expected


def test_record_media_seconds_keeps_unit_stable_when_duration_missing() -> None:
    # Same model, one call with a duration and one without: both must report
    # seconds so billing sees a single line item, not two different units.
    with TokenContextManager() as manager:
        record_media_seconds(30.0, call_type="video", model="veo")
        record_media_seconds(None, call_type="video", model="veo")
        details = manager.get_usage().details

    assert [entry["unit"] for entry in details] == ["seconds", "seconds"]
    # The unmeasured call records 0 seconds and is deliberately KEPT in the
    # rollup: it is the only evidence that a billable provider call happened.
    # It contributes nothing to quantity but still counts toward calls.
    assert [entry["quantity"] for entry in details] == [30.0, 0.0]
    groups = aggregate_media_usage_by_model(details)
    assert len(groups) == 1
    assert groups[0]["unit"] == "seconds"
    assert groups[0]["quantity"] == 30.0
    assert groups[0]["calls"] == 2


def test_record_media_seconds_warns_when_unmeasured(caplog) -> None:
    with caplog.at_level("WARNING"):
        with TokenContextManager():
            record_media_seconds(None, call_type="video", model="veo")
    assert "unmeasured" in caplog.text


def test_record_media_usage_never_raises() -> None:
    # Accounting must never break the underlying media call.
    with TokenContextManager() as manager:
        record_media_usage("video", None, model="m")  # type: ignore[arg-type]
        assert manager.get_usage().media_calls == 1


def test_resolve_billing_model_prefers_real_identities_over_the_fallback() -> None:
    """`_configured_model_id`-style lookups return Optional[str]; passing that
    through str() records a model literally named "None"."""

    class _Model:
        model_name = "elevenlabs-music-v1"

    # None id -> falls back to the provider's own name, not "None".
    assert resolve_billing_model(None, _Model()) == "elevenlabs-music-v1"
    assert resolve_billing_model("", _Model()) == "elevenlabs-music-v1"
    # A real configured id always wins.
    assert resolve_billing_model("cfg-id", _Model()) == "cfg-id"
    # Nothing identifies the model: an explicit fallback, never None/"None".
    assert resolve_billing_model(None, None) == "default"

    # Placeholder names on the model are not treated as identities.
    class _Placeholder:
        model_name = "None"

    assert resolve_billing_model(None, _Placeholder()) == "default"


def test_record_media_usage_swallows_recording_errors(monkeypatch) -> None:
    # The best-effort guarantee is the whole point of this wrapper: a metering
    # bug must never break the media call the user asked for. The existing
    # quantity=None case coerces to 0 and records fine, so it would still pass
    # with the try/except deleted — this one would not.
    import xagent.core.tools.core.media_usage as mu

    def _boom(**kwargs: object) -> None:
        raise RuntimeError("recording backend exploded")

    monkeypatch.setattr(mu, "add_media_usage", _boom)

    with TokenContextManager() as manager:
        # Must not raise.
        mu.record_media_usage(MediaCallType.GENERATE_IMAGE, 1, model="m")
        usage = manager.get_usage()

    # And must leave no partial state behind.
    assert usage.media_calls == 0
    assert usage.details == []


def test_record_media_usage_swallows_invalid_call_type() -> None:
    # A typo'd call_type raises ValueError from the validator; the wrapper drops
    # the record with a warning rather than propagating into the media call.
    # Note "tts" is now a *valid* call_type -- the unit it used to be checked
    # against no longer exists -- so the invalid value has to be a real typo.
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_usage("ttts", 1, model="m")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_record_media_seconds_swallows_invalid_call_type() -> None:
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_seconds(3.0, call_type="not-a-call-type", model="m")
        usage = manager.get_usage()

    assert usage.media_calls == 0
    assert usage.details == []


def test_record_media_seconds_ignores_non_finite_duration() -> None:
    # inf > 0 is True, so without the finiteness guard this would record a
    # non-JSON-serialisable quantity.
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_seconds(
            mu.coerce_duration(float("inf")), model="m", call_type=MediaCallType.ASR
        )
        entry = manager.get_usage().details[0]

    # Treated as "no duration reported": recorded as 0 seconds, unit unchanged.
    assert entry["unit"] == "seconds"
    assert entry["quantity"] == 0.0


def test_record_media_usage_forwards_optional_billing_metadata() -> None:
    # The wrapper must mirror the core primitive's optional fields: passing them
    # previously raised TypeError during argument binding, before the try block.
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_usage(
            MediaCallType.GENERATE_IMAGE,
            1,
            model="m",
            resolution="2K",
            input_tokens=7,
            output_tokens=3,
            tokens_estimated=True,
        )
        entry = manager.get_usage().details[0]

    assert entry["resolution"] == "2K"
    assert entry["provider_tokens"] == 10
    assert entry["tokens_estimated"] is True


def test_resolve_billing_model_survives_raising_descriptor() -> None:
    # Identity resolution runs before record_media_usage's try/except, so a
    # provider whose model_name property raises would otherwise break the call.
    import xagent.core.tools.core.media_usage as mu

    class _Hostile:
        @property
        def model_name(self) -> str:
            raise RuntimeError("descriptor exploded")

        @property
        def model(self) -> str:
            return "real-name"

    assert mu.resolve_billing_model(None, _Hostile()) == "real-name"


@pytest.mark.parametrize(
    "call_type",
    [c for c in MediaCallType if c.unit is not MediaUnit.SECONDS],
)
def test_record_media_seconds_drops_non_duration_modalities(call_type, caplog) -> None:
    # Handing this a modality billed in images or characters used to record a
    # row whose quantity was a duration but whose unit said otherwise -- a
    # silently mispriced row. Dropped and logged instead.
    #
    # Deliberately not raised: producers call this from inside their own
    # `try/except Exception` (music_tool), whose handler returns
    # `success: False`, so raising would turn a generated-and-billed media call
    # into a reported failure.
    import xagent.core.tools.core.media_usage as mu

    with caplog.at_level("ERROR"):
        with TokenContextManager() as manager:
            mu.record_media_seconds(1.5, model="m", call_type=call_type)
            assert manager.get_usage().details == []

    assert "record_media_seconds" in caplog.text
    assert call_type.value in caplog.text


@pytest.mark.parametrize(
    "call_type", [c for c in MediaCallType if c.unit is MediaUnit.SECONDS]
)
def test_record_media_seconds_accepts_every_duration_modality(call_type) -> None:
    import xagent.core.tools.core.media_usage as mu

    with TokenContextManager() as manager:
        mu.record_media_seconds(1.5, model="m", call_type=call_type)
        details = manager.get_usage().details

    assert len(details) == 1
    assert details[0]["unit"] == "seconds"
    assert details[0]["quantity"] == 1.5


@pytest.mark.parametrize(
    "overflowing",
    [
        # float() of a too-large int raises OverflowError, not ValueError. A
        # 400-digit integer literal is exactly what json.loads yields for an
        # oversized JSON number in a provider response.
        10**400,
        -(10**400),
        # These convert fine and are caught by the isfinite guard instead.
        float("inf"),
        float("-inf"),
        float("nan"),
        "1e400",
    ],
)
def test_coerce_duration_survives_non_finite_values(overflowing) -> None:
    # int(float("inf")) raises OverflowError, not ValueError; an ASR provider
    # reporting a non-finite duration must degrade to an unmeasured row rather
    # than break the transcription call it is measuring.
    import xagent.core.tools.core.media_usage as mu

    # None, not 0.0: the helper's documented contract distinguishes "provider
    # reported no duration" from "provider reported zero seconds".
    assert mu.coerce_duration(overflowing) is None

    with TokenContextManager() as manager:
        mu.record_media_seconds(
            mu.coerce_duration(overflowing), model="m", call_type=MediaCallType.ASR
        )
        details = manager.get_usage().details

    assert len(details) == 1
    assert details[0]["quantity"] == 0.0


@pytest.mark.parametrize("empty", [None, ""])
def test_record_media_seconds_leaves_empty_call_types_to_the_primitive(empty) -> None:
    # An absent call_type has no modality to check the unit against, so the
    # duration guard must not claim a mismatch. Validation belongs to
    # add_media_usage, which produces the message listing valid options -- and
    # the wrapper swallows it so a media call is never broken by accounting.
    import xagent.core.tools.core.media_usage as mu

    assert mu._resolved_call_type(empty) is None

    with TokenContextManager() as manager:
        mu.record_media_seconds(1.5, model="m", call_type=empty)

        # Swallowed, not raised, and no half-written row left behind.
        assert manager.get_usage().details == []


# --- Web search: one row per search the provider accepted ---------------------

# Every provider's id lives in one map; the cores name the provider, never the
# id. The platform's own search key pays the provider, so the identity is the
# provider's platform id rather than a user-configured model.
WEB_SEARCH_PROVIDERS = ["zhipu", "tavily", "exa", "google"]


class _ProviderAnswer:
    """The ``httpx`` response shape the three HTTP cores read."""

    def __init__(
        self,
        status_code: int = 200,
        payload: Any = None,
        json_error: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        request = httpx.Request("GET", "https://provider.example.test")
        raise httpx.HTTPStatusError(
            "provider error",
            request=request,
            response=httpx.Response(self.status_code, request=request),
        )

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class _FakeAsyncClient:
    """What ``create_public_async_client`` yields, with one fake ``post``."""

    def __init__(self, answer: Any) -> None:
        self._answer = answer

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self._answer, BaseException):
            raise self._answer
        return self._answer


def _answered(provider: str) -> Any:
    """A 2xx answer in the shape that provider's core really reads."""
    if provider == "zhipu":
        return _ProviderAnswer(
            200,
            {"search_result": [{"title": "t", "link": "https://example.com"}]},
        )
    if provider == "exa":
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    title="t",
                    url="https://example.com",
                    highlights=None,
                    text="",
                    summary="",
                )
            ]
        )
    if provider == "google":
        return _ProviderAnswer(
            200,
            {"items": [{"title": "t", "link": "https://example.com", "snippet": "s"}]},
        )
    return _ProviderAnswer(
        200, {"results": [{"title": "t", "url": "https://example.com", "content": "s"}]}
    )


def _http_mock(answer: Any) -> AsyncMock:
    if isinstance(answer, BaseException):
        return AsyncMock(side_effect=answer)
    return AsyncMock(return_value=answer)


async def _search_once(
    provider: str, answer: Any, *, include_content: bool = False
) -> None:
    """Run one real core search with the provider's transport faked.

    ``answer`` is a response object, or an exception the transport raises.
    """
    if provider == "google":
        with patch("httpx.AsyncClient.get", new=_http_mock(answer)):
            await WebSearchCore(api_key="k", cse_id="c").search(
                "q", include_content=include_content
            )
        return
    if provider == "tavily":
        with patch("httpx.AsyncClient.post", new=_http_mock(answer)):
            await TavilyWebSearchCore(api_key="k").search("q")
        return
    if provider == "zhipu":
        client = _FakeAsyncClient(answer)
        with patch.object(
            zhipu_web_search_module,
            "create_public_async_client",
            lambda **kwargs: client,
        ):
            await ZhipuWebSearchCore(api_key="k").search("q")
        return
    client = MagicMock()
    client.headers = {}
    if isinstance(answer, BaseException):
        client.search.side_effect = answer
        client.search_and_contents.side_effect = answer
    else:
        client.search.return_value = answer
        client.search_and_contents.return_value = answer
    with patch("exa_py.Exa", return_value=client):
        await ExaWebSearchCore(api_key="k").search("q")


@pytest.mark.parametrize(("provider", "model_id"), sorted(WEB_SEARCH_MODEL_IDS.items()))
def test_each_provider_has_one_platform_id(provider: str, model_id: str) -> None:
    assert model_id == f"platform/web-search-{provider}"


@pytest.mark.parametrize(("provider", "model_id"), sorted(WEB_SEARCH_MODEL_IDS.items()))
def test_record_web_search_usage_records_the_provider_id(
    provider: str, model_id: str
) -> None:
    with TokenContextManager() as manager:
        record_web_search_usage(provider)
        details = manager.get_usage().details

    assert len(details) == 1
    assert details[0]["call_type"] == MediaCallType.WEB_SEARCH
    assert details[0]["unit"] == "requests"
    assert details[0]["quantity"] == 1.0
    assert details[0]["model_id"] == model_id


def test_an_unknown_web_search_provider_records_nothing() -> None:
    with TokenContextManager() as manager:
        record_web_search_usage("not-a-provider")
        details = manager.get_usage().details

    assert details == []


@pytest.mark.parametrize("provider", WEB_SEARCH_PROVIDERS)
@pytest.mark.asyncio
async def test_a_provider_answer_records_exactly_one_row(provider: str) -> None:
    with TokenContextManager() as manager:
        await _search_once(provider, _answered(provider))
        details = manager.get_usage().details

    assert len(details) == 1
    assert details[0]["call_type"] == "web_search"
    assert details[0]["unit"] == "requests"
    assert details[0]["quantity"] == 1.0
    assert details[0]["model_id"] == WEB_SEARCH_MODEL_IDS[provider]


def _provider_failures() -> list[Any]:
    """One case per auth/rate/network failure, in each provider's terms."""
    cases: list[Any] = []
    for provider in WEB_SEARCH_PROVIDERS:
        for status in (401, 403, 429):
            answer: Any
            if provider == "exa":
                # The SDK reports these as its own exception, not a response.
                answer = RuntimeError(f"Exa answered {status}")
            else:
                answer = _ProviderAnswer(status)
            cases.append(pytest.param(provider, answer, id=f"{provider}-{status}"))
        cases.append(
            pytest.param(
                provider, httpx.ConnectError("network down"), id=f"{provider}-network"
            )
        )
    return cases


@pytest.mark.parametrize(("provider", "answer"), _provider_failures())
@pytest.mark.asyncio
async def test_a_failed_search_records_nothing(provider: str, answer: Any) -> None:
    # A missing key, a timeout, an auth failure or an HTTP error means the
    # provider never billed the request, so there is no call to record. The
    # search still reports the failure to the model exactly as before.
    with TokenContextManager() as manager:
        with pytest.raises(ValueError):
            await _search_once(provider, answer)
        details = manager.get_usage().details

    assert details == []


def _unreadable_answer(provider: str) -> Any:
    """A 2xx whose body cannot be turned into results."""
    if provider == "exa":
        return SimpleNamespace(results=None)
    return _ProviderAnswer(200, json_error=ValueError("unusable body"))


@pytest.mark.parametrize("provider", WEB_SEARCH_PROVIDERS)
@pytest.mark.asyncio
async def test_a_2xx_with_an_unreadable_body_is_still_recorded(provider: str) -> None:
    # The provider billed the request when it answered 2xx, so the row is
    # written before the body is parsed. A parse failure must not lose it.
    with TokenContextManager() as manager:
        with pytest.raises((TypeError, ValueError)):
            await _search_once(provider, _unreadable_answer(provider))
        details = manager.get_usage().details

    assert len(details) == 1
    assert details[0]["model_id"] == WEB_SEARCH_MODEL_IDS[provider]


@pytest.mark.asyncio
async def test_google_records_before_fetching_page_content() -> None:
    # include_content fetches a page per result after Google answered. A page
    # fetch that fails cannot un-bill the search.
    answer = _answered("google")
    with patch(
        "xagent.core.tools.core.web_search.WebContentFetcher.fetch_text",
        new=AsyncMock(side_effect=RuntimeError("page fetch failed")),
    ):
        with TokenContextManager() as manager:
            with pytest.raises(ValueError):
                await _search_once("google", answer, include_content=True)
            details = manager.get_usage().details

    assert len(details) == 1
    assert details[0]["model_id"] == "platform/web-search-google"


def test_search_metering_without_a_run_context_does_not_raise() -> None:
    # A search can run with nothing bound to report its usage. The row then goes
    # to the unbound usage, which is what every other media producer does, and
    # the search itself is unaffected.
    token_context.token_context.set(None)
    try:
        record_web_search_usage("tavily")
        usage = get_token_usage()
    finally:
        token_context.token_context.set(None)

    assert usage.media_calls == 1
