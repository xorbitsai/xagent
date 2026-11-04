"""``get_image_models`` publishes providers wrapped in the retry proxy.

This is the live production path: ``image_tool`` -> ``web/tools/config.py`` ->
``model_service.get_image_models``. Before this, the four providers it built
were published unwrapped, so a transient network failure was never retried and
the retry policy that ``get_image_model_instance`` already applies on the other
construction path did not apply here.

Tests go through the public ``get_image_models`` only -- never
``published._inner`` -- with a fake DB session and a fake OpenAI transport, so
what is asserted is exactly what a real caller gets back.
"""

from __future__ import annotations

from typing import Any

import pytest

from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.image import openai as openai_provider
from xagent.web.services import model_service


class _FakeQuery:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def filter(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def all(self) -> list[Any]:
        return self._rows


class _FakeSession:
    """Minimal stand-in for the SQLAlchemy Session ``get_image_models`` reads."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def query(self, *args: Any, **kwargs: Any) -> _FakeQuery:
        return _FakeQuery(self._rows)


def _db_row(*, model_id: str = "row-1", max_retries: Any = 3) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        id=1,
        model_id=model_id,
        category="image",
        is_active=True,
        model_provider="openai",
        model_name="gpt-image-1",
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        abilities=["generate", "edit"],
        description=None,
        max_retries=max_retries,
    )


class _FakeImages:
    """Queues one effect per call: an exception to raise, or a value to return."""

    def __init__(self, effects: list[Any]) -> None:
        self._effects = list(effects)
        self.calls: list[str] = []

    async def _next(self, name: str) -> Any:
        self.calls.append(name)
        effect = self._effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect

    async def generate(self, **kwargs: Any) -> Any:
        return await self._next("generate")

    async def edit(self, **kwargs: Any) -> Any:
        return await self._next("edit")


class _StubOpenAIImage:
    url = "https://example.com/1.png"
    b64_json = None


class _StubOpenAIResponse:
    data = [_StubOpenAIImage()]
    usage: dict = {}
    id = "req-1"


def _patch_openai_transport(
    monkeypatch: pytest.MonkeyPatch, effects: list[Any]
) -> _FakeImages:
    """Replace the SDK client class so every instance built during the test
    shares one fake transport, without ever reaching into the retry wrapper's
    ``_inner``."""
    images = _FakeImages(effects)

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.images = images

    monkeypatch.setattr(openai_provider, "AsyncOpenAI", _FakeAsyncOpenAI)
    return images


@pytest.fixture(autouse=True)
def _no_visibility_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        model_service, "_is_model_visible_to_user", lambda *a, **k: True
    )


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralize the 200ms exponential backoff so retry tests don't sleep."""
    from xagent.core.retry.strategy import ExponentialBackoff

    monkeypatch.setattr(ExponentialBackoff, "get_delay", lambda self, attempt: 0.0)


def _published_model(rows: list[Any], monkeypatch: pytest.MonkeyPatch) -> Any:
    db = _FakeSession(rows)
    models = model_service.get_image_models(db, user_id=1)
    assert len(models) == 1
    return next(iter(models.values()))


@pytest.mark.asyncio
async def test_transient_generate_failure_is_retried_max_retries_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _db_row(max_retries=3)
    images = _patch_openai_transport(
        monkeypatch, [RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")]
    )
    model = _published_model([row], monkeypatch)

    with pytest.raises(RuntimeError):
        await model.generate_image(prompt="p")

    assert images.calls == ["generate", "generate", "generate"]


@pytest.mark.asyncio
async def test_transient_edit_failure_is_retried_max_retries_times(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    row = _db_row(max_retries=2)
    images = _patch_openai_transport(
        monkeypatch, [RuntimeError("boom"), RuntimeError("boom")]
    )
    model = _published_model([row], monkeypatch)

    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(RuntimeError):
        await model.edit_image(image_url=str(image_path), prompt="edit")

    assert images.calls == ["edit", "edit"]


@pytest.mark.asyncio
async def test_billed_invalid_body_is_not_retried_and_bills_the_row_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests
    from openai import APIResponseValidationError

    row = _db_row(model_id="row-billed", max_retries=5)
    validation_error = APIResponseValidationError(
        response=requests.Response(), body=None, message="bad body"
    )
    images = _patch_openai_transport(monkeypatch, [validation_error])
    model = _published_model([row], monkeypatch)

    with TokenContextManager() as manager:
        with pytest.raises(Exception):
            await model.generate_image(prompt="p")
        media_rows = [
            d for d in manager.get_usage().details if d.get("type") == "media"
        ]

    # Exactly one attempt: the billed-invalid classification is not retryable.
    assert images.calls == ["generate"]
    # Exactly one row, billed under the configured row id, not the model name.
    assert len(media_rows) == 1
    assert media_rows[0]["model_id"] == "row-billed"
    assert media_rows[0]["model_id"] != row.model_name


@pytest.mark.asyncio
async def test_max_retries_none_falls_back_to_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _db_row(max_retries=None)
    images = _patch_openai_transport(
        monkeypatch, [RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")]
    )
    model = _published_model([row], monkeypatch)

    with pytest.raises(RuntimeError):
        await model.generate_image(prompt="p")

    assert images.calls == ["generate", "generate", "generate"]


@pytest.mark.asyncio
async def test_negative_max_retries_is_clamped_to_one_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row's negative max_retries must not reach RetryWrapper unclamped.

    ``-1`` is truthy, so ``getattr(db_model, "max_retries", 3) or 3`` passes it
    straight through instead of substituting the fallback (0 would already be
    caught by ``or 3``, since 0 is falsy -- only a negative value survives).
    ``range(max_retries)`` is then empty for anything <= 0, so RetryWrapper
    skips the target entirely -- not even one attempt -- and raises a
    synthetic ``RuntimeError("Retry failed with no exception")``. The caller
    never sees the real failure, and a billed-invalid response is never
    recorded at all. Clamped to at least 1, the real error and exactly one
    attempt come through instead.
    """
    row = _db_row(max_retries=-1)
    images = _patch_openai_transport(monkeypatch, [RuntimeError("boom")])
    model = _published_model([row], monkeypatch)

    with pytest.raises(RuntimeError, match="boom"):
        await model.generate_image(prompt="p")

    assert images.calls == ["generate"]


@pytest.mark.asyncio
async def test_falsy_zero_max_retries_still_falls_back_to_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 is falsy, so it was never affected by the clamp bug: `0 or 3` already
    substitutes the fallback before the clamp ever sees the value. Pinned
    separately from the negative case above so the two are not conflated."""
    row = _db_row(max_retries=0)
    images = _patch_openai_transport(
        monkeypatch, [RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")]
    )
    model = _published_model([row], monkeypatch)

    with pytest.raises(RuntimeError):
        await model.generate_image(prompt="p")

    assert images.calls == ["generate", "generate", "generate"]


@pytest.mark.asyncio
async def test_negative_max_retries_still_bills_the_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests
    from openai import APIResponseValidationError

    row = _db_row(model_id="row-billed", max_retries=-1)
    validation_error = APIResponseValidationError(
        response=requests.Response(), body=None, message="bad body"
    )
    images = _patch_openai_transport(monkeypatch, [validation_error])
    model = _published_model([row], monkeypatch)

    with TokenContextManager() as manager:
        with pytest.raises(Exception):
            await model.generate_image(prompt="p")
        media_rows = [
            d for d in manager.get_usage().details if d.get("type") == "media"
        ]

    assert images.calls == ["generate"]
    assert len(media_rows) == 1
    assert media_rows[0]["model_id"] == "row-billed"


def test_adapter_path_also_clamps_negative_max_retries() -> None:
    """The other construction path (get_image_model_instance) applies the same
    clamp, so the two never disagree about how many attempts a row gets."""
    from types import SimpleNamespace

    from xagent.core.model.image.adapter import get_image_model_instance

    row = SimpleNamespace(
        model_id="adapter-row",
        model_name="gpt-image-1",
        model_provider="openai",
        api_key="k",
        base_url=None,
        abilities=["generate", "edit"],
        timeout=5.0,
        max_retries=-1,
    )
    wrapper = get_image_model_instance(row)

    assert wrapper._retry_wrapper.max_retries == 1
