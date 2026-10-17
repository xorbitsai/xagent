"""Video providers meter at the provider boundary: one row per billed task.

Each test asserts the two numbers that matter: how many generations were
submitted to the provider, and how many usage rows exist afterwards.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from xagent.core.model import VideoModelConfig
from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.video.adapter import create_video_model, get_video_model_instance
from xagent.core.model.video.ark import ArkVideoModel
from xagent.core.model.video.base import BilledVideoTaskError, BilledVideoTaskTimeout
from xagent.core.model.video.xinference import XinferenceVideoModel


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    real_sleep = asyncio.sleep

    async def fast_sleep(_delay: float, *args: Any, **kwargs: Any) -> None:
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)


def _rows(manager: TokenContextManager) -> list[dict]:
    return [d for d in manager.get_usage().details if d.get("type") == "media"]


def _wrapped(provider: str, max_retries: int = 3) -> Any:
    config = VideoModelConfig(
        id="ignored",
        model_name="seedance-pro" if provider != "xinference" else "Wan2.1",
        model_provider=provider,
        base_url="http://provider.test",
        api_key="key",
        abilities=["generate"],
        max_retries=max_retries,
    )
    return create_video_model(config, model_id="platform/7")


class _ArkFake:
    """Stands in for the Ark SDK calls; counts task submissions."""

    def __init__(self, polls: list[Any], created: dict | None = None) -> None:
        self.creates = 0
        self.polls = list(polls)
        self.created = {"id": "task-1"} if created is None else created

    def create(self, content, **kwargs):
        self.creates += 1
        return self.created

    def get(self, task_id):
        result = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
        if isinstance(result, Exception):
            raise result
        return result


def _ark(fake: _ArkFake, max_retries: int = 3) -> Any:
    wrapper = _wrapped("volcengine-ark", max_retries)
    inner = wrapper._inner
    inner._client_create = fake.create
    inner._client_get = fake.get
    return wrapper


def _succeeded(**overrides: Any) -> dict:
    task = {
        "id": "task-1",
        "status": "succeeded",
        "duration": 5,
        "resolution": "720p",
        "usage": {"completion_tokens": 108900, "total_tokens": 108900},
        "content": {"video_url": "https://cdn.test/v.mp4"},
    }
    task.update(overrides)
    return task


async def test_ark_success_records_reported_seconds_tokens_and_identity():
    fake = _ArkFake([_succeeded()])
    model = _ark(fake)

    with TokenContextManager() as manager:
        result = await model.generate_video(prompt="a fox", poll_interval=0)

    assert result["video_url"] == "https://cdn.test/v.mp4"
    [row] = _rows(manager)
    assert row["call_type"] == "video"
    assert (row["unit"], row["quantity"]) == ("seconds", 5.0)
    assert row["provider_output_tokens"] == 108900
    assert row["resolution"] == "720p"
    assert (row["model_id"], row["model"]) == ("platform/7", "seedance-pro")
    assert fake.creates == 1


@pytest.mark.parametrize("status", ["failed", "cancelled", "expired"])
async def test_ark_terminal_failure_records_once_and_is_not_retried(status):
    fake = _ArkFake(
        [_succeeded(status=status, duration=5, usage={"completion_tokens": 0})]
    )
    model = _ark(fake)

    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskError):
            await model.generate_video(prompt="a fox", poll_interval=0)

    assert fake.creates == 1
    [row] = _rows(manager)
    assert row["quantity"] == 0.0
    assert row["provider_output_tokens"] == 0


async def test_ark_deadline_records_once_and_is_not_retried():
    fake = _ArkFake([{"id": "task-1", "status": "running"}])
    model = _ark(fake)

    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskTimeout) as excinfo:
            await model.generate_video(prompt="a fox", poll_interval=0, timeout=0.001)

    assert isinstance(excinfo.value, TimeoutError)
    assert fake.creates == 1
    [row] = _rows(manager)
    assert row["quantity"] == 0.0


async def test_ark_persistent_poll_network_errors_do_not_resubmit_the_task():
    # The timeout is raised *from* an httpx network error, which the adapter's
    # retry predicate treats as transient; it must not re-create the task while
    # the first one may still finish and be billed.
    fake = _ArkFake([httpx.ConnectError("reset")])
    model = _ark(fake)

    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskTimeout):
            await model.generate_video(prompt="a fox", poll_interval=0, timeout=0.001)

    assert fake.creates == 1
    assert len(_rows(manager)) == 1


async def test_ark_missing_task_id_records_once_and_is_not_retried():
    fake = _ArkFake([_succeeded()], created={})
    model = _ark(fake)

    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskError):
            await model.generate_video(prompt="a fox", poll_interval=0)

    assert fake.creates == 1
    assert [row["quantity"] for row in _rows(manager)] == [0.0]


async def test_ark_create_transport_failure_is_retried_without_rows():
    # Before a task exists nothing is billed; each attempt is a fresh request.
    fake = _ArkFake([_succeeded()])
    model = _ark(fake)

    def failing_create(content, **kwargs):
        fake.creates += 1
        raise httpx.ConnectError("refused")

    model._inner._client_create = failing_create

    with TokenContextManager() as manager:
        with pytest.raises(httpx.ConnectError):
            await model.generate_video(prompt="a fox", poll_interval=0)

    assert fake.creates == 3
    assert _rows(manager) == []


async def test_ark_rejects_not_waiting_before_submitting():
    fake = _ArkFake([_succeeded()])
    model = _ark(fake)

    with TokenContextManager() as manager:
        with pytest.raises(ValueError, match="wait_for_result"):
            await model.generate_video(prompt="a fox", wait_for_result=False)

    assert fake.creates == 0
    assert _rows(manager) == []


def test_db_built_provider_carries_the_configured_id_on_the_inner_model():
    db_model = SimpleNamespace(
        model_provider="volcengine-ark",
        model_name="seedance-pro",
        model_id="platform/9",
        api_key="key",
        base_url="https://ark.test",
        abilities=["generate"],
        timeout=10.0,
        max_retries=2,
    )
    wrapper = get_video_model_instance(db_model)
    assert isinstance(wrapper._inner, ArkVideoModel)
    assert wrapper._inner.model_id == "platform/9"


# --- Xinference ----------------------------------------------------------------


def _xinference(monkeypatch, responder) -> tuple[Any, dict]:
    calls = {"posts": 0}

    async def post_json(self, session, path, body):
        calls["posts"] += 1
        return await responder()

    monkeypatch.setattr(XinferenceVideoModel, "_post_video_json", post_json)
    return _wrapped("xinference"), calls


async def test_xinference_records_server_duration(monkeypatch):
    async def respond():
        return {"data": [{"url": "https://x.test/v.mp4", "duration": 8}]}

    model, calls = _xinference(monkeypatch, respond)
    with TokenContextManager() as manager:
        await model.generate_video(prompt="a fox", seconds=5)

    [row] = _rows(manager)
    assert (row["quantity"], row["model_id"], row["model"]) == (
        8.0,
        "platform/7",
        "Wan2.1",
    )
    assert calls["posts"] == 1


async def test_xinference_falls_back_to_the_requested_length(monkeypatch):
    async def respond():
        return {"data": [{"url": "https://x.test/v.mp4"}]}

    model, _calls = _xinference(monkeypatch, respond)
    with TokenContextManager() as manager:
        await model.generate_video(prompt="a fox", seconds=5)

    assert [row["quantity"] for row in _rows(manager)] == [5.0]


async def test_xinference_unusable_body_records_once_and_is_not_retried(monkeypatch):
    async def respond():
        return {"data": [{"id": "no-video"}]}

    model, calls = _xinference(monkeypatch, respond)
    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskError):
            await model.generate_video(prompt="a fox", seconds=5)

    assert calls["posts"] == 1
    assert len(_rows(manager)) == 1


async def test_xinference_undecodable_body_records_once_and_is_not_retried(monkeypatch):
    async def respond():
        raise BilledVideoTaskError("Xinference video response could not be decoded")

    model, calls = _xinference(monkeypatch, respond)
    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskError):
            await model.generate_video(prompt="a fox", seconds=5)

    assert calls["posts"] == 1
    assert [row["quantity"] for row in _rows(manager)] == [0.0]


async def test_xinference_decode_failure_is_classified_as_billed(monkeypatch):
    class _Response:
        status = 200

        async def json(self):
            raise ValueError("not json")

    class _Request:
        async def __aenter__(self):
            return _Response()

        async def __aexit__(self, *exc):
            return False

    session = SimpleNamespace(request=lambda *a, **k: _Request())
    model = XinferenceVideoModel(model_name="Wan2.1", base_url="http://x.test")

    with pytest.raises(BilledVideoTaskError):
        await model._request_json(session, "POST", "/v1/video/generations")


async def test_xinference_rejects_batches_before_rendering(monkeypatch):
    async def respond():
        return {"data": [{"url": "https://x.test/v.mp4", "duration": 8}]}

    model, calls = _xinference(monkeypatch, respond)
    with TokenContextManager() as manager:
        with pytest.raises(ValueError, match="one video"):
            await model.generate_video(prompt="a fox", seconds=5, n=3)

    assert calls["posts"] == 0
    assert _rows(manager) == []
