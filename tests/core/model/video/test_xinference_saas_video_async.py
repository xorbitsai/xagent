"""Async job API of the ``xinference-saas`` video provider.

The real adapter builds the model; only the aiohttp session is replaced by a
scripted fake server, so every assertion is on the exact URL, method and body
the client sends.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import aiohttp
import pytest

from xagent.core.model import VideoModelConfig
from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.video import xinference as xinference_module
from xagent.core.model.video.adapter import (
    XINFERENCE_SAAS_PROVIDER,
    create_video_model,
    retry_on,
)
from xagent.core.model.video.base import BilledVideoTaskError, BilledVideoTaskTimeout
from xagent.core.model.video.xinference import XinferenceVideoModel

pytestmark = pytest.mark.asyncio

HOST = "https://cloud.xinference.co"
MP4 = b"\x00\x00\x00\x18ftypmp42-fake"


class FakeResponse:
    def __init__(self, status: int, body: Any = None, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {}
        self.request_info = None
        self.history = ()

    async def json(self, content_type=None):
        if self._body is None or isinstance(self._body, bytes):
            raise ValueError("no json")
        return self._body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeServer:
    """Scripted job server; records every call as (method, url, headers, json)."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict, Any]] = []
        self.post_replies: list[Any] = []
        self.get_replies: list[Any] = []
        self.delete_reply: Any = FakeResponse(
            200, {"id": "vid_1", "object": "video.deleted", "deleted": True}
        )
        self.content_reply: Any = FakeResponse(200, MP4)

    def urls(self, method: str) -> list[str]:
        return [u for m, u, _, _ in self.calls if m == method]

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append((method, url, dict(headers or {}), json))
        if method == "POST":
            reply = self.post_replies.pop(0)
        elif method == "DELETE":
            reply = self.delete_reply
        elif url.endswith("/content"):
            reply = self.content_reply
        else:
            reply = (
                self.get_replies.pop(0)
                if len(self.get_replies) > 1
                else (self.get_replies[0])
            )
        if isinstance(reply, BaseException):
            raise reply
        return reply


class FakeSessionCM:
    def __init__(self, server):
        self.server = server

    async def __aenter__(self):
        return self.server

    async def __aexit__(self, *exc):
        return False


def accepted(job_id="vid_1"):
    return FakeResponse(202, {"id": job_id, "object": "video", "status": "queued"})


def job(status, **extra):
    return FakeResponse(
        200, {"id": "vid_1", "object": "video", "status": status, **extra}
    )


def completed(seconds=5):
    return job("completed", usage={"video_seconds": seconds}, expires_at=1)


@pytest.fixture
def server(monkeypatch):
    srv = FakeServer()
    monkeypatch.setenv("XAGENT_XINFERENCE_SAAS_VIDEO_ASYNC", "true")
    monkeypatch.setattr(
        xinference_module,
        "create_public_model_aiohttp_session",
        lambda **kw: FakeSessionCM(srv),
    )
    monkeypatch.setattr(xinference_module, "SAAS_VIDEO_JOB_POLL_SECONDS", 0.0)
    monkeypatch.setattr(xinference_module, "SAAS_VIDEO_SUBMIT_PAUSE_SECONDS", 0.0)
    monkeypatch.setattr(xinference_module, "SAAS_VIDEO_QUEUE_FULL_RETRY_SECONDS", 0.0)
    return srv


def _build(provider: str = XINFERENCE_SAAS_PROVIDER, base_url: str = HOST) -> Any:
    return create_video_model(
        VideoModelConfig(
            id="ignored",
            model_name="minimax-h3",
            model_provider=provider,
            base_url=base_url,
            api_key="key",
            abilities=["generate"],
            max_retries=1,
        ),
        model_id="platform/h3",
    )


def _rows(manager):
    return [d for d in manager.get_usage().details if d.get("type") == "media"]


async def test_happy_path_polls_then_downloads_the_mp4(server):
    server.post_replies = [accepted()]
    server.get_replies = [
        job("queued", queue_position=2),
        job("in_progress"),
        completed(6),
    ]
    with TokenContextManager() as manager:
        result = await _build().generate_video(
            prompt="a fox", seconds=4.2, size="1344x768", seed=7
        )

    methods_urls = [(m, u) for m, u, _, _ in server.calls]
    assert methods_urls == [
        ("POST", f"{HOST}/v1/videos"),
        ("GET", f"{HOST}/v1/videos/vid_1"),
        ("GET", f"{HOST}/v1/videos/vid_1"),
        ("GET", f"{HOST}/v1/videos/vid_1"),
        ("GET", f"{HOST}/v1/videos/vid_1/content"),
    ]
    _, _, headers, body = server.calls[0]
    assert body == {
        "model": "minimax-h3",
        "prompt": "a fox",
        "seconds": 5,
        "size": "1344x768",
        "seed": 7,
    }
    assert headers["Authorization"] == "Bearer key"
    assert headers["Idempotency-Key"]
    assert result["video_url"] == "data:video/mp4;base64," + base64.b64encode(
        MP4
    ).decode("ascii")
    assert result["task_id"] == "vid_1" and result["duration"] == 6
    [row] = _rows(manager)
    assert (row["quantity"], row["model_id"]) == (6.0, "platform/h3")


async def test_base_url_with_v1_suffix_does_not_double_the_prefix(server):
    server.post_replies = [accepted()]
    server.get_replies = [completed()]
    await _build(base_url=f"{HOST}/v1").generate_video(prompt="a fox")
    assert server.urls("POST") == [f"{HOST}/v1/videos"]


async def test_minimal_body_omits_unset_fields(server):
    server.post_replies = [accepted()]
    server.get_replies = [completed()]
    await _build().generate_video(prompt="a fox")
    assert server.calls[0][3] == {"model": "minimax-h3", "prompt": "a fox"}


async def test_usage_falls_back_to_requested_seconds(server):
    server.post_replies = [accepted()]
    server.get_replies = [job("completed")]
    with TokenContextManager() as manager:
        await _build().generate_video(prompt="a fox", seconds=9)
    [row] = _rows(manager)
    assert row["quantity"] == 9.0


async def test_queue_full_retries_with_the_same_idempotency_key(server):
    full = FakeResponse(
        429, {"error": {"code": "video_queue_full"}}, headers={"Retry-After": "30"}
    )
    server.post_replies = [full, accepted()]
    server.get_replies = [completed()]
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        sleeps.append(delay)
        await real_sleep(0)

    xinference_module.asyncio.sleep = fake_sleep  # restored below
    try:
        await _build().generate_video(prompt="a fox")
    finally:
        xinference_module.asyncio.sleep = real_sleep
    posts = [h for m, _, h, _ in server.calls if m == "POST"]
    assert len(posts) == 2
    assert posts[0]["Idempotency-Key"] == posts[1]["Idempotency-Key"]
    assert 30.0 in sleeps


async def test_queue_full_exhausted_is_retryable_and_bills_nothing(server):
    full = FakeResponse(429, {"error": {"code": "video_queue_full"}})
    server.post_replies = [full, full, full]
    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError) as info:
            await _build().generate_video(prompt="a fox")
    assert retry_on(info.value) is True
    assert _rows(manager) == []
    assert server.urls("DELETE") == []


async def test_server_error_on_submit_is_not_retried_into_a_second_job(server):
    boom = FakeResponse(503, {"error": {"code": "unavailable"}})
    server.post_replies = [boom, boom, boom]
    with pytest.raises(RuntimeError) as info:
        await _build().generate_video(prompt="a fox")
    assert retry_on(info.value) is False
    assert (
        len({h["Idempotency-Key"] for m, _, h, _ in server.calls if m == "POST"}) == 1
    )


async def test_client_error_on_submit_is_a_plain_failure(server):
    server.post_replies = [
        FakeResponse(409, {"error": {"code": "idempotency_conflict"}})
    ]
    with pytest.raises(RuntimeError, match="idempotency_conflict") as info:
        await _build().generate_video(prompt="a fox")
    assert retry_on(info.value) is False
    assert len(server.calls) == 1


@pytest.mark.parametrize(
    "code", ["video_expired", "video_cancelled", "upstream_failure"]
)
async def test_unbilled_failures_record_nothing_and_are_not_retried(server, code):
    server.post_replies = [accepted()]
    server.get_replies = [
        job("failed", error={"code": code, "message": "boom"}),
    ]
    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError, match=code) as info:
            await _build().generate_video(prompt="a fox", seconds=4)
    assert not isinstance(info.value, BilledVideoTaskError)
    assert retry_on(info.value) is False
    assert _rows(manager) == []
    assert server.urls("DELETE") == [] and len(server.urls("POST")) == 1


async def test_unknown_failure_code_is_recorded_unmeasured_and_not_retried(server):
    server.post_replies = [accepted()]
    server.get_replies = [job("failed", error={"code": "weird", "message": "?"})]
    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskError) as info:
            await _build().generate_video(prompt="a fox", seconds=4)
    assert retry_on(info.value) is False
    [row] = _rows(manager)
    assert row["quantity"] == 0.0


async def test_cancel_while_queued_deletes_the_job_and_records_nothing(server):
    server.post_replies = [accepted()]
    server.get_replies = [job("queued")]
    gate = asyncio.Event()
    real_sleep = asyncio.sleep

    async def stall(delay):
        gate.set()
        await real_sleep(3600)

    xinference_module.asyncio.sleep = stall
    try:
        with TokenContextManager() as manager:
            task = asyncio.ensure_future(
                _build().generate_video(prompt="a fox", seconds=4)
            )
            await gate.wait()
            task.cancel()
            xinference_module.asyncio.sleep = real_sleep
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        xinference_module.asyncio.sleep = real_sleep
    assert server.urls("DELETE") == [f"{HOST}/v1/videos/vid_1"]
    assert _rows(manager) == []


async def test_cancel_while_in_progress_records_the_charged_render(server):
    server.post_replies = [accepted()]
    server.get_replies = [job("in_progress")]
    server.delete_reply = FakeResponse(
        409, {"error": {"code": "video_in_progress", "message": "rendering"}}
    )
    gate = asyncio.Event()
    real_sleep = asyncio.sleep

    async def stall(delay):
        gate.set()
        await real_sleep(3600)

    xinference_module.asyncio.sleep = stall
    try:
        with TokenContextManager() as manager:
            task = asyncio.ensure_future(
                _build().generate_video(prompt="a fox", seconds=7)
            )
            await gate.wait()
            task.cancel()
            xinference_module.asyncio.sleep = real_sleep
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        xinference_module.asyncio.sleep = real_sleep
    assert server.urls("DELETE") == [f"{HOST}/v1/videos/vid_1"]
    [row] = _rows(manager)
    assert row["quantity"] == 7.0


async def test_delete_200_on_a_job_last_seen_rendering_is_still_charged(server):
    # It finished between the last poll and the DELETE; the file is deleted
    # without a refund.
    server.post_replies = [accepted()]
    server.get_replies = [job("in_progress")]
    gate = asyncio.Event()
    real_sleep = asyncio.sleep

    async def stall(delay):
        gate.set()
        await real_sleep(3600)

    xinference_module.asyncio.sleep = stall
    try:
        with TokenContextManager() as manager:
            task = asyncio.ensure_future(
                _build().generate_video(prompt="a fox", seconds=6)
            )
            await gate.wait()
            task.cancel()
            xinference_module.asyncio.sleep = real_sleep
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        xinference_module.asyncio.sleep = real_sleep
    [row] = _rows(manager)
    assert row["quantity"] == 6.0


async def test_deadline_while_in_progress_records_and_raises_a_billed_timeout(
    server, monkeypatch
):
    server.post_replies = [accepted()]
    server.get_replies = [job("in_progress")]
    server.delete_reply = FakeResponse(409, {"error": {"code": "video_in_progress"}})
    monkeypatch.setattr(xinference_module, "SAAS_VIDEO_MIN_JOB_DEADLINE_SECONDS", 0.0)
    monkeypatch.setattr(xinference_module, "SAAS_VIDEO_JOB_POLL_SECONDS", 0.01)
    model = _build()
    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskTimeout) as info:
            await model.generate_video(prompt="a fox", seconds=5, timeout=0.05)
    assert retry_on(info.value) is False
    assert server.urls("DELETE") == [f"{HOST}/v1/videos/vid_1"]
    [row] = _rows(manager)
    assert row["quantity"] == 5.0


async def test_deadline_while_queued_cancels_for_free(server, monkeypatch):
    server.post_replies = [accepted()]
    server.get_replies = [job("queued")]
    monkeypatch.setattr(xinference_module, "SAAS_VIDEO_MIN_JOB_DEADLINE_SECONDS", 0.0)
    monkeypatch.setattr(xinference_module, "SAAS_VIDEO_JOB_POLL_SECONDS", 0.01)
    with TokenContextManager() as manager:
        with pytest.raises(TimeoutError) as info:
            await _build().generate_video(prompt="a fox", seconds=5, timeout=0.05)
    assert not isinstance(info.value, BilledVideoTaskError)
    assert retry_on(info.value) is False
    assert _rows(manager) == []


async def test_deadline_floor_covers_queue_expiry_plus_render():
    assert xinference_module.SAAS_VIDEO_MIN_JOB_DEADLINE_SECONDS >= 1800 + 600


async def test_cancel_with_unreachable_delete_records_an_unmeasured_call(server):
    server.post_replies = [accepted()]
    server.get_replies = [job("queued")]
    server.delete_reply = aiohttp.ClientConnectionError("down")
    gate = asyncio.Event()
    real_sleep = asyncio.sleep

    async def stall(delay):
        gate.set()
        await real_sleep(3600)

    xinference_module.asyncio.sleep = stall
    try:
        with TokenContextManager() as manager:
            task = asyncio.ensure_future(
                _build().generate_video(prompt="a fox", seconds=4)
            )
            await gate.wait()
            task.cancel()
            xinference_module.asyncio.sleep = real_sleep
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        xinference_module.asyncio.sleep = real_sleep
    [row] = _rows(manager)
    assert row["quantity"] == 0.0


async def test_content_gone_after_completion_is_billed_and_not_retried(server):
    server.post_replies = [accepted()]
    server.get_replies = [completed(4)]
    server.content_reply = FakeResponse(410, {"error": {"code": "video_expired"}})
    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskError) as info:
            await _build().generate_video(prompt="a fox", seconds=4)
    assert retry_on(info.value) is False
    [row] = _rows(manager)
    assert row["quantity"] == 4.0
    assert server.urls("DELETE") == []


async def test_polling_errors_are_tolerated_until_completion(server):
    server.post_replies = [accepted()]
    server.get_replies = [aiohttp.ClientConnectionError("blip"), completed()]
    result = await _build().generate_video(prompt="a fox", seconds=4)
    assert result["status"] == "succeeded"


async def test_validation_still_runs_before_any_request(server):
    from xagent.core.tools.core.media_errors import MediaParameterError

    for kwargs in (
        {"seconds": 3},
        {"size": "banana"},
        {"n": 2},
        {"input_reference": "https://x.test/a.png"},
    ):
        with pytest.raises(MediaParameterError):
            await _build().generate_video(prompt="a fox", **kwargs)
    assert server.calls == []


@pytest.mark.parametrize("value", [None, "", "false", "0", "off", "garbage"])
async def test_default_path_is_still_the_sync_endpoint(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("XAGENT_XINFERENCE_SAAS_VIDEO_ASYNC", raising=False)
    else:
        monkeypatch.setenv("XAGENT_XINFERENCE_SAAS_VIDEO_ASYNC", value)
    posted: list[Any] = []

    async def post_json(self, session, path, body):
        posted.append((path, body))
        return {"data": [{"b64_json": "dmlkZW8="}], "usage": {"video_seconds": 4}}

    monkeypatch.setattr(XinferenceVideoModel, "_post_video_json", post_json)
    result = await _build().generate_video(prompt="a fox", seconds=4, size="1344x768")
    assert posted == [
        (
            "/v1/video/generations",
            {
                "model": "minimax-h3",
                "prompt": "a fox",
                "n": 1,
                "response_format": "b64_json",
                "duration": 4,
                "width": 1344,
                "height": 768,
            },
        )
    ]
    assert result["video_url"] == "data:video/mp4;base64,dmlkZW8="


async def test_legacy_xinference_provider_ignores_the_switch(monkeypatch):
    monkeypatch.setenv("XAGENT_XINFERENCE_SAAS_VIDEO_ASYNC", "true")
    posted: list[str] = []

    async def post_json(self, session, path, body):
        posted.append(path)
        return {"data": [{"b64_json": "dmlkZW8="}]}

    monkeypatch.setattr(XinferenceVideoModel, "_post_video_json", post_json)
    await _build("xinference").generate_video(prompt="a fox")
    assert posted == ["/v1/video/generations"]
