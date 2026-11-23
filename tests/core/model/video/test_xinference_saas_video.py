"""The ``xinference-saas`` video provider speaks the prepaid Model API shape."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest

from xagent.core.model import VideoModelConfig
from xagent.core.model.chat.token_context import TokenContextManager
from xagent.core.model.video.adapter import (
    XINFERENCE_SAAS_PROVIDER,
    create_video_model,
)
from xagent.core.model.video.base import BilledVideoTaskError
from xagent.core.model.video.xinference import XinferenceVideoModel
from xagent.core.tools.core.media_errors import MediaParameterError

pytestmark = pytest.mark.asyncio


def _build(provider: str = XINFERENCE_SAAS_PROVIDER) -> Any:
    config = VideoModelConfig(
        id="ignored",
        model_name="minimax-h3",
        model_provider=provider,
        base_url="https://cloud.test",
        api_key="key",
        abilities=["generate"],
        max_retries=1,
    )
    return create_video_model(config, model_id="platform/h3")


@pytest.fixture
def transport(monkeypatch):
    calls: dict[str, Any] = {"json": [], "form": [], "response": None}

    async def post_json(self, session, path, body):
        calls["json"].append({"path": path, "body": body})
        return calls["response"] or {
            "data": [{"url": None, "b64_json": "dmlkZW8="}],
            "usage": {"video_seconds": 4},
        }

    async def post_form(self, session, path, fields, files):
        calls["form"].append(path)
        return {"data": [{"url": "https://x.test/v.mp4"}]}

    monkeypatch.setattr(XinferenceVideoModel, "_post_video_json", post_json)
    monkeypatch.setattr(XinferenceVideoModel, "_post_video_form", post_form)
    return calls


def _rows(manager):
    return [d for d in manager.get_usage().details if d.get("type") == "media"]


async def test_saas_body_has_top_level_fields_and_no_kwargs(transport):
    model = _build()
    result = await model.generate_video(
        prompt="a fox", seconds=4, size="1344x768", negative_prompt="blur", seed=7
    )

    assert transport["json"] == [
        {
            "path": "/v1/video/generations",
            "body": {
                "model": "minimax-h3",
                "prompt": "a fox",
                "n": 1,
                "response_format": "b64_json",
                "duration": 4,
                "width": 1344,
                "height": 768,
                "seed": 7,
            },
        }
    ]
    assert result["video_url"] == "data:video/mp4;base64,dmlkZW8="
    assert transport["form"] == []


async def test_saas_minimal_body_omits_unset_fields(transport):
    await _build().generate_video(prompt="a fox")
    assert transport["json"][0]["body"] == {
        "model": "minimax-h3",
        "prompt": "a fox",
        "n": 1,
        "response_format": "b64_json",
    }


async def test_saas_maps_resolution_and_ratio_and_float_duration(transport):
    await _build().generate_video(
        prompt="a fox", duration="6.5", resolution="768p", ratio="1:1"
    )
    body = transport["json"][0]["body"]
    assert (body["duration"], body["width"], body["height"]) == (6.5, 768, 768)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1280x720", (1344, 768)),
        ("1920x1080", (1344, 768)),
        ("720x1280", (768, 1344)),
        ("1080x1080", (768, 768)),
        ("1344x768", (1344, 768)),
        ("1792x768", (1792, 768)),
    ],
)
async def test_saas_snaps_other_sizes_to_the_nearest_ratio(transport, size, expected):
    await _build().generate_video(prompt="a fox", size=size)
    body = transport["json"][0]["body"]
    assert (body["width"], body["height"]) == expected


async def test_saas_snaps_resolution_and_ratio_that_miss_the_list(transport):
    await _build().generate_video(prompt="a fox", resolution="768p", ratio="16:9")
    body = transport["json"][0]["body"]
    assert (body["width"], body["height"]) == (1344, 768)


@pytest.mark.parametrize("size", ["banana", "0x768", "-4x3"])
async def test_saas_rejects_malformed_sizes_listing_allowed_values(transport, size):
    with pytest.raises(MediaParameterError) as info:
        await _build().generate_video(prompt="a fox", size=size)
    assert info.value.field == "size"
    assert "1792x768" in info.value.allowed_values["size"]
    assert transport["json"] == []


@pytest.mark.parametrize("seconds", [3, 16, "abc", True])
async def test_saas_rejects_duration_out_of_range(transport, seconds):
    with pytest.raises(MediaParameterError) as info:
        await _build().generate_video(prompt="a fox", seconds=seconds)
    assert info.value.field == "duration"
    assert transport["json"] == []


async def test_saas_rejects_bad_seed(transport):
    with pytest.raises(MediaParameterError):
        await _build().generate_video(prompt="a fox", seed=2**31)
    assert transport["json"] == []


@pytest.mark.parametrize(
    "extra",
    [
        {"input_reference": "https://x.test/a.png"},
        {"image_url": "https://x.test/a.png"},
        {"first_frame_image_url": "https://x.test/a.png"},
        {"last_frame_image_url": "https://x.test/b.png"},
        {
            "first_frame_image_url": "https://x.test/a.png",
            "last_frame_image_url": "https://x.test/b.png",
        },
    ],
)
async def test_saas_refuses_image_and_frame_input_before_any_request(transport, extra):
    with pytest.raises(MediaParameterError):
        await _build().generate_video(prompt="a fox", **extra)
    assert transport["json"] == [] and transport["form"] == []


async def test_saas_records_usage_from_video_seconds(transport):
    transport["response"] = {
        "data": [{"b64_json": "dmlkZW8="}],
        "usage": {"video_seconds": 5},
    }
    with TokenContextManager() as manager:
        result = await _build().generate_video(prompt="a fox", seconds=4)
    [row] = _rows(manager)
    assert (row["quantity"], row["model_id"]) == (5.0, "platform/h3")
    assert result["duration"] == 5


async def test_saas_usage_falls_back_to_requested_seconds(transport):
    transport["response"] = {"data": [{"b64_json": "dmlkZW8="}]}
    with TokenContextManager() as manager:
        await _build().generate_video(prompt="a fox", seconds=9)
    [row] = _rows(manager)
    assert row["quantity"] == 9.0


async def test_saas_returned_duration_follows_the_billed_quantity(transport):
    transport["response"] = {"data": [{"b64_json": "dmlkZW8=", "duration": 7}]}
    with TokenContextManager() as manager:
        result = await _build().generate_video(prompt="a fox", seconds=4)
    [row] = _rows(manager)
    assert (row["quantity"], result["duration"]) == (7.0, 7)


@pytest.mark.parametrize(
    "response",
    [
        {"data": [{"b64_json": None, "url": None}]},
        {"data": ["not-a-dict"]},
        {"data": []},
    ],
)
async def test_saas_billed_but_unusable_response_is_a_billed_task_error(
    transport, response
):
    transport["response"] = response
    with TokenContextManager() as manager:
        with pytest.raises(BilledVideoTaskError):
            await _build().generate_video(prompt="a fox", seconds=4)
    assert len(_rows(manager)) == 1


async def test_saas_transport_error_becomes_runtime_error_without_usage(monkeypatch):
    async def boom(self, session, path, body):
        raise aiohttp.ClientConnectionError("down")

    monkeypatch.setattr(XinferenceVideoModel, "_post_video_json", boom)
    with TokenContextManager() as manager:
        with pytest.raises(RuntimeError, match="Xinference video generation failed"):
            await _build().generate_video(prompt="a fox")
    assert _rows(manager) == []


async def test_saas_cancel_before_the_body_arrives_records_nothing(monkeypatch):
    async def cancelled_after_body(self, session, path, body):
        raise asyncio.CancelledError

    monkeypatch.setattr(XinferenceVideoModel, "_post_video_json", cancelled_after_body)
    with TokenContextManager() as manager:
        with pytest.raises(asyncio.CancelledError):
            await _build().generate_video(prompt="a fox")
    assert _rows(manager) == []


async def test_saas_prefers_url_when_present(transport):
    transport["response"] = {
        "data": [{"url": "https://x.test/v.mp4", "b64_json": "dmlkZW8="}]
    }
    result = await _build().generate_video(prompt="a fox")
    assert result["video_url"] == "https://x.test/v.mp4"


async def test_legacy_provider_still_sends_kwargs(transport):
    model = _build("xinference")
    await model.generate_video(prompt="a fox", seconds=5, size="1280x720")
    assert transport["json"][0]["body"] == {
        "model": "minimax-h3",
        "prompt": "a fox",
        "n": 1,
        "kwargs": json.dumps({"seconds": 5, "size": "1280x720"}),
    }


async def test_legacy_provider_still_uses_image_endpoint(transport, tmp_path):
    model = _build("xinference")
    await model.generate_video(
        prompt="a fox", input_reference="data:image/png;base64,AAAA"
    )
    assert transport["form"] == ["/v1/video/generations/image"]


def test_observation_omits_inline_data_urls():
    from xagent.core.tools.artifacts import format_tool_result_for_observation

    text = format_tool_result_for_observation(
        "generate_video",
        {
            "success": True,
            "video_url": "data:video/mp4;base64," + "A" * 5000,
            "x": "data:video/mp4;base64,QQ==",
        },
    )
    assert "AAAA" not in text and "QQ==" not in text
    assert "inline video/mp4 data omitted" in text


async def test_saas_video_model_reports_text_to_video_only():
    assert _build().supports_image_input() is False


async def test_self_hosted_xinference_video_accepts_image_input():
    model = XinferenceVideoModel(
        model_name="m", base_url="http://localhost:9997", saas_api=False
    )
    assert model.supports_image_input() is True
