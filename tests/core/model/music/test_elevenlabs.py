"""Tests for the independent ElevenLabs music model."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.model.music import (
    ElevenLabsMusicModel,
    MusicResult,
    get_music_model_instance,
)


async def test_generate_music_uses_sdk_and_joins_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def compose(**kwargs: object):
        calls.append(kwargs)
        yield b"music "
        yield b"audio"

    fake_client = SimpleNamespace(music=SimpleNamespace(compose=compose))
    monkeypatch.setattr(
        ElevenLabsMusicModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    result = await ElevenLabsMusicModel(
        model_name="music_v2",
        api_key="test-key",
        timeout=45,
        max_retries=4,
    ).generate_music(
        prompt="Cinematic orchestral score",
        music_length_seconds=30,
        force_instrumental=True,
    )

    assert isinstance(result, MusicResult)
    assert result.audio == b"music audio"
    assert result.format == "mp3"
    assert result.sample_rate == 48000
    assert calls == [
        {
            "prompt": "Cinematic orchestral score",
            "model_id": "music_v2",
            "force_instrumental": True,
            "output_format": "auto",
            "request_options": {
                "timeout_in_seconds": 45,
                "max_retries": 4,
            },
            "music_length_ms": 30000,
        }
    ]


@pytest.mark.parametrize(
    ("prompt", "length", "message"),
    [
        ("   ", None, "must not be empty"),
        ("Music", 2.9, "between 3 and 600"),
        ("Music", 601, "between 3 and 600"),
        ("x" * 4101, None, "must not exceed 4100"),
    ],
)
async def test_generate_music_validates_parameters(
    prompt: str, length: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await ElevenLabsMusicModel(api_key="test-key").generate_music(
            prompt=prompt,
            music_length_seconds=length,
        )


async def test_validate_connection_uses_non_billed_plan_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_plan = AsyncMock(return_value={"chunks": []})
    close = AsyncMock()
    fake_client = SimpleNamespace(
        music=SimpleNamespace(
            composition_plan=SimpleNamespace(create=create_plan),
        ),
        aclose=close,
    )
    monkeypatch.setattr(
        ElevenLabsMusicModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    model = ElevenLabsMusicModel(
        model_name="music_v2",
        api_key="test-key",
        timeout=30,
        max_retries=2,
    )
    await model.validate_connection()
    await model.aclose()

    create_plan.assert_awaited_once_with(
        prompt="A short instrumental music cue",
        music_length_ms=3000,
        model_id="music_v2",
        request_options={"timeout_in_seconds": 30, "max_retries": 2},
    )
    close.assert_awaited_once_with()


async def test_generate_music_normalizes_non_streaming_sdk_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def compose(**kwargs: object) -> bytes:
        return b"not-an-async-stream"

    fake_client = SimpleNamespace(music=SimpleNamespace(compose=compose))
    monkeypatch.setattr(
        ElevenLabsMusicModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    with pytest.raises(RuntimeError, match="ElevenLabs music generation failed"):
        await ElevenLabsMusicModel(api_key="test-key").generate_music(
            prompt="Ambient score"
        )


async def test_list_models_reads_provider_list_and_keeps_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_models = AsyncMock(
        return_value=[
            SimpleNamespace(
                model_id="music_v10",
                name="Eleven Music v10",
                description="Future music generation",
            ),
            SimpleNamespace(
                model_id="music_v3",
                name="Eleven Music v3",
                description="Next-generation music generation",
            ),
            SimpleNamespace(model_id="eleven_v3", name="Eleven v3"),
        ]
    )
    close = AsyncMock()
    fake_client = SimpleNamespace(
        models=SimpleNamespace(list=list_models),
        aclose=close,
    )
    monkeypatch.setattr(
        ElevenLabsMusicModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    models = await ElevenLabsMusicModel.async_list_available_models(api_key="test-key")

    assert [model["id"] for model in models] == [
        "music_v10",
        "music_v3",
        "music_v2",
        "music_v1",
    ]
    assert models[0]["name"] == "Eleven Music v10"
    assert all(model["category"] == "music" for model in models)
    assert all(model["abilities"] == ["generate"] for model in models)
    list_models.assert_awaited_once_with()
    close.assert_awaited_once_with()


async def test_list_models_falls_back_when_key_lacks_list_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_models = AsyncMock(
        side_effect=RuntimeError(
            "status: missing_permissions; missing permission user_read"
        )
    )
    close = AsyncMock()
    fake_client = SimpleNamespace(
        models=SimpleNamespace(list=list_models),
        aclose=close,
    )
    monkeypatch.setattr(
        ElevenLabsMusicModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    models = await ElevenLabsMusicModel.async_list_available_models(api_key="test-key")

    assert [model["id"] for model in models][:2] == ["music_v2", "music_v1"]
    close.assert_awaited_once_with()


async def test_list_models_does_not_hide_other_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_models = AsyncMock(side_effect=RuntimeError("invalid API key"))
    close = AsyncMock()
    fake_client = SimpleNamespace(
        models=SimpleNamespace(list=list_models),
        aclose=close,
    )
    monkeypatch.setattr(
        ElevenLabsMusicModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    with pytest.raises(RuntimeError, match="invalid API key"):
        await ElevenLabsMusicModel.async_list_available_models(api_key="test-key")

    close.assert_awaited_once_with()


async def test_list_models_treats_none_response_as_empty_provider_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_models = AsyncMock(return_value=None)
    close = AsyncMock()
    fake_client = SimpleNamespace(
        models=SimpleNamespace(list=list_models),
        aclose=close,
    )
    monkeypatch.setattr(
        ElevenLabsMusicModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    models = await ElevenLabsMusicModel.async_list_available_models(api_key="test-key")

    assert [model["id"] for model in models][:2] == ["music_v2", "music_v1"]
    list_models.assert_awaited_once_with()
    close.assert_awaited_once_with()


def test_adapter_builds_independent_music_model() -> None:
    db_model = SimpleNamespace(
        model_id="music-config",
        model_provider=" ELEVENLABS ",
        model_name="music_v2",
        api_key="test-key",
        base_url=None,
        timeout=75,
        max_retries=6,
    )

    model = get_music_model_instance(db_model)

    assert isinstance(model, ElevenLabsMusicModel)
    assert model.model_name == "music_v2"
    assert model.abilities == ["generate"]
    assert model.timeout == 75
    assert model.max_retries == 6
