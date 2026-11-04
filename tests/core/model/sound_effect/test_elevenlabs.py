"""Tests for the independent ElevenLabs sound effect model."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xagent.core.model.sound_effect import (
    ElevenLabsSoundEffectModel,
    SoundEffectResult,
    get_sound_effect_model_instance,
)
from xagent.core.model.sound_effect.elevenlabs import (
    ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL,
)


async def test_generate_sound_effect_uses_sdk_and_joins_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def convert(**kwargs: object):
        calls.append(kwargs)
        yield b"thunder "
        yield b"audio"

    fake_client = SimpleNamespace(
        text_to_sound_effects=SimpleNamespace(convert=convert)
    )
    monkeypatch.setattr(
        ElevenLabsSoundEffectModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    result = await ElevenLabsSoundEffectModel(
        api_key="test-key",
        timeout=25,
        max_retries=5,
    ).generate_sound_effect(
        text="  Distant thunder over a valley  ",
        duration_seconds=4.5,
        prompt_influence=0.7,
        loop=True,
    )

    assert isinstance(result, SoundEffectResult)
    assert result.audio == b"thunder audio"
    assert result.format == "mp3"
    assert result.sample_rate == 44100
    assert calls == [
        {
            "text": "Distant thunder over a valley",
            "output_format": "mp3_44100_128",
            "loop": True,
            "prompt_influence": 0.7,
            "model_id": ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL,
            "request_options": {
                "timeout_in_seconds": 25,
                "max_retries": 5,
            },
            "duration_seconds": 4.5,
        }
    ]


async def test_generate_sound_effect_omits_automatic_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def convert(**kwargs: object):
        calls.append(kwargs)

        async def gen():
            yield b"audio"

        return gen()

    fake_client = SimpleNamespace(
        text_to_sound_effects=SimpleNamespace(convert=convert)
    )
    monkeypatch.setattr(
        ElevenLabsSoundEffectModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    await ElevenLabsSoundEffectModel(api_key="test-key").generate_sound_effect(
        text="Door slam"
    )

    assert "duration_seconds" not in calls[0]


@pytest.mark.parametrize(
    ("model_name", "kwargs", "message"),
    [
        (ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL, {"text": "   "}, "must not be empty"),
        (
            ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL,
            {"text": "Rain", "duration_seconds": 0.4},
            "duration_seconds must be between 0.5 and 30",
        ),
        (
            ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL,
            {"text": "Rain", "prompt_influence": 1.1},
            "prompt_influence must be between 0 and 1",
        ),
        (
            "legacy-model",
            {"text": "Rain", "loop": True},
            "loop is only supported by eleven_text_to_sound_v2",
        ),
    ],
)
async def test_generate_sound_effect_validates_parameters(
    model_name: str, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await ElevenLabsSoundEffectModel(
            model_name=model_name, api_key="test-key"
        ).generate_sound_effect(**kwargs)


async def test_generate_sound_effect_redacts_sdk_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def convert(**kwargs: object):
        raise RuntimeError("request failed api_key=sk-elevenlabs-secret123")

    fake_client = SimpleNamespace(
        text_to_sound_effects=SimpleNamespace(convert=convert)
    )
    monkeypatch.setattr(
        ElevenLabsSoundEffectModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await ElevenLabsSoundEffectModel(api_key="test-key").generate_sound_effect(
            text="Rain"
        )

    assert "sk-elevenlabs-secret123" not in str(exc_info.value)
    assert "api_key=***t123" in str(exc_info.value)


async def test_validate_connection_probes_account_without_model_name_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_models = AsyncMock(return_value=[])
    close = AsyncMock()
    fake_client = SimpleNamespace(
        models=SimpleNamespace(list=list_models),
        aclose=close,
    )
    monkeypatch.setattr(
        ElevenLabsSoundEffectModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    model = ElevenLabsSoundEffectModel(
        model_name="future-sfx-model",
        api_key="test-key",
        timeout=20,
        max_retries=2,
    )
    await model.validate_connection()
    await model.aclose()

    list_models.assert_awaited_once_with(
        request_options={"timeout_in_seconds": 20, "max_retries": 2}
    )
    close.assert_awaited_once_with()


async def test_list_models_discovers_sound_effect_models_from_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_models = AsyncMock(
        return_value=[
            SimpleNamespace(
                model_id="eleven_text_to_sound_v2",
                name="Text to Sound v2",
                description="Generate sound effects",
            ),
            SimpleNamespace(
                model_id="eleven_text_to_sound_v3",
                name="Text to Sound v3",
                description="Next-generation sound effects",
            ),
            SimpleNamespace(
                model_id="future-sfx-model",
                name="Future audio model",
                description="Provider capability based model",
                can_do_text_to_sound_effects=True,
            ),
            SimpleNamespace(
                model_id="eleven_multilingual_v2",
                name="Multilingual v2",
                description="Text to speech",
                can_do_text_to_speech=True,
            ),
        ]
    )
    close = AsyncMock()
    fake_client = SimpleNamespace(
        models=SimpleNamespace(list=list_models),
        aclose=close,
    )
    monkeypatch.setattr(
        ElevenLabsSoundEffectModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    models = await ElevenLabsSoundEffectModel.async_list_available_models(
        api_key="test-key"
    )

    assert [model["id"] for model in models] == [
        "eleven_text_to_sound_v2",
        "eleven_text_to_sound_v3",
        "future-sfx-model",
    ]
    assert all(model["category"] == "sound_effect" for model in models)
    assert all(model["abilities"] == ["generate"] for model in models)
    assert models[1]["name"] == "Text to Sound v3"
    assert models[1]["description"] == "Next-generation sound effects"
    list_models.assert_awaited_once_with()
    close.assert_awaited_once_with()


async def test_list_models_uses_documented_fallback_when_provider_omits_sfx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_models = AsyncMock(
        return_value=[
            SimpleNamespace(
                model_id="eleven_v3",
                name="Eleven v3",
                description="Text to speech",
                can_do_text_to_speech=True,
            )
        ]
    )
    close = AsyncMock()
    fake_client = SimpleNamespace(
        models=SimpleNamespace(list=list_models),
        aclose=close,
    )
    monkeypatch.setattr(
        ElevenLabsSoundEffectModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    models = await ElevenLabsSoundEffectModel.async_list_available_models(
        api_key="test-key"
    )

    assert models == [
        {
            "id": ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL,
            "object": "model",
            "owned_by": "elevenlabs",
            "category": "sound_effect",
            "abilities": ["generate"],
            "name": "Text to Sound v2",
            "description": "Sound effects generation from text prompts",
        }
    ]
    list_models.assert_awaited_once_with()
    close.assert_awaited_once_with()


async def test_list_models_uses_fallback_when_key_lacks_user_read(
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
        ElevenLabsSoundEffectModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    models = await ElevenLabsSoundEffectModel.async_list_available_models(
        api_key="test-key"
    )

    assert [model["id"] for model in models] == [ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL]
    list_models.assert_awaited_once_with()
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
        ElevenLabsSoundEffectModel,
        "_create_async_client",
        staticmethod(lambda api_key, base_url=None: fake_client),
    )

    with pytest.raises(RuntimeError, match="invalid API key"):
        await ElevenLabsSoundEffectModel.async_list_available_models(api_key="test-key")

    close.assert_awaited_once_with()


def test_adapter_builds_independent_sound_effect_model() -> None:
    db_model = SimpleNamespace(
        model_id="sound-fx",
        model_provider="elevenlabs",
        model_name=ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL,
        api_key="test-key",
        base_url=None,
        timeout=65,
        max_retries=7,
    )

    model = get_sound_effect_model_instance(db_model)

    assert isinstance(model, ElevenLabsSoundEffectModel)
    assert model.model_name == ELEVENLABS_DEFAULT_SOUND_EFFECT_MODEL
    assert model.abilities == ["generate"]
    assert model.timeout == 65
    assert model.max_retries == 7
