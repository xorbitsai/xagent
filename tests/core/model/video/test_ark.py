from types import SimpleNamespace

import pytest

from xagent.core.model.video.ark import (
    ARK_BYTEPLUS_BASE_URL,
    ARK_DOMESTIC_BASE_URL,
    ArkVideoModel,
    _size_to_seedance_params,
)


def test_dreamina_models_default_to_byteplus_base_url(monkeypatch):
    monkeypatch.delenv("ARK_BASE_URL", raising=False)
    monkeypatch.delenv("MODELARK_BASE_URL", raising=False)

    domestic = ArkVideoModel(
        model_name="doubao-seedance-2-0-fast-260128",
        api_key="test-key",
    )
    byteplus = ArkVideoModel(
        model_name="dreamina-seedance-2-0-fast-260128",
        api_key="test-key",
    )

    assert domestic.base_url == ARK_DOMESTIC_BASE_URL
    assert domestic._uses_byteplus_sdk is False
    assert byteplus.base_url == ARK_BYTEPLUS_BASE_URL
    assert byteplus._uses_byteplus_sdk is True


def test_size_to_seedance_params_uses_standard_ratios_for_near_sizes():
    assert _size_to_seedance_params("854x480") == ("16:9", "480p")
    assert _size_to_seedance_params("480x853") == ("9:16", "480p")


@pytest.mark.asyncio
async def test_generate_video_creates_and_polls_task():
    class FakeTasks:
        def __init__(self):
            self.create_calls = []
            self.get_calls = []

        def create(self, *, model, content, **kwargs):
            self.create_calls.append(
                {"model": model, "content": content, "kwargs": kwargs}
            )
            return SimpleNamespace(id="task-1")

        def get(self, *, task_id):
            self.get_calls.append(task_id)
            return {
                "id": task_id,
                "model": "doubao-seedance-2-0-fast-260128",
                "status": "succeeded",
                "content": {
                    "video_url": "https://example.com/result.mp4",
                    "last_frame_url": "https://example.com/last.png",
                },
                "seed": 42,
                "duration": 5,
                "ratio": "16:9",
            }

    class FakeArk:
        instances = []

        def __init__(self, *, base_url, api_key):
            self.base_url = base_url
            self.api_key = api_key
            self.content_generation = SimpleNamespace(tasks=FakeTasks())
            FakeArk.instances.append(self)

    model = ArkVideoModel(
        model_name="doubao-seedance-2-0-fast-260128",
        api_key="test-key",
        base_url=ARK_DOMESTIC_BASE_URL,
    )
    model._load_ark_class = lambda: FakeArk  # type: ignore[method-assign]

    result = await model.generate_video(
        prompt="A cinematic product shot",
        ratio="16:9",
        duration=5,
        generate_audio=True,
        watermark=False,
        poll_interval=0.01,
    )

    assert result["task_id"] == "task-1"
    assert result["status"] == "succeeded"
    assert result["video_url"] == "https://example.com/result.mp4"
    assert result["last_frame_url"] == "https://example.com/last.png"
    assert result["seed"] == 42

    tasks = FakeArk.instances[0].content_generation.tasks
    assert tasks.create_calls == [
        {
            "model": "doubao-seedance-2-0-fast-260128",
            "content": [
                {"type": "text", "text": "A cinematic product shot"},
            ],
            "kwargs": {
                "ratio": "16:9",
                "duration": 5,
                "generate_audio": True,
                "watermark": False,
            },
        }
    ]
    assert tasks.get_calls == ["task-1"]


@pytest.mark.asyncio
async def test_generate_video_retries_transient_polling_errors():
    class FakeTasks:
        def __init__(self):
            self.get_calls = []

        def create(self, *, model, content, **kwargs):
            return SimpleNamespace(id="task-1")

        def get(self, *, task_id):
            self.get_calls.append(task_id)
            if len(self.get_calls) == 1:
                raise RuntimeError("temporary network issue")
            return {
                "id": task_id,
                "model": "doubao-seedance-2-0-fast-260128",
                "status": "succeeded",
                "content": {"video_url": "https://example.com/result.mp4"},
            }

    class FakeArk:
        instances = []

        def __init__(self, *, base_url, api_key):
            self.content_generation = SimpleNamespace(tasks=FakeTasks())
            FakeArk.instances.append(self)

    model = ArkVideoModel(
        model_name="doubao-seedance-2-0-fast-260128",
        api_key="test-key",
        base_url=ARK_DOMESTIC_BASE_URL,
    )
    model._load_ark_class = lambda: FakeArk  # type: ignore[method-assign]

    result = await model.generate_video(
        prompt="A cinematic product shot",
        poll_interval=0.01,
        timeout=2.0,
    )

    assert result["status"] == "succeeded"
    assert result["video_url"] == "https://example.com/result.mp4"
    assert FakeArk.instances[0].content_generation.tasks.get_calls == [
        "task-1",
        "task-1",
    ]


@pytest.mark.asyncio
async def test_duration_values_are_passed_through_without_guessing():
    class FakeTasks:
        def __init__(self):
            self.create_calls = []

        def create(self, *, model, content, **kwargs):
            self.create_calls.append(
                {"model": model, "content": content, "kwargs": kwargs}
            )
            return {"id": "task-1"}

        def get(self, *, task_id):
            return {
                "id": task_id,
                "status": "succeeded",
                "content": {"video_url": "https://example.com/result.mp4"},
            }

    class FakeArk:
        instances = []

        def __init__(self, *, base_url, api_key):
            self.content_generation = SimpleNamespace(tasks=FakeTasks())
            FakeArk.instances.append(self)

    model = ArkVideoModel(
        model_name="doubao-seedance-1-5-pro-251215",
        api_key="test-key",
        base_url=ARK_DOMESTIC_BASE_URL,
    )
    model._load_ark_class = lambda: FakeArk  # type: ignore[method-assign]

    await model.generate_video(
        prompt="A short clip",
        seconds=4.5,
        ratio="16:9",
        resolution="480p",
        poll_interval=0.01,
    )

    tasks = FakeArk.instances[0].content_generation.tasks
    assert tasks.create_calls[0]["kwargs"]["duration"] == 4.5


@pytest.mark.asyncio
async def test_openai_compatible_params_are_translated_for_seedance():
    class FakeTasks:
        def __init__(self):
            self.create_calls = []

        def create(self, *, model, content, **kwargs):
            self.create_calls.append(
                {"model": model, "content": content, "kwargs": kwargs}
            )
            return {"id": "task-1"}

        def get(self, *, task_id):
            return {
                "id": task_id,
                "status": "succeeded",
                "content": {"video_url": "https://example.com/result.mp4"},
            }

    class FakeArk:
        instances = []

        def __init__(self, *, base_url, api_key):
            self.content_generation = SimpleNamespace(tasks=FakeTasks())
            FakeArk.instances.append(self)

    model = ArkVideoModel(
        model_name="doubao-seedance-2-0-fast-260128",
        api_key="test-key",
        base_url=ARK_DOMESTIC_BASE_URL,
    )
    model._load_ark_class = lambda: FakeArk  # type: ignore[method-assign]

    await model.generate_video(
        prompt="Animate this frame",
        seconds=5,
        size="720x1280",
        input_reference="https://example.com/frame.png",
        negative_prompt="blur",
        n=1,
        poll_interval=0.01,
    )

    tasks = FakeArk.instances[0].content_generation.tasks
    assert tasks.create_calls == [
        {
            "model": "doubao-seedance-2-0-fast-260128",
            "content": [
                {"type": "text", "text": "Animate this frame"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/frame.png"},
                    "role": "first_frame",
                },
            ],
            "kwargs": {
                "duration": 5,
                "ratio": "9:16",
                "resolution": "720p",
            },
        }
    ]


@pytest.mark.asyncio
async def test_audio_reference_requires_visual_reference():
    model = ArkVideoModel(api_key="test-key")

    with pytest.raises(ValueError, match="audio references require"):
        await model.generate_video(
            prompt="",
            reference_audio_urls=["https://example.com/audio.mp3"],
        )
