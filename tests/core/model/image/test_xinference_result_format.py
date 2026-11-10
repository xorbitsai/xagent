"""Private Xinference hosts must not require a second, unsafe result fetch."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from xagent.core.model.image.xinference import XinferenceImageModel


@pytest.mark.parametrize("method", ["generate", "edit"])
@pytest.mark.parametrize("explicit", [None, "url"])
async def test_xinference_defaults_to_inline_data_and_honors_explicit_format(
    monkeypatch, method, explicit
):
    model = XinferenceImageModel(model_name="sd", abilities=["generate", "edit"])
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    payload = (
        {"url": "https://example.com/image.png"}
        if explicit
        else {"b64_json": "aW1hZ2U="}
    )
    call = Mock(return_value={"data": [payload]})
    monkeypatch.setattr(
        model, "_model_handle", SimpleNamespace(text_to_image=call, image_to_image=call)
    )
    kwargs = {"response_format": explicit} if explicit else {}
    if method == "generate":
        result = await model.generate_image(prompt="fixture", **kwargs)
    else:
        result = await model.edit_image(
            image_url="fixture.png", prompt="fixture", **kwargs
        )
    assert call.call_args.kwargs["response_format"] == (explicit or "b64_json")
    assert result["image_url"] == (
        "https://example.com/image.png"
        if explicit
        else "data:image/png;base64,aW1hZ2U="
    )


async def test_xinference_keeps_size_when_aspect_ratio_is_set(monkeypatch):
    """The adapter ignores resolution/width+height when a ratio is given, so
    size is the pixel size the provider receives."""
    model = XinferenceImageModel(model_name="sd", abilities=["generate"])
    monkeypatch.setattr(model, "_ensure_client", lambda: None)
    call = Mock(return_value={"data": [{"b64_json": "aW1hZ2U="}]})
    monkeypatch.setattr(model, "_model_handle", SimpleNamespace(text_to_image=call))

    await model.generate_image(
        prompt="fixture",
        size="1080*1350",
        resolution="999x999",
        aspect_ratio="4:5",
    )

    assert call.call_args.kwargs["size"] == "1080*1350"
