"""Test cases for the OpenAI-compatible image model."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from xagent.core.model.image.openai import OpenAIImageModel


def _response(url="https://example.com/out.png"):
    return SimpleNamespace(
        data=[SimpleNamespace(url=url, b64_json=None)],
        usage={"total_tokens": 1},
        id="req-1",
    )


def _model(model_name):
    model = OpenAIImageModel(model_name=model_name, api_key="test-key")
    client = MagicMock()
    client.images.generate = AsyncMock(return_value=_response())
    client.images.edit = AsyncMock(return_value=_response())
    model._client = client
    return model, client


class TestTransparencyCapability:
    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-image-1",
            "gpt-image-1-mini",
            # Proxies routinely namespace the upstream name.
            "openai.gpt-image-1",
            "GPT-Image-1",
        ],
    )
    def test_gpt_image_family_supports_transparency(self, model_name):
        model = OpenAIImageModel(model_name=model_name, api_key="k")
        assert model.supports_transparent_background is True

    @pytest.mark.parametrize("model_name", ["dall-e-3", "dall-e-2", "flux-pro"])
    def test_everything_else_does_not(self, model_name):
        model = OpenAIImageModel(model_name=model_name, api_key="k")
        assert model.supports_transparent_background is False


class TestGenerateRequestShape:
    @pytest.mark.asyncio
    async def test_transparency_asks_for_alpha_and_png(self):
        model, client = _model("gpt-image-1")

        await model.generate_image(prompt="a logo", transparent_background=True)

        kwargs = client.images.generate.call_args.kwargs
        assert kwargs["background"] == "transparent"
        # Without this the endpoint may answer webp or jpeg and flatten the
        # alpha that background=transparent just produced.
        assert kwargs["output_format"] == "png"

    @pytest.mark.asyncio
    async def test_gpt_image_never_receives_response_format(self):
        # gpt-image-* rejects the field outright and always answers b64_json.
        model, client = _model("gpt-image-1")

        await model.generate_image(prompt="a logo")

        assert "response_format" not in client.images.generate.call_args.kwargs

    @pytest.mark.asyncio
    async def test_other_models_still_receive_response_format(self):
        model, client = _model("dall-e-3")

        await model.generate_image(prompt="a logo")

        assert client.images.generate.call_args.kwargs["response_format"] == "url"

    @pytest.mark.asyncio
    async def test_explicit_response_format_is_honoured(self):
        model, client = _model("dall-e-3")

        await model.generate_image(prompt="a logo", response_format="b64_json")

        assert client.images.generate.call_args.kwargs["response_format"] == "b64_json"

    @pytest.mark.asyncio
    async def test_no_transparency_kwargs_when_not_requested(self):
        model, client = _model("gpt-image-1")

        await model.generate_image(prompt="a logo")

        kwargs = client.images.generate.call_args.kwargs
        assert "background" not in kwargs
        assert "output_format" not in kwargs

    @pytest.mark.asyncio
    async def test_incapable_model_raises_instead_of_returning_opaque(self):
        model, client = _model("dall-e-3")

        with pytest.raises(RuntimeError, match="gpt-image"):
            await model.generate_image(prompt="a logo", transparent_background=True)

        client.images.generate.assert_not_awaited()


class TestEditRequestShape:
    @pytest.mark.asyncio
    async def test_transparency_asks_for_alpha_and_png(self, tmp_path):
        source = tmp_path / "in.png"
        source.write_bytes(b"stub")
        model, client = _model("gpt-image-1")

        await model.edit_image(
            image_url=str(source),
            prompt="cut the product out",
            transparent_background=True,
        )

        kwargs = client.images.edit.call_args.kwargs
        assert kwargs["background"] == "transparent"
        assert kwargs["output_format"] == "png"
        assert "response_format" not in kwargs

    @pytest.mark.asyncio
    async def test_size_is_passed_once(self, tmp_path):
        # size is popped out of kwargs before the rest is forwarded, so it must
        # not also arrive as a duplicate keyword.
        source = tmp_path / "in.png"
        source.write_bytes(b"stub")
        model, client = _model("gpt-image-1")

        await model.edit_image(image_url=str(source), prompt="tweak", size="512*512")

        assert client.images.edit.call_args.kwargs["size"] == "512x512"


class TestSizeDefaulting:
    """An explicitly forwarded None is not the same as a missing key."""

    @pytest.mark.asyncio
    async def test_generate_falls_back_when_size_is_none(self):
        # image_tool always puts "size" in its params dict, so an unset size
        # arrives as an explicit None that a parameter default never covers.
        model, client = _model("gpt-image-1")

        await model.generate_image(prompt="a logo", size=None)

        assert client.images.generate.call_args.kwargs["size"] == "1024x1024"

    @pytest.mark.asyncio
    async def test_edit_falls_back_when_size_is_none(self, tmp_path):
        source = tmp_path / "in.png"
        source.write_bytes(b"stub")
        model, client = _model("gpt-image-1")

        await model.edit_image(image_url=str(source), prompt="tweak", size=None)

        assert client.images.edit.call_args.kwargs["size"] == "1024x1024"
