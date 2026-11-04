"""Test cases for the OpenAI-compatible image model."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import RootModel

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


def _sdk_response_payload():
    return {
        "created": 1,
        "data": [{"b64_json": "aGVsbG8="}],
        "usage": {
            "input_tokens": 5,
            "input_tokens_details": {"image_tokens": 2, "text_tokens": 3},
            "output_tokens": 7,
            "output_tokens_details": {"image_tokens": 7, "text_tokens": 0},
            "total_tokens": 12,
        },
    }


def _sdk_model(handler):
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = OpenAIImageModel(model_name="gpt-image-1", api_key="test-key")
    model._client = AsyncOpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=http_client,
    )
    return model, http_client


class _ToolReturn(RootModel[dict]):
    """The return boundary FunctionTool applies to dict-returning tools."""


class TestSdkUsageSerialization:
    @pytest.mark.asyncio
    async def test_generate_returns_plain_usage_for_tool_result(self):
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(200, json=_sdk_response_payload())

        model, http_client = _sdk_model(handler)
        try:
            result = await model.generate_image(
                prompt="a cat",
                transparent_background=True,
            )
        finally:
            await http_client.aclose()

        assert len(requests) == 1
        request = json.loads(requests[0].content)
        assert request["background"] == "transparent"
        assert request["output_format"] == "png"
        assert result["image_url"] == "data:image/png;base64,aGVsbG8="
        assert type(result["usage"]) is dict
        assert result["usage"] == _sdk_response_payload()["usage"]
        assert (
            _ToolReturn.model_validate(result).model_dump()["usage"] == result["usage"]
        )

    @pytest.mark.asyncio
    async def test_edit_returns_plain_usage_for_tool_result(self, tmp_path):
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(200, json=_sdk_response_payload())

        source = tmp_path / "in.png"
        source.write_bytes(b"stub")
        model, http_client = _sdk_model(handler)
        try:
            result = await model.edit_image(
                image_url=str(source),
                prompt="cut out the cat",
                transparent_background=True,
            )
        finally:
            await http_client.aclose()

        assert len(requests) == 1
        assert b'name="background"' in requests[0].content
        assert b"transparent" in requests[0].content
        assert b'name="output_format"' in requests[0].content
        assert b"png" in requests[0].content
        assert result["image_url"] == "data:image/png;base64,aGVsbG8="
        assert type(result["usage"]) is dict
        assert result["usage"] == _sdk_response_payload()["usage"]
        assert (
            _ToolReturn.model_validate(result).model_dump()["usage"] == result["usage"]
        )

    @pytest.mark.asyncio
    async def test_usage_dump_failure_does_not_fail_generated_image(
        self, caplog, monkeypatch
    ):
        events = []

        def record_usage(*args, **kwargs):
            events.append("metered")

        monkeypatch.setattr(
            "xagent.core.model.image.openai.record_image_usage", record_usage
        )

        class BrokenUsage:
            input_tokens = 5
            output_tokens = 7
            total_tokens = 12

            def model_dump(self, **kwargs):
                events.append("dumped")
                raise TypeError("broken usage serializer")

        model, client = _model("gpt-image-1")
        client.images.generate.return_value = SimpleNamespace(
            data=[SimpleNamespace(url="https://example.com/out.png", b64_json=None)],
            usage=BrokenUsage(),
            id="req-1",
        )

        result = await model.generate_image(prompt="a cat")

        client.images.generate.assert_awaited_once()
        assert result["image_url"] == "https://example.com/out.png"
        assert result["usage"] == {}
        assert events == ["metered", "dumped"]
        assert "after metering" in caplog.text


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
