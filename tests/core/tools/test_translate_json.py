"""Tests for translate_json tool"""

import json
from unittest.mock import AsyncMock

import pytest

from xagent.core.tools.adapters.vibe.translate_json import TranslateJsonTool
from xagent.core.tools.core.translate_json_tool import TranslateJSONToolCore


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns translations"""

    async def mock_chat(messages):
        # Extract number of texts from the prompt
        prompt = messages[0]["content"] if messages else ""
        # Count the number of "N." patterns to determine expected translations
        lines = prompt.split("\n")
        count = sum(
            1 for line in lines if line.strip() and line[0].isdigit() and "." in line
        )

        # Return that many translations
        translations = [f"Translation{i + 1}" for i in range(count)]
        return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(translations))

    async def mock_stream_chat(messages):
        """Mock stream_chat that yields chunks"""
        from xagent.core.model.chat.types import ChunkType, StreamChunk

        # Get the response from regular chat
        response = await mock_chat(messages)

        # Yield as a single chunk
        yield StreamChunk(
            type=ChunkType.TOKEN,
            content=response,
            delta=response,
        )

    llm = AsyncMock()
    llm.chat = AsyncMock(side_effect=mock_chat)
    llm.stream_chat = mock_stream_chat
    return llm


@pytest.fixture
def translate_tool(mock_llm):
    """Create translate tool with mock LLM"""
    return TranslateJSONToolCore(llm=mock_llm)


@pytest.fixture
def translate_adapter(mock_llm):
    """Create translate adapter with mock LLM"""
    return TranslateJsonTool(llm=mock_llm)


@pytest.mark.asyncio
async def test_translate_simple_json(translate_tool):
    """Test translating simple JSON structure"""
    json_data = {"text": "你好", "content": "世界"}
    target_fields = ["text", "content"]

    result = await translate_tool.translate_json(
        json_data=json_data,
        target_fields=target_fields,
        output_field="translated_text",
        target_lang="en",
        source_lang="zh",
    )

    assert result["success"] is True

    result_json = json.loads(result["result"])

    assert "text" in result_json
    assert result_json["text"] == "你好"
    assert "content" in result_json
    assert result_json["content"] == "世界"
    # Field-specific output fields to avoid overwriting
    assert "text_translated_text" in result_json
    assert result_json["text_translated_text"] == "Translation1"
    assert "content_translated_text" in result_json
    assert result_json["content_translated_text"] == "Translation2"


@pytest.mark.asyncio
async def test_translate_nested_json(translate_tool):
    """Test translating nested JSON structure like segments"""
    json_data = {
        "segments": [
            {"text": "你好", "start": 0.0, "end": 1.0},
            {"text": "世界", "start": 1.0, "end": 2.0},
        ]
    }
    target_fields = ["segments.text"]

    result = await translate_tool.translate_json(
        json_data=json_data,
        target_fields=target_fields,
        output_field="translated_text",
        target_lang="en",
        source_lang="zh",
    )

    assert result["success"] is True
    assert result["fields_translated"] == 2

    result_json = json.loads(result["result"])

    assert "segments" in result_json
    assert len(result_json["segments"]) == 2
    assert result_json["segments"][0]["text"] == "你好"
    assert "translated_text" in result_json["segments"][0]
    assert result_json["segments"][0]["translated_text"] == "Translation1"
    assert result_json["segments"][1]["translated_text"] == "Translation2"


@pytest.mark.asyncio
async def test_translate_with_different_output_field(translate_tool):
    """Test translating with custom output field name"""
    json_data = {"title": "标题"}
    target_fields = ["title"]

    result = await translate_tool.translate_json(
        json_data=json_data,
        target_fields=target_fields,
        output_field="en_title",
        target_lang="en",
    )

    assert result["success"] is True

    result_json = json.loads(result["result"])

    assert "title" in result_json
    assert result_json["title"] == "标题"
    assert "title_en_title" in result_json
    assert result_json["title_en_title"] == "Translation1"


@pytest.mark.asyncio
async def test_get_field_value_nested():
    """Test _get_field_value with nested structures"""
    tool = TranslateJSONToolCore()
    data = {
        "segments": [
            {"text": "Hello", "start": 0.0},
            {"text": "World", "start": 1.0},
        ]
    }

    results = tool._get_field_value(data, "segments.text")

    assert len(results) == 2
    assert results[0]["value"] == "Hello"
    assert results[1]["value"] == "World"


@pytest.mark.asyncio
async def test_translate_values_batch(translate_tool, mock_llm):
    """Test batch translation of multiple texts"""
    texts = ["你好", "世界", "测试"]

    translations = await translate_tool.translate_values(
        texts=texts, target_lang="en", source_lang="zh"
    )

    assert len(translations) == 3
    # Since we now use stream_chat, we can't check if chat was called
    # but we can verify the translations worked correctly
    assert translations == ["Translation1", "Translation2", "Translation3"]


@pytest.mark.asyncio
async def test_translate_json_string_input(translate_tool):
    """Test translating when input is JSON string instead of dict"""
    json_str = '{"text": "你好"}'
    target_fields = ["text"]

    result = await translate_tool.translate_json(
        json_data=json_str,
        target_fields=target_fields,
        target_lang="en",
    )

    assert result["success"] is True

    result_json = json.loads(result["result"])

    assert "text" in result_json
    assert result_json["text"] == "你好"
    assert "text_translated_text" in result_json
    assert result_json["text_translated_text"] == "Translation1"


@pytest.mark.asyncio
async def test_no_matching_fields():
    """Test when no fields match the target paths"""
    tool = TranslateJSONToolCore()
    json_data = {"foo": "bar"}
    target_fields = ["text"]

    result = await tool.translate_json(
        json_data=json_data,
        target_fields=target_fields,
        target_lang="en",
    )

    assert result["success"] is False
    assert "No fields found" in result["error"]


def test_translate_json_sync_wrapper():
    """Test synchronous wrapper function"""
    from xagent.core.tools.core.translate_json_tool import translate_json

    json_data = {"text": "测试"}
    target_fields = ["text"]

    # This should handle the sync/async conversion
    # Note: This test may fail if no LLM is configured
    try:
        result = translate_json(json_data, target_fields, target_lang="en")
        # Now returns a dictionary instead of a string
        assert isinstance(result, dict)
        assert "success" in result
        assert "result" in result
    except ValueError as e:
        # Expected if no LLM is configured
        assert "No LLM instance available" in str(e)


# Adapter layer tests
@pytest.mark.asyncio
async def test_adapter_run_json_async(translate_adapter):
    """Test adapter's async execution method"""
    args = {
        "json_data": '{"text": "你好"}',
        "target_fields": ["text"],
        "target_lang": "en",
    }

    result = await translate_adapter.run_json_async(args)

    assert result["success"] is True
    assert result["fields_translated"] == 1
    assert result["target_lang"] == "en"

    # Verify result contains translated JSON
    translated_data = json.loads(result["result"])
    assert "text_translated_text" in translated_data


def test_adapter_run_json_sync(translate_adapter):
    """Test adapter's sync execution method"""
    args = {
        "json_data": '{"text": "你好"}',
        "target_fields": ["text"],
        "target_lang": "en",
    }

    result = translate_adapter.run_json_sync(args)

    assert result["success"] is True
    assert result["fields_translated"] == 1
    assert result["target_lang"] == "en"


def test_adapter_requires_llm():
    """Test that adapter asserts LLM is available during execution"""
    # Create adapter without LLM
    adapter = TranslateJsonTool(llm=None)

    args = {
        "json_data": '{"text": "你好"}',
        "target_fields": ["text"],
        "target_lang": "en",
    }

    # Should raise AssertionError when trying to execute without LLM
    with pytest.raises(AssertionError, match="requires an LLM"):
        adapter.run_json_sync(args)
