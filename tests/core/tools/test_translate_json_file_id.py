"""Tests for translate_json tool file_id parameter"""

import json
from unittest.mock import AsyncMock

import pytest

from xagent.core.tools.adapters.vibe.translate_json import TranslateJsonTool
from xagent.core.workspace import TaskWorkspace


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace for testing"""
    workspace = TaskWorkspace(
        id="test_workspace",
        base_dir=str(tmp_path / "workspaces"),
    )
    yield workspace
    # Cleanup
    if workspace.workspace_dir.exists():
        import shutil

        shutil.rmtree(workspace.workspace_dir)


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
def translate_adapter(mock_llm, temp_workspace):
    """Create translate adapter with mock LLM and workspace"""
    return TranslateJsonTool(llm=mock_llm, workspace=temp_workspace)


# File ID parameter tests
@pytest.mark.asyncio
async def test_adapter_with_file_id(translate_adapter, temp_workspace):
    """Test adapter's file_id parameter handling"""
    # Create a test JSON file in workspace
    test_data = {"title": "人工智能", "content": "AI技术"}
    test_file = temp_workspace.output_dir / "test_input.json"

    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False)

    # Register the file to get file_id
    file_id = temp_workspace.register_file(str(test_file))

    args = {
        "file_id": file_id,
        "target_fields": ["title", "content"],
        "target_lang": "en",
    }

    result = await translate_adapter.run_json_async(args)

    assert result["success"] is True
    assert result["fields_translated"] == 2

    # Verify the result contains translated data
    result_json = json.loads(result["result"])
    assert "title_translated_text" in result_json
    assert "content_translated_text" in result_json


@pytest.mark.asyncio
async def test_adapter_file_id_not_found(mock_llm, temp_workspace):
    """Test adapter's error handling when file_id doesn't exist"""
    adapter = TranslateJsonTool(llm=mock_llm, workspace=temp_workspace)

    args = {
        "file_id": "nonexistent-file-id",
        "target_fields": ["text"],
        "target_lang": "en",
    }

    result = await adapter.run_json_async(args)

    assert result["success"] is False
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_adapter_requires_json_or_file_id(mock_llm, temp_workspace):
    """Test that either json_data or file_id must be provided"""
    adapter = TranslateJsonTool(llm=mock_llm, workspace=temp_workspace)

    args = {
        "target_fields": ["text"],
        "target_lang": "en",
    }

    result = await adapter.run_json_async(args)

    assert result["success"] is False
    assert "Either json_data or file_id" in result["error"]


def test_adapter_file_id_without_workspace():
    """Test that file_id requires workspace"""
    from unittest.mock import AsyncMock

    from xagent.core.tools.adapters.vibe.translate_json import TranslateJsonTool

    # Create adapter without workspace but with mock LLM
    mock_llm = AsyncMock()
    adapter = TranslateJsonTool(llm=mock_llm, workspace=None)

    args = {
        "file_id": "some-file-id",
        "target_fields": ["text"],
        "target_lang": "en",
    }

    # Should return error dict with workspace requirement message
    result = adapter.run_json_sync(args)
    assert result["success"] is False
    assert "Workspace is required" in result["error"]
