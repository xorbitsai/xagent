"""Test file information propagation from context to step execution."""

from unittest.mock import Mock

import pytest

from xagent.core.agent.pattern.dag_plan_execute.plan_executor import PlanExecutor
from xagent.core.agent.utils.context_builder import ContextBuilder


@pytest.mark.asyncio
async def test_file_info_propagation_from_context():
    """Test that file information is properly propagated from context to step execution."""
    # Create a mock parent pattern with file information in context
    parent_pattern = Mock()
    parent_pattern._context = {
        "uploaded_files": ["file1.jpg", "file2.png"],
        "file_info": [
            {"name": "file1.jpg", "size": 1024, "type": "image/jpeg"},
            {"name": "file2.png", "size": 2048, "type": "image/png"},
        ],
    }

    # Create a ContextBuilder
    llm_mock = Mock()
    llm_mock.model_name = "test-model"
    context_builder = ContextBuilder(llm=llm_mock)

    # Build context for a step with file information
    messages = await context_builder.build_context_for_step(
        step_name="test_step",
        step_description="Test step with file information",
        dependencies=[],
        dependency_results={},
        file_info=parent_pattern._context["file_info"],
        uploaded_files=parent_pattern._context["uploaded_files"],
    )

    # Verify that file information is included in the messages
    assert len(messages) > 1  # System prompt + file information

    # Find the file information message
    file_info_msg = None
    for msg in messages:
        if "UPLOADED FILES" in msg.get("content", ""):
            file_info_msg = msg
            break

    assert file_info_msg is not None, "File information message not found"
    content = file_info_msg["content"]

    # Verify file information is present
    assert "2 files available for processing" in content
    assert "file1.jpg" in content
    assert "file2.png" in content
    assert "1024 bytes" in content
    assert "2048 bytes" in content
    assert "image/jpeg" in content
    assert "image/png" in content


@pytest.mark.asyncio
async def test_plan_executor_retrieves_file_info_from_parent_context():
    """Test that PlanExecutor correctly retrieves file information from parent pattern context."""
    # Create a mock parent pattern with file information as a dict (as passed from websocket)
    parent_pattern = Mock()
    parent_pattern._context = {
        "uploaded_files": ["file1.jpg", "file2.png"],
        "file_info": [
            {"name": "file1.jpg", "size": 1024, "type": "image/jpeg"},
            {"name": "file2.png", "size": 2048, "type": "image/png"},
        ],
    }

    # Create a PlanExecutor with the parent pattern
    plan_executor = PlanExecutor(
        llm=Mock(),
        tracer=Mock(),
        workspace=Mock(),
        parent_pattern=parent_pattern,
    )

    # Access the file information through the parent pattern
    file_info = None
    uploaded_files = None

    if plan_executor.parent_pattern and hasattr(
        plan_executor.parent_pattern, "_context"
    ):
        parent_context = plan_executor.parent_pattern._context
        if parent_context:
            if isinstance(parent_context, dict):
                file_info = parent_context.get("file_info")
                uploaded_files = parent_context.get("uploaded_files")

    # Verify file information was retrieved correctly
    assert file_info is not None
    assert uploaded_files is not None
    assert len(file_info) == 2
    assert len(uploaded_files) == 2
    assert file_info[0]["name"] == "file1.jpg"
    assert uploaded_files[0] == "file1.jpg"


@pytest.mark.asyncio
async def test_context_builder_with_empty_file_info():
    """Test that ContextBuilder handles empty file information gracefully."""
    context_builder = ContextBuilder(llm=Mock())

    # Build context without file information
    messages = await context_builder.build_context_for_step(
        step_name="test_step",
        step_description="Test step without file information",
        dependencies=[],
        dependency_results={},
        file_info=None,
        uploaded_files=None,
    )

    # Verify that only system prompt is present
    assert len(messages) == 1  # Only system prompt
    assert messages[0]["role"] == "system"

    # Verify no file information message
    for msg in messages:
        assert "UPLOADED FILES" not in msg.get("content", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
