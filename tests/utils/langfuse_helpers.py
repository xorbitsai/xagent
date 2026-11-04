"""Langfuse-specific test utilities and helpers."""

from typing import Any, Dict

from tests.utils.assertions import (
    assert_dict_contains_keys,
    assert_execution_time_reasonable,
)


def verify_langfuse_span_metadata(
    mock_span,
    expected_node_id: str = None,
    expected_node_type: str = None,
    expected_tool_id: str = None,
    check_execution_time: bool = True,
    check_error: bool = False,
):
    """Verify standard Langfuse span metadata structure.

    Args:
        mock_span: Mock span object to verify
        expected_node_id: Expected node ID in metadata
        expected_node_type: Expected node type in metadata
        expected_tool_id: Expected tool ID in metadata
        check_execution_time: Whether to verify execution time is present
        check_error: Whether to verify error field is present
    """
    # Verify span methods were called
    mock_span.update.assert_called_once()
    mock_span.end.assert_called_once()

    # Get metadata from the call
    call_args = mock_span.update.call_args
    metadata = call_args[1]["metadata"]

    # Check common fields
    if check_execution_time:
        assert "execution_time_seconds" in metadata
        assert_execution_time_reasonable(metadata["execution_time_seconds"])

    # Check node-specific fields
    if expected_node_id:
        assert metadata["node_id"] == expected_node_id

    if expected_node_type:
        assert metadata["node_type"] == expected_node_type

    # Check tool-specific fields
    if expected_tool_id:
        assert metadata["id"] == expected_tool_id

    # Check error field if expected
    if check_error:
        assert "error" in metadata
        assert isinstance(metadata["error"], str)

    return metadata


def verify_langfuse_messages_metadata(
    metadata: Dict[str, Any],
    expected_before_count: int = None,
    expected_after_count: int = None,
):
    """Verify message metadata in Langfuse span.

    Args:
        metadata: Metadata dictionary from span update
        expected_before_count: Expected number of messages before execution
        expected_after_count: Expected number of messages after execution
    """
    if expected_before_count is not None:
        required_keys = ["messages_before", "messages_before_count"]
        assert_dict_contains_keys(metadata, required_keys)
        assert metadata["messages_before_count"] == expected_before_count
        assert len(metadata["messages_before"]) == expected_before_count

        # Verify message structure
        for msg in metadata["messages_before"]:
            assert_dict_contains_keys(msg, ["type", "content"])

    if expected_after_count is not None:
        required_keys = ["messages_after", "messages_after_count"]
        assert_dict_contains_keys(metadata, required_keys)
        assert metadata["messages_after_count"] == expected_after_count
        assert len(metadata["messages_after"]) == expected_after_count

        # Verify message structure
        for msg in metadata["messages_after"]:
            assert_dict_contains_keys(msg, ["type", "content"])
