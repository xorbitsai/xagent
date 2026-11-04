"""Mock factory functions for common test scenarios."""

import json
from typing import Any, Dict, List, Optional


def create_langfuse_mock(mocker) -> tuple:
    """Create standard Langfuse mock with common setup.

    Returns:
        tuple: (mock_langfuse_class, mock_langfuse_instance)
    """
    mock_langfuse_class = mocker.patch("xagent.core.tracing.langfuse.client.Langfuse")
    mock_langfuse_instance = mocker.Mock()
    mock_langfuse_class.return_value = mock_langfuse_instance
    return mock_langfuse_class, mock_langfuse_instance


def create_langfuse_span_mock(mocker, langfuse_instance) -> object:
    """Create a mock span for Langfuse tracing tests.

    Args:
        mocker: pytest mocker fixture
        langfuse_instance: Mock langfuse instance

    Returns:
        Mock span object
    """
    mock_span = mocker.Mock()
    # Xagent targets the v4 observation API and still exposes start_span on the
    # mock to keep older tests tolerant of direct span helpers.
    langfuse_instance.start_span.return_value = mock_span
    langfuse_instance.start_observation.return_value = mock_span
    # Also add start_observation to the mock span for nested spans
    mock_span.start_span.return_value = mocker.Mock()
    mock_span.start_observation.return_value = mocker.Mock()
    return mock_span


def create_http_client_mock(
    mocker, response_data: Dict[str, Any], status_code: int = 200
):
    """Create httpx.AsyncClient mock with configurable response.

    Args:
        mocker: pytest mocker fixture
        response_data: Response data to return
        status_code: HTTP status code to return

    Returns:
        Mock HTTP client
    """
    mock_client = mocker.Mock()
    mock_response = mocker.Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_data
    mock_response.text = json.dumps(response_data)

    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response

    return mock_client


def create_openai_model_mock(mocker, responses: List[str]):
    """Create ChatOpenAI mock with predefined responses.

    Args:
        mocker: pytest mocker fixture
        responses: List of response strings to cycle through

    Returns:
        Mock OpenAI model
    """
    mock_model = mocker.Mock()

    # Create AI message responses
    ai_responses = [mocker.Mock(content=response) for response in responses]

    if len(ai_responses) == 1:
        mock_model.ainvoke.return_value = ai_responses[0]
        mock_model.invoke.return_value = ai_responses[0]
    else:
        mock_model.ainvoke.side_effect = ai_responses
        mock_model.invoke.side_effect = ai_responses

    return mock_model


def create_mock_message(mocker, content: str, message_type: str = "human", **kwargs):
    """Create a mock message object.

    Args:
        mocker: pytest mocker fixture
        content: Message content
        message_type: Type of message ('human', 'ai', 'system', 'tool')
        **kwargs: Additional attributes for the message

    Returns:
        Mock message object
    """
    mock_message = mocker.Mock()
    mock_message.content = content
    mock_message.__class__.__name__ = f"{message_type.capitalize()}Message"

    # Add tool_calls attribute for AI messages only
    if message_type.lower() == "ai":
        mock_message.tool_calls = []

    # Add tool_call_id and name for ToolMessage only
    if message_type.lower() == "tool":
        mock_message.tool_call_id = kwargs.get("tool_call_id", "default_call_id")
        mock_message.name = kwargs.get("name", "default_tool")
        # Remove these from kwargs so they don't get added again
        kwargs.pop("tool_call_id", None)
        kwargs.pop("name", None)
    else:
        # For non-tool messages, ensure these attributes don't exist or are None
        mock_message.tool_call_id = None
        mock_message.name = None

    # Add any additional attributes
    for key, value in kwargs.items():
        setattr(mock_message, key, value)

    return mock_message


def create_mock_tool_calls(
    tool_names: List[str], args_list: Optional[List[Dict]] = None
) -> List[Dict]:
    """Create mock tool call data structures.

    Args:
        tool_names: List of tool names
        args_list: Optional list of arguments for each tool call

    Returns:
        List of tool call dictionaries
    """
    if args_list is None:
        args_list = [{"param": f"value_{i}"} for i in range(len(tool_names))]

    return [
        {"name": name, "args": args, "id": f"call_{i + 1}"}
        for i, (name, args) in enumerate(zip(tool_names, args_list))
    ]


def create_temp_config_file(
    temp_dir: str, config_data: Dict[str, Any], filename: str = "config.json"
) -> str:
    """Create a temporary configuration file.

    Args:
        temp_dir: Temporary directory path
        config_data: Configuration data to write
        filename: Name of the config file

    Returns:
        Path to the created config file
    """
    config_path = f"{temp_dir}/{filename}"
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    return config_path
