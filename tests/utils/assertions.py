"""Custom assertion helpers for common test patterns."""

from typing import Any, Dict, List


def assert_cli_success_response(console_mock, expected_message: str = None):
    """Verify CLI success response pattern.

    Args:
        console_mock: Mock console object
        expected_message: Expected success message
    """
    console_mock.print.assert_called()
    if expected_message:
        # Check if any call contains the expected message
        calls = console_mock.print.call_args_list
        messages = [str(call[0][0]) for call in calls]
        assert any(expected_message in message for message in messages), (
            f"Expected message '{expected_message}' not found in {messages}"
        )


def assert_cli_error_response(console_mock, expected_error: str = None):
    """Verify CLI error response pattern.

    Args:
        console_mock: Mock console object
        expected_error: Expected error message
    """
    console_mock.print.assert_called()
    if expected_error:
        calls = console_mock.print.call_args_list
        messages = [str(call[0][0]) for call in calls]
        assert any(expected_error in message for message in messages), (
            f"Expected error '{expected_error}' not found in {messages}"
        )


def assert_dict_contains_keys(data: Dict[str, Any], required_keys: List[str]):
    """Assert that dictionary contains all required keys.

    Args:
        data: Dictionary to check
        required_keys: List of required keys
    """
    missing_keys = [key for key in required_keys if key not in data]
    assert not missing_keys, f"Missing required keys: {missing_keys}"


def assert_message_structure(message_data: Dict[str, Any], expected_type: str = None):
    """Assert message has expected structure.

    Args:
        message_data: Message data dictionary
        expected_type: Expected message type
    """
    required_keys = ["type", "content"]
    assert_dict_contains_keys(message_data, required_keys)

    if expected_type:
        assert message_data["type"] == expected_type


def assert_model_data_valid(model_data: Dict[str, Any]):
    """Assert model data has valid structure.

    Args:
        model_data: Model data dictionary
    """
    required_keys = ["id", "model", "temperature", "api_key"]
    assert_dict_contains_keys(model_data, required_keys)

    # Type checks
    assert isinstance(model_data["temperature"], (int, float))
    assert 0 <= model_data["temperature"] <= 2
    assert isinstance(model_data["api_key"], str)
    assert len(model_data["api_key"]) > 0


def assert_http_response_structure(
    response_data: Dict[str, Any], expected_status: str = None
):
    """Assert HTTP response has expected structure.

    Args:
        response_data: Response data dictionary
        expected_status: Expected status value
    """
    if expected_status:
        assert response_data.get("status") == expected_status


def assert_execution_time_reasonable(
    execution_time: float, min_time: float = 0.0, max_time: float = 10.0
):
    """Assert execution time is within reasonable bounds.

    Args:
        execution_time: Actual execution time
        min_time: Minimum expected time
        max_time: Maximum expected time
    """
    assert isinstance(execution_time, (int, float)), (
        f"Execution time should be numeric, got {type(execution_time)}"
    )
    assert min_time <= execution_time <= max_time, (
        f"Execution time {execution_time} not in range [{min_time}, {max_time}]"
    )


def assert_file_exists_with_content(file_path: str, expected_content: str = None):
    """Assert file exists and optionally check content.

    Args:
        file_path: Path to file
        expected_content: Expected file content (partial match)
    """
    import os

    assert os.path.exists(file_path), f"File {file_path} does not exist"

    if expected_content:
        with open(file_path, "r") as f:
            content = f.read()
            assert expected_content in content, (
                f"Expected content '{expected_content}' not found in file"
            )


def assert_json_structure_matches(
    actual: Dict[str, Any], expected_structure: Dict[str, type]
):
    """Assert JSON data matches expected structure.

    Args:
        actual: Actual JSON data
        expected_structure: Dictionary mapping keys to expected types
    """
    for key, expected_type in expected_structure.items():
        assert key in actual, f"Missing key '{key}' in JSON data"
        assert isinstance(actual[key], expected_type), (
            f"Key '{key}' should be {expected_type}, got {type(actual[key])}"
        )
