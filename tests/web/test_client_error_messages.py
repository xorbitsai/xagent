from xagent.core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from xagent.web.services.client_error_messages import (
    CLIENT_SAFE_TASK_FAILURE,
    required_mcp_unavailable_client_message,
)


def test_required_mcp_error_preserves_its_curated_client_message() -> None:
    error = RequiredMCPUnavailableError([])

    assert required_mcp_unavailable_client_message(error) == str(error)


def test_required_mcp_adapter_rejects_incidental_exceptions() -> None:
    error = RuntimeError("provider token=secret")

    assert (
        required_mcp_unavailable_client_message(
            error,
            fallback=CLIENT_SAFE_TASK_FAILURE,
        )
        == CLIENT_SAFE_TASK_FAILURE
    )
