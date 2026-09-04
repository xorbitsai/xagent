"""Contracts for the wire-safe projection of connector-runtime failures.

The projection has two halves and both are pinned here: the message adapter
(fail-closed on anything that is not a ``ConnectorRuntimeError``) and the
code projector, which is fail-closed the same way.
"""

from __future__ import annotations

import pytest

from xagent.core.tools.adapters.vibe.config import RequiredMCPUnavailableError
from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.services.client_error_messages import (
    CLIENT_SAFE_TASK_FAILURE,
    connector_runtime_client_code,
    connector_runtime_client_message,
)

# --------------------------------------------------------------------------
# connector_runtime_client_message
# --------------------------------------------------------------------------


def test_client_message_returns_the_curated_safe_message() -> None:
    error = ConnectorRuntimeError(
        "missing_runtime_context",
        "Required connector runtime context is missing.",
    )

    assert (
        connector_runtime_client_message(error)
        == "Required connector runtime context is missing."
    )


@pytest.mark.parametrize("safe_message", ["", "   ", "\n\t"])
def test_client_message_falls_back_on_a_blank_safe_message(safe_message: str) -> None:
    error = ConnectorRuntimeError("missing_runtime_context", safe_message)

    assert connector_runtime_client_message(error) == CLIENT_SAFE_TASK_FAILURE


@pytest.mark.parametrize(
    "error",
    [
        ValueError("secret-token-xyz"),
        KeyError("secret-token-xyz"),
        RuntimeError("secret-token-xyz"),
        RequiredMCPUnavailableError("secret-token-xyz"),
    ],
)
def test_client_message_is_fail_closed_for_an_incidental_exception(
    error: BaseException,
) -> None:
    """The specific name is not the gate; the isinstance check is."""

    assert connector_runtime_client_message(error) == CLIENT_SAFE_TASK_FAILURE


# --------------------------------------------------------------------------
# connector_runtime_client_code
# --------------------------------------------------------------------------


def test_client_code_projects_a_connector_runtime_error() -> None:
    error = ConnectorRuntimeError(
        "missing_runtime_context",
        "Required connector runtime context is missing.",
    )

    assert connector_runtime_client_code(error) == "missing_runtime_context"


@pytest.mark.parametrize(
    "error",
    [
        ValueError("secret-token-xyz"),
        KeyError("secret-token-xyz"),
        RuntimeError("secret-token-xyz"),
        RequiredMCPUnavailableError("secret-token-xyz"),
    ],
)
def test_client_code_is_fail_closed_for_an_incidental_exception(
    error: BaseException,
) -> None:
    """The specific name is not the gate; the isinstance check is."""

    assert connector_runtime_client_code(error) is None
