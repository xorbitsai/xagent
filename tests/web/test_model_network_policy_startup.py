"""Bad operator model or MCP policy fails before either host admits work."""

from unittest.mock import Mock

import pytest

from xagent.core.utils.security import PrivateNetworkHostError


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["web", "worker"])
@pytest.mark.parametrize(
    "variable,value",
    [
        ("XAGENT_MODEL_PRIVATE_ENDPOINTS", "http://localhost:9997/not-an-origin"),
        ("XAGENT_MODEL_PRIVATE_CIDRS", "10.0.0.1/8"),
        ("XAGENT_MCP_PRIVATE_ENDPOINTS", "http://mcp-tools:8000/mcp"),
        ("XAGENT_MCP_PRIVATE_CIDRS", "169.254.0.0/16"),
    ],
)
async def test_invalid_network_policy_fails_before_host_initialization(
    monkeypatch, host, variable, value
):
    monkeypatch.setenv(variable, value)
    init = Mock(side_effect=AssertionError("host admitted before policy validation"))
    if host == "web":
        from xagent.web import app

        monkeypatch.setattr(app, "validate_auth_config", lambda: None)
        monkeypatch.setattr(app, "_initialize_database_and_admit_runtime", init)
        start = app.startup_event
    else:
        from xagent.web import worker

        monkeypatch.setattr(worker, "validate_encryption_config", lambda: None)
        monkeypatch.setattr(worker, "configure_db", init)
        start = worker.run_worker
    with pytest.raises(PrivateNetworkHostError, match=variable):
        await start()
    init.assert_not_called()
