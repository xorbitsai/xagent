"""Tests for connector_auth_type() and its agreement with the HTTP mcp_oauth
runtime classifier (_is_mcp_oauth_http_server) in mcp_runtime.py.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from xagent.web.models import MCPServer
from xagent.web.services.mcp_runtime import (
    _is_mcp_oauth_http_server,
    connector_auth_type,
)


def test_connector_auth_type_no_auth_attribute_is_none():
    assert connector_auth_type(SimpleNamespace()) == "none"


@pytest.mark.parametrize(
    ("auth", "expected"),
    [
        (None, "none"),
        ({}, "none"),
        ({"type": None}, "none"),
        ({"type": "bearer"}, "bearer"),
        ({"type": "api_key"}, "api_key"),
        ({"type": "oauth2"}, "oauth2"),
        ({"type": "mcp_oauth"}, "mcp_oauth"),
        ({"type": "none"}, "none"),
        ({"type": ""}, None),
        ({"type": 3}, None),
        ("string", None),
    ],
    ids=[
        "auth-none",
        "auth-empty-dict",
        "type-none",
        "type-bearer",
        "type-api-key",
        "type-oauth2",
        "type-mcp-oauth",
        "type-literal-none",
        "type-empty-string",
        "type-non-string",
        "auth-non-mapping-string",
    ],
)
def test_connector_auth_type_classifies_declared_shapes(auth, expected):
    server = SimpleNamespace(auth=auth)
    assert connector_auth_type(server) == expected


def test_connector_auth_type_non_mapping_mock_auth_is_unrecognisable():
    # A Mock is not a dict, so this must fall into "unknown", not "none".
    server = SimpleNamespace(auth=MagicMock())
    assert connector_auth_type(server) is None


@pytest.mark.parametrize(
    "auth",
    [
        None,
        {},
        {"type": None},
        {"type": ""},
        {"type": 3},
        {"type": "bearer"},
        {"type": "api_key"},
        {"type": "oauth2"},
        {"type": "mcp_oauth"},
        "string",
    ],
    ids=[
        "auth-none",
        "auth-empty-dict",
        "type-none",
        "type-empty-string",
        "type-non-string",
        "type-bearer",
        "type-api-key",
        "type-oauth2",
        "type-mcp-oauth",
        "auth-non-mapping-string",
    ],
)
def test_connector_auth_type_agrees_with_http_oauth_classifier(auth):
    """connector_auth_type() must say "mcp_oauth" exactly when the runtime's
    own HTTP mcp_oauth classifier (_is_mcp_oauth_http_server) says so, for an
    HTTP-transport server. Both must move together or resolvers can be told
    "mcp_oauth" for a connector the runtime does not actually treat as one
    (or vice versa)."""
    server = SimpleNamespace(transport="streamable_http", auth=auth)

    # Build the second argument the way the production caller does, in
    # build_mcp_runtime_connection: decrypt the raw auth before classifying.
    # This expression mirrors the call site in mcp_runtime.py's
    # build_mcp_runtime_connection and must be kept in sync with it.
    auth_config = MCPServer._decrypt_auth_config(getattr(server, "auth", None))

    is_mcp_oauth = _is_mcp_oauth_http_server(server, auth_config)
    assert is_mcp_oauth == (connector_auth_type(server) == "mcp_oauth")
